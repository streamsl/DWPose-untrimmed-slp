"""Video probing (ffprobe) and the threaded chunk reader (spec §4.3 step 1).

The reader decodes every frame with cv2's FFmpeg backend (BGR, native fps, no cropping),
letterboxes each frame on the CPU into a pinned buffer and hands out `Chunk`s. One background
thread walks a sequence of videos, so the next video is prefetched while the current one's last
chunks are still being processed. Host memory is bounded by (queue_depth + max_held) chunk slots.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Iterator, Optional

import cv2
import numpy as np
import torch

from .prep import DET_SIZE, letterbox, letterbox_geometry
from .settings import Settings
from .types import Chunk, VideoInfo


class VideoError(RuntimeError):
    """The video cannot be processed as-is (unreadable, VFR, missing frame count)."""


def probe(path: Path, video_id: Optional[str] = None) -> VideoInfo:
    """ffprobe the first video stream: size, exact rational fps, nb_frames; refuse VFR input."""
    path = Path(path)
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
           'stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames', '-of', 'json', str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        raise VideoError(f'ffprobe failed on {path}: {e}') from e
    streams = json.loads(out).get('streams') or []
    if not streams:
        raise VideoError(f'{path}: no video stream')
    s = streams[0]
    try:
        r_rate, avg_rate = Fraction(s['r_frame_rate']), Fraction(s['avg_frame_rate'])
        nb_frames = int(s['nb_frames'])
    except (KeyError, ValueError, ZeroDivisionError) as e:
        raise VideoError(f'{path}: missing frame rate or nb_frames in {s}') from e
    if r_rate <= 0 or r_rate != avg_rate:
        raise VideoError(f'{path}: variable frame rate (r_frame_rate {r_rate}, avg_frame_rate {avg_rate})')
    return VideoInfo(path=path, video_id=video_id or path.stem, width=int(s['width']), height=int(s['height']),
                     fps_num=r_rate.numerator, fps_den=r_rate.denominator, nb_frames=nb_frames,
                     codec=str(s.get('codec_name', '')), size_bytes=os.path.getsize(path))


def chunk_frames(width: int, height: int, settings: Settings) -> int:
    """Frames per chunk from the byte budget (original + letterbox bytes per frame).

    Capped at max_chunk_frames and rounded down to a multiple of det_batch when at least one full
    detector batch fits: 256 at 444x444, 128 at 1280x720, 64 at 1920x1080 with the defaults.
    """
    per_frame = width * height * 3 + DET_SIZE * DET_SIZE * 3
    n = min(settings.max_chunk_frames, max(1, settings.chunk_bytes // per_frame))
    if n >= settings.det_batch:
        n -= n % settings.det_batch
    return n


def read_frames(path: Path, start: int, count: int) -> np.ndarray:
    """(n,H,W,3) uint8 BGR frames start..start+count-1, n <= count at the end of the video.

    Frame-exact: decodes sequentially from frame 0 (cv2 seeking is not frame-exact for every
    codec), so it costs O(start) decoding.
    """
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise VideoError(f'cv2 cannot open {path}')
    empty = np.zeros((0, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3),
                     np.uint8)
    frames = []
    try:
        for _ in range(start):
            if not cap.grab():
                return empty
        for _ in range(count):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return np.stack(frames) if frames else empty


def chunk_from_frames(video: VideoInfo, start: int, frames: np.ndarray, last: bool = True) -> Chunk:
    """A standalone (non-pooled) Chunk for already-decoded frames, e.g. parity windows and tests."""
    geom = letterbox_geometry(video.width, video.height)
    boxes = torch.empty((len(frames), DET_SIZE, DET_SIZE, 3), dtype=torch.uint8,
                        pin_memory=torch.cuda.is_available())
    view = boxes.numpy()
    for i, frame in enumerate(frames):
        letterbox(frame, geom, view[i])
    return Chunk(video, start, np.ascontiguousarray(frames), boxes, last=last)


class _Slot:
    """Reusable buffers for one chunk: originals (grown on demand) and a pinned letterbox."""

    def __init__(self, max_frames: int, pin: bool) -> None:
        self.letterbox = torch.empty((max_frames, DET_SIZE, DET_SIZE, 3), dtype=torch.uint8, pin_memory=pin)
        self._frames = np.empty(0, np.uint8)

    def frames(self, n: int, height: int, width: int) -> np.ndarray:
        need = n * height * width * 3
        if self._frames.size < need:
            self._frames = np.empty(need, np.uint8)
        return self._frames[:need].reshape(n, height, width, 3)


_END = object()


class ChunkReader:
    """Iterate `Chunk`s of `videos` in order, decoded by a background thread.

    Every yielded chunk must be `release()`d once its buffers are no longer needed; the reader
    blocks when all slots are held. Frame-count rule (spec §4.4): the decoded frame count must equal
    ffprobe nb_frames, otherwise the video ends with an error chunk, unless `allow_frame_mismatch`
    (then every decodable frame is read and T is whatever was decoded).
    Use as a context manager, or call `close()`, to stop the thread early.
    """

    def __init__(self, videos: Iterable[VideoInfo], settings: Settings, allow_frame_mismatch: bool = False,
                 max_held: int = 1) -> None:
        self._videos = iter(videos)
        self._settings = settings
        self._allow_mismatch = allow_frame_mismatch
        pin = torch.cuda.is_available()
        self._free: 'queue.Queue[_Slot]' = queue.Queue()
        for _ in range(settings.queue_depth + max_held):
            self._free.put(_Slot(settings.max_chunk_frames, pin))
        self._out: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._done = False
        self._thread = threading.Thread(target=self._run, name='slp-decode', daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ consumer side
    def __iter__(self) -> Iterator[Chunk]:
        return self

    def __next__(self) -> Chunk:
        if self._done:
            raise StopIteration
        item = self._out.get()
        if item is _END:
            self._done = True
            raise StopIteration
        if isinstance(item, BaseException):
            self._done = True
            raise item
        return item

    def close(self) -> None:
        """Stop decoding and wait for the thread (chunks already handed out stay valid)."""
        self._stop.set()
        self._thread.join(timeout=30)

    def __enter__(self) -> 'ChunkReader':
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -------------------------------------------------------------- decode thread
    def _run(self) -> None:
        try:
            for video in self._videos:
                if self._stop.is_set():
                    break
                self._decode(video)
        except BaseException as e:  # surfaced to the consumer by __next__
            self._out.put(e)
        finally:
            self._out.put(_END)

    def _take_slot(self) -> Optional[_Slot]:
        while not self._stop.is_set():
            try:
                return self._free.get(timeout=0.1)
            except queue.Empty:
                continue
        return None

    def _fail(self, video: VideoInfo, message: str) -> None:
        empty = np.zeros((0, video.height, video.width, 3), np.uint8)
        self._out.put(Chunk(video, 0, empty, torch.empty((0, DET_SIZE, DET_SIZE, 3), dtype=torch.uint8),
                            last=True, error=message))

    def _decode(self, video: VideoInfo) -> None:
        if video.nb_frames <= 0 and not self._allow_mismatch:
            self._fail(video, f'{video.video_id}: ffprobe reports no frames')
            return
        geom = letterbox_geometry(video.width, video.height)
        n_chunk = chunk_frames(video.width, video.height, self._settings)
        cap = cv2.VideoCapture(str(video.path), cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_N_THREADS, self._settings.decode_threads])
        if not cap.isOpened():
            self._fail(video, f'{video.video_id}: cv2 cannot open {video.path}')
            return
        try:
            start = 0
            while True:
                slot = self._take_slot()
                if slot is None:
                    return
                frames = slot.frames(n_chunk, video.height, video.width)
                boxes = slot.letterbox.numpy()
                want = n_chunk if self._allow_mismatch else min(n_chunk, video.nb_frames - start)
                k, error = 0, None
                while k < want:
                    view = frames[k]
                    ok, img = cap.read(view)
                    if not ok:
                        break
                    if img is not view:  # cv2 reallocates only when the decoded size differs
                        error = f'{video.video_id}: frame {start + k} has shape {img.shape}, ffprobe says ' \
                                f'{video.height}x{video.width}'
                        break
                    letterbox(view, geom, boxes[k])
                    k += 1
                end, eof = start + k, k < want
                last = eof
                if error is None and not self._allow_mismatch:
                    if eof:
                        error = f'{video.video_id}: decoded {end} frames, ffprobe nb_frames={video.nb_frames}'
                    elif end == video.nb_frames:
                        last = True
                        if cap.read()[0]:
                            error = f'{video.video_id}: more than ffprobe nb_frames={video.nb_frames} frames'
                if error is not None:
                    self._free.put(slot)
                    self._fail(video, error)
                    return
                self._out.put(Chunk(video, start, frames[:k], slot.letterbox[:k], last=last,
                                    _release=lambda s=slot: self._free.put(s)))
                if last:
                    return
                start = end
        finally:
            cap.release()
