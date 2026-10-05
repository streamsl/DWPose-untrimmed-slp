"""Overlay renderer for SAVED keypoints (spec §4.5, D2).

CPU only: it reads the saved files, never initialises CUDA or builds a model, and streams one decoded
frame at a time, so memory stays flat for any video or window length. `poses/<vid>.npy` and the
members of `persons/<vid>.npz` (record.load_persons with mmap_mode='r') are memory-mapped; only the
rows of the frame being drawn are read. Frames are piped to an ffmpeg/libx264 process (H.264 mp4).
The COCO-WholeBody-133 skeleton and colours come from mmpose's coco_wholebody metainfo.
"""
from __future__ import annotations

import errno
import functools
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import cv2
import numpy as np

from .types import NUM_KEYPOINTS, POSED, VideoInfo
from .video import VideoError, probe

if TYPE_CHECKING:
    from .record import PersonsRecord

# mmpose's dataset metainfo, relative to the mmpose package directory: a wheel ships it under .mim/; an
# editable install links .mim/configs to ../configs, a link some filesystems cannot follow (ntfs-3g reads
# the ntfs3 driver's directory symlinks as 'unsupported reparse tag'), so the source tree is the fallback.
COCO_WHOLEBODY_METAINFO = Path('configs') / '_base_' / 'datasets' / 'coco_wholebody.py'
METAINFO_ROOTS = (Path('.mim'), Path('..'))
# BGR colours of everything that is not the primary's skeleton.
OTHER_COLOR = (170, 170, 170)        # other posed people: skeleton, box and detector score
CANDIDATE_COLOR = (110, 110, 110)    # candidates that were not posed: thin boxes
PRIMARY_BOX_COLOR = (0, 215, 255)    # the primary's box and score (with `persons`)
TEXT_COLOR = (255, 255, 255)
_SHIFT = 4                           # cv2 fractional bits: shapes are drawn at sub-pixel positions
X264_OPTIONS = ('-c:v', 'libx264', '-preset', 'veryfast', '-crf', '26', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart')


@dataclass(frozen=True)
class Skeleton:
    """COCO-WholeBody-133 drawing spec, colours in BGR."""

    links: Tuple[Tuple[int, int], ...]
    link_colors: Tuple[Tuple[int, int, int], ...]
    kpt_colors: Tuple[Tuple[int, int, int], ...]


def coco_wholebody_metainfo_file() -> Path:
    """mmpose's coco_wholebody.py: the first readable copy under METAINFO_ROOTS of the mmpose package."""
    import mmpose
    package = Path(mmpose.__file__).parent
    for root in METAINFO_ROOTS:
        path = package / root / COCO_WHOLEBODY_METAINFO
        if path.is_file():
            return path
    raise FileNotFoundError(f'no readable coco_wholebody.py under {package} ({", ".join(map(str, METAINFO_ROOTS))})')


@functools.lru_cache(maxsize=None)
def coco_wholebody_skeleton() -> Skeleton:
    """Skeleton links and colours from mmpose's coco_wholebody.py (RGB there, as mmpose draws on RGB)."""
    from mmpose.datasets.datasets.utils import parse_pose_metainfo
    meta = parse_pose_metainfo(dict(from_file=str(coco_wholebody_metainfo_file())))
    if meta['num_keypoints'] != NUM_KEYPOINTS:
        raise ValueError(f"coco_wholebody metainfo has {meta['num_keypoints']} keypoints")

    def bgr(colors: np.ndarray) -> Tuple[Tuple[int, int, int], ...]:
        return tuple((int(b), int(g), int(r)) for r, g, b in colors)

    return Skeleton(links=tuple((int(a), int(b)) for a, b in meta['skeleton_links']),
                    link_colors=bgr(meta['skeleton_link_colors']), kpt_colors=bgr(meta['keypoint_colors']))


class FrameCursor:
    """Forward-only, frame-exact decoding with `video.read_frames` semantics.

    Frames are decoded sequentially from frame 0 with cv2's FFmpeg backend and never seeked (cv2
    seeking is not frame-exact for every codec). Reads must move forward, so any number of windows
    of one video costs a single decoding pass.
    """

    def __init__(self, path: Path) -> None:
        self._cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise VideoError(f'cv2 cannot open {path}')
        self._shape = (int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3)
        self.position = 0  # index of the next frame the decoder returns

    def read(self, start: int, count: int) -> np.ndarray:
        """(n,H,W,3) uint8 BGR frames start..start+count-1; n < count only at the end of the video."""
        if start < self.position:
            raise ValueError(f'FrameCursor only moves forward: frame {start} requested at position {self.position}')
        while self.position < start:
            if not self._cap.grab():
                return np.zeros((0,) + self._shape, np.uint8)
            self.position += 1
        frames = []
        for _ in range(count):
            ok, frame = self._cap.read()
            if not ok:
                break
            frames.append(frame)
            self.position += 1
        return np.stack(frames) if frames else np.zeros((0,) + self._shape, np.uint8)

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> 'FrameCursor':
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class H264Writer:
    """Pipe BGR frames of `video`'s size into ffmpeg, which writes an H.264 mp4 at `path`.

    The frame rate is the exact rational fps_num/fps_den. ffmpeg runs in its own session, so a
    terminal Ctrl-C reaches only this process, which then kill()s it; its stderr goes to a
    temporary file (a pipe could fill up and stall both processes). Every writer must end with
    close() or kill(); both reap the ffmpeg process.
    """

    def __init__(self, path: Path, video: VideoInfo) -> None:
        self.path = Path(path)
        self.frames = 0
        self._shape = (video.height, video.width, 3)
        cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-video_size', f'{video.width}x{video.height}',
               '-framerate', f'{video.fps_num}/{video.fps_den}', '-i', 'pipe:0',
               '-an', *X264_OPTIONS, '-f', 'mp4', str(self.path)]
        self._stderr = tempfile.TemporaryFile()
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                          stderr=self._stderr, start_new_session=True)
        except BaseException:
            self._stderr.close()
            raise

    def write(self, image: np.ndarray) -> None:
        """Append one (H,W,3) uint8 BGR frame."""
        if image.shape != self._shape or image.dtype != np.uint8:
            raise ValueError(f'frame is {image.dtype} {image.shape}, expected uint8 {self._shape}')
        try:
            self._proc.stdin.write(np.ascontiguousarray(image).data)
        except BrokenPipeError:
            raise RuntimeError(f'ffmpeg stopped reading after {self.frames} frames: {self._reap()}') from None
        self.frames += 1

    def close(self) -> None:
        """End the input and wait; raise RuntimeError if ffmpeg failed or printed an error."""
        try:
            self._proc.stdin.close()
        except BrokenPipeError:   # ffmpeg is gone; _reap says why
            pass
        error = self._reap()
        if error:
            raise RuntimeError(f'ffmpeg failed writing {self.path}: {error}')

    def kill(self) -> None:
        """Stop ffmpeg at once and reap it (idempotent); the partial file is the caller's."""
        if self._proc.poll() is None:
            self._proc.kill()
        self._reap()

    def _reap(self) -> str:
        """Wait for ffmpeg and release its pipes; its error, '' after a clean exit (or a second call)."""
        code = self._proc.wait()
        try:
            self._proc.stdin.close()
        except OSError:           # frames still buffered for a process that is gone
            pass
        if self._stderr.closed:
            return ''
        self._stderr.seek(0)
        text = self._stderr.read().decode(errors='replace').strip()
        self._stderr.close()
        if code:
            return f'exit status {code}: {text}' if text else f'exit status {code}'
        return text


def _point(xy: np.ndarray) -> Tuple[int, int]:
    return int(round(float(xy[0]) * (1 << _SHIFT))), int(round(float(xy[1]) * (1 << _SHIFT)))


def draw_pose(image: np.ndarray, pixels: np.ndarray, scores: np.ndarray, score_thr: float, skeleton: Skeleton,
              color: Optional[Tuple[int, int, int]] = None, unit: int = 1) -> None:
    """Draw one person's joints and limbs whose scores are >= score_thr (in place).

    pixels (133,2) original-frame pixels, scores (133,). `color` None draws the COCO-133 colours,
    otherwise everything in that BGR colour. `unit` is the line thickness; joints have radius unit + 1.
    """
    visible = (scores >= score_thr) & np.isfinite(pixels).all(1) & (np.abs(pixels) < 1e6).all(1)
    for (a, b), link_color in zip(skeleton.links, skeleton.link_colors):
        if visible[a] and visible[b]:
            cv2.line(image, _point(pixels[a]), _point(pixels[b]), link_color if color is None else color, unit,
                     cv2.LINE_AA, _SHIFT)
    for k in np.flatnonzero(visible):
        cv2.circle(image, _point(pixels[k]), (unit + 1) << _SHIFT,
                   skeleton.kpt_colors[k] if color is None else color, -1, cv2.LINE_AA, _SHIFT)


def draw_box(image: np.ndarray, box: np.ndarray, color: Tuple[int, int, int], thickness: int,
             label: Optional[str] = None, unit: int = 1) -> None:
    """Draw an xyxy box (original pixels) with an optional label above its top-left corner (in place)."""
    cv2.rectangle(image, _point(box[:2]), _point(box[2:]), color, thickness, cv2.LINE_AA, _SHIFT)
    if label:
        origin = (int(max(box[0], 0)), int(max(box[1] - 3 * unit, 10 * unit)))
        cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.35 * unit, color, 1, cv2.LINE_AA)


def _open_persons(root: Path, video: VideoInfo, num_frames: int) -> 'PersonsRecord':
    """Memory-mapped persons record of `video`, checked against the poses file and the video."""
    from .record import load_persons
    path = root / 'persons' / f'{video.video_id}.npz'
    record = load_persons(path, mmap_mode='r')
    if record.num_frames != num_frames:
        raise ValueError(f'{path}: {record.num_frames} frames, poses file has {num_frames}')
    if tuple(record.frame_size) != (video.width, video.height):
        raise ValueError(f'{path}: frame_size {record.frame_size} != video {(video.width, video.height)}')
    return record


def _draw_people(image: np.ndarray, record: 'PersonsRecord', t: int, size: np.ndarray, score_thr: float,
                 skeleton: Skeleton, unit: int) -> None:
    """Candidates, other posed people and the primary's box of frame t (the primary skeleton is drawn separately)."""
    d0, d1 = int(record.det_offsets[t]), int(record.det_offsets[t + 1])
    k0, k1 = int(record.kpt_offsets[t]), int(record.kpt_offsets[t + 1])
    boxes, scores = np.asarray(record.det_boxes[d0:d1]), np.asarray(record.det_scores[d0:d1])
    flags = np.asarray(record.det_flags[d0:d1])
    for box in boxes[(flags & POSED) == 0]:
        draw_box(image, box, CANDIDATE_COLOR, 1)
    posed = np.asarray(record.kpt_det[k0:k1], np.int64) - d0
    primary = int(record.kpt_primary[t])
    for p, row in enumerate(posed):
        if p != primary:
            kpts = np.asarray(record.kpts[k0 + p])
            draw_pose(image, kpts[:, :2] * size, kpts[:, 2], score_thr, skeleton, OTHER_COLOR, unit)
            draw_box(image, boxes[row], OTHER_COLOR, unit, f'{scores[row]:.2f}', unit)
    if primary >= 0:
        row = posed[primary]
        draw_box(image, boxes[row], PRIMARY_BOX_COLOR, unit, f'{scores[row]:.2f}', unit)


def render_video(root: Path, video: VideoInfo, out_path: Path, start_frame: int = 0,
                 num_frames: Optional[int] = None, persons: bool = False, score_thr: float = 0.3) -> Path:
    """Draw frames [start_frame, start_frame + num_frames) of `video` with their saved poses.

    - Reads <root>/poses/<vid>.npy with mmap (only the needed rows) and, with `persons`,
      <root>/persons/<vid>.npz (all posed people and every candidate box), also memory-mapped.
    - Frame-exact: frame i of the output is decoded frame start_frame + i (video.read_frames
      semantics; never trust cv2 seeking).
    - Keypoints are stored as x/W, y/H: pixel = (x * W, y * H). Joints and limbs with score
      < score_thr are hidden. Primary signer in the COCO-133 colours from the metainfo
      (skeleton_links, keypoint/link colours); with `persons`, the other posed people in grey with
      their boxes and detector scores, the primary's box in gold, non-posed candidates as thin grey
      boxes. The top-left corner shows the frame index, time and (with `persons`) num_persons.
    - Writes an H.264 mp4 (H264Writer, X264_OPTIONS) at the exact native fps fps_num/fps_den to
      <out_path>.<pid>.part (one name per process, so concurrent renders of one output never share
      a file), checks its frame count, size and fps with ffprobe, moves it to `out_path` with
      durable_replace and returns that. On any error or interrupt ffmpeg is killed and reaped and
      the .part file removed. num_frames None = to the end of the saved poses.
    """
    root, out_path = Path(root), Path(out_path)
    poses = np.load(root / 'poses' / f'{video.video_id}.npy', mmap_mode='r')
    if poses.ndim != 3 or poses.shape[1:] != (NUM_KEYPOINTS, 3) or poses.dtype != np.float32:
        raise ValueError(f'{video.video_id}: poses must be (T,133,3) float32, got {poses.dtype} {poses.shape}')
    total = poses.shape[0]
    end = total if num_frames is None else min(total, start_frame + num_frames)
    if not 0 <= start_frame < end:
        raise ValueError(f'{video.video_id}: no frames to render in [{start_frame}, {end}) of {total} saved frames')
    record = _open_persons(root, video, total) if persons else None
    skeleton = coco_wholebody_skeleton()
    size = np.array([video.width, video.height], np.float64)
    unit = max(1, round(max(video.width, video.height) / 480))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f'{out_path.name}.{os.getpid()}.part')
    writer = H264Writer(tmp, video)
    try:
        with FrameCursor(video.path) as cursor:
            for t in range(start_frame, end):
                frames = cursor.read(t, 1)
                if not len(frames):
                    raise VideoError(f'{video.video_id}: the video ends before frame {t} ({total} frames saved)')
                image = frames[0]
                if image.shape != (video.height, video.width, 3):
                    raise VideoError(f'{video.video_id}: frame {t} is {image.shape}, '
                                     f'expected {video.height}x{video.width}')
                hud = f'{t}  {t * video.fps_den / video.fps_num:.2f}s'
                if record is not None:
                    _draw_people(image, record, t, size, score_thr, skeleton, unit)
                    hud += f'  persons {int(record.num_persons[t])}'
                pose = np.asarray(poses[t])
                draw_pose(image, pose[:, :2] * size, pose[:, 2], score_thr, skeleton, unit=unit)
                cv2.putText(image, hud, (4 * unit, 14 * unit), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * unit, TEXT_COLOR, 1,
                            cv2.LINE_AA)
                writer.write(image)
        writer.close()
        _check_encoded(tmp, video, end - start_frame)
        durable_replace(tmp, out_path)
    except BaseException:
        writer.kill()
        tmp.unlink(missing_ok=True)
        raise
    return out_path


def durable_replace(src: Path, dst: Path) -> None:
    """os.replace(src, dst) with src fsynced before and dst's directory after, so that after a crash
    `dst` is either its old file or all of `src` (ffmpeg itself never fsyncs)."""
    _fsync(src)
    os.replace(src, dst)
    _fsync(Path(dst).parent)


def _fsync(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError as e:
        if e.errno != errno.EINVAL:   # some filesystems cannot fsync a directory; renames stay atomic
            raise
    finally:
        os.close(fd)


def _check_encoded(path: Path, video: VideoInfo, num_frames: int) -> None:
    """The encoded file must hold exactly the frames written, at the video's size and rational fps."""
    got = probe(path)
    have = (got.nb_frames, got.width, got.height, got.fps_num, got.fps_den)
    want = (num_frames, video.width, video.height, video.fps_num, video.fps_den)
    if have != want:
        raise RuntimeError(f'{path}: encoded (frames, width, height, fps_num, fps_den) {have}, expected {want}')
