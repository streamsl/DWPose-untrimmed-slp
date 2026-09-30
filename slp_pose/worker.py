"""One extraction process per GPU (spec §4.3): decode thread -> detect -> select -> crops -> pose -> append.

`ChunkProcessor` is the whole per-chunk GPU pipeline and is shared with parity.py, so parity
measures exactly the code that extracts. Importing this module does not import torch, so it
cannot initialise CUDA: `worker_main` sets CUDA_VISIBLE_DEVICES first (TensorRT needs one visible
GPU per process).

Protocol with the parent (run.py is the other side):
- `tasks` is this worker's own queue of VideoTask items; None ends the stream. The parent keeps
  two outstanding, so the reader decodes the next video while the current one finishes.
- `events` is the write end of a multiprocessing Pipe. Events are sent synchronously (under a
  lock, several threads send), so an event is in the pipe before the worker does anything else
  and survives a later crash.
- SIGINT / SIGTERM, or the death of the parent, request a stop: the worker stops at the next chunk
  boundary (never inside a commit), deletes the unfinished spill and sends 'exit'.
"""
from __future__ import annotations

import contextlib
import dataclasses
import multiprocessing
import os
import queue
import re
import signal
import sys
import threading
import time
import traceback
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, ContextManager, Dict, Iterator, List, Optional, Tuple

import numpy as np

from . import env
from .settings import Settings
from .types import NUM_KEYPOINTS, Chunk, ChunkDets, ChunkPoses, VideoInfo

if TYPE_CHECKING:
    from .engines import DetMeta, PoseMeta

HEARTBEAT_S = 15.0   # 'progress' interval; the parent's watchdog allows 10 minutes of silence
EXIT_FATAL = 3
_GPU_ERROR = re.compile(r'cuda|cudnn|cublas|tensorrt|nvrtc|device-side assert', re.IGNORECASE)


class ChunkProcessor:
    """GPU pipeline for one chunk; single CUDA stream, synchronous (spec §4.3 last line).

    For a chunk of F frames of size (W, H): detect on the letterbox (det_batch frames per call),
    select the posed rows (at most K per frame; POSED bits are set in place), cut their crops from
    the ORIGINAL frames into a reusable pinned buffer (with `crop_pool` threads; set
    cv2.setNumThreads(1) in the process), run the pose estimator (flip on) and assemble
    ChunkPoses. Frames with an empty pose set get no rows and primary -1 (no full-image fallback);
    F == 0 gives empty outputs. Run it under a non-default CUDA stream with TensorRT engines.
    """

    def __init__(self, settings: Settings, det_engine, pose_engine, det_meta: DetMeta, pose_meta: PoseMeta,
                 primary_rule: str, crop_pool: Optional[Executor] = None) -> None:
        from . import detpost, posepost
        self._settings = settings
        self._det_engine = det_engine
        self._det_post = detpost.DetPostprocessor(det_meta, settings)
        self._pose = posepost.PoseEstimator(pose_engine, pose_meta, settings)
        self._rule = primary_rule
        self._pool = crop_pool
        self._pin = pose_engine.device.type == 'cuda'
        self._crops = None   # (rows,384,288,3) uint8 torch buffer, pinned for CUDA engines, grown on demand

    def process(self, chunk: Chunk) -> Tuple[ChunkDets, ChunkPoses]:
        """Detections (with POSE_SET/COUNT_SET/POSED flags) and poses of `chunk`; does not release it.

        Detection and pose estimation synchronise before returning, so the caller may release the
        chunk (and this call may reuse the crop buffer) as soon as it returns.
        """
        from . import detpost, prep, select
        num_frames = chunk.num_frames
        if num_frames == 0:
            return ChunkDets.empty(0), _empty_poses(0)
        size = (chunk.video.width, chunk.video.height)
        dets = detpost.detect(self._det_engine, self._det_post, chunk.letterbox,
                              prep.letterbox_geometry(*size).scale_factor, self._settings.det_batch)
        rows, primary = select.select_chunk(dets, self._settings.max_posed, self._rule)
        frame_idx = dets.frame_of_row()[rows]
        if len(rows):
            centers, scales = prep.crop_params(dets.boxes[rows])
            crops = self._crop_buffer(len(rows))
            prep.crop_many(chunk.frames, frame_idx, centers, scales, crops.numpy(), pool=self._pool)
            kpts = self._pose(crops, centers, scales, size)
        else:
            kpts = np.zeros((0, NUM_KEYPOINTS, 3), np.float32)
        offsets = np.zeros(num_frames + 1, np.int64)
        np.cumsum(np.bincount(frame_idx, minlength=num_frames), out=offsets[1:])
        return dets, ChunkPoses(offsets, rows.astype(np.int32), kpts, primary)

    def _crop_buffer(self, rows: int):
        import torch
        from .prep import POSE_INPUT_SIZE
        if self._crops is None or len(self._crops) < rows:
            step = self._settings.pose_batch
            width, height = POSE_INPUT_SIZE
            self._crops = torch.empty((-(-rows // step) * step, height, width, 3), dtype=torch.uint8,
                                      pin_memory=self._pin)
        return self._crops[:rows]


def _empty_poses(num_frames: int) -> ChunkPoses:
    return ChunkPoses(np.zeros(num_frames + 1, np.int64), np.zeros(0, np.int32),
                      np.zeros((0, NUM_KEYPOINTS, 3), np.float32), np.full(num_frames, -1, np.int8))


@dataclass(frozen=True)
class VideoTask:
    """One unit of work sent by the parent to a worker (picklable)."""

    video: VideoInfo
    out_root: str               # absolute output root
    primary_rule: str
    allow_frame_mismatch: bool = False
    attempt: int = 1


@dataclass(frozen=True)
class WorkerEvent:
    """Worker -> parent message (picklable).

    kind:
      'ready'     engines loaded; payload {'gpu_name', 'extraction_hash'}
      'claimed'   the reader took the video from the task queue and starts decoding it
      'started'   a video's first chunk reached the GPU pipeline
      'progress'  heartbeat, every HEARTBEAT_S s while the worker makes progress or waits for
                  work (a hung worker goes silent); frames = frames done in the current video
      'done'      payload = dataclasses.asdict(record.CommitInfo)
      'failed'    per-video failure, worker continues; payload {'error', 'traceback', 'attempt'}
      'fatal'     the worker cannot continue (CUDA/TensorRT error, failed setup); payload
                  {'error', 'traceback'}; video_id = the video in progress; exit code 3
      'exit'      clean exit after the None sentinel or a stop request; payload {'interrupted'}
    """

    kind: str
    gpu: int
    video_id: Optional[str] = None
    frames: int = 0
    seconds: float = 0.0        # wall time of the video so far
    time: float = 0.0           # time.time() when sent
    payload: Dict[str, object] = field(default_factory=dict)


def is_gpu_error(exc: BaseException) -> bool:
    """CUDA / cuDNN / cuBLAS / TensorRT failures (incl. CUDA OOM, engine RuntimeErrors).

    After one of these the process's CUDA state is suspect, so the worker exits and the parent
    respawns it instead of carrying on with the next video (spec §4.4).
    """
    return isinstance(exc, RuntimeError) and bool(_GPU_ERROR.search(str(exc)))


class Pipeline:
    """What the worker loop needs: a ChunkProcessor per primary rule, and the provenance.

    `make_processor(rule)` builds a processor (anything with `process(chunk)`); processors and
    provenance dicts are cached per rule.
    """

    def __init__(self, settings: Settings, make_processor: Callable[[str], object],
                 model_provenance: Dict[str, object], gpu_name: str) -> None:
        from . import record
        self.settings = settings
        self.gpu_name = gpu_name
        self.model_provenance = model_provenance
        self.extraction_hash = record.extraction_hash(settings, model_provenance)
        self._make_processor = make_processor
        self._processors: Dict[str, object] = {}
        self._provenance: Dict[str, Dict[str, object]] = {}

    def processor(self, rule: str):
        if rule not in self._processors:
            self._processors[rule] = self._make_processor(rule)
        return self._processors[rule]

    def provenance(self, rule: str) -> Dict[str, object]:
        from . import record
        if rule not in self._provenance:
            self._provenance[rule] = record.make_provenance(self.settings, rule, self.model_provenance,
                                                            self.gpu_name)
        return self._provenance[rule]


@contextlib.contextmanager
def _gpu_pipeline(settings: Settings) -> Iterator[Pipeline]:
    """Worker setup in the contract order; the loop then runs inside a non-default CUDA stream."""
    import cv2
    import torch
    from . import engines
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    env.strict_fp32()
    env.check_environment(need_trt=settings.backend == 'trt')
    cv2.setNumThreads(1)
    det_engine, pose_engine = engines.load_engines(settings, device)
    det_meta, pose_meta = engines.det_meta(settings), engines.pose_meta(settings)
    model_provenance = engines.model_provenance(settings)
    # TensorRT on the legacy default stream adds a stream synchronisation per call.
    with ThreadPoolExecutor(settings.crop_threads, thread_name_prefix='slp-crop') as pool, \
            torch.cuda.stream(torch.cuda.Stream(device)):
        yield Pipeline(settings, lambda rule: ChunkProcessor(settings, det_engine, pose_engine, det_meta, pose_meta,
                                                             rule, pool),
                       model_provenance, env.gpu_name(0))


class _EventSender:
    """Sends WorkerEvents from any thread; a broken pipe (the parent is gone) requests a stop."""

    def __init__(self, conn, gpu: int, stop: threading.Event) -> None:
        self._conn = conn
        self._gpu = gpu
        self._stop = stop
        self._lock = threading.Lock()

    def __call__(self, kind: str, video_id: Optional[str] = None, frames: int = 0, seconds: float = 0.0,
                 payload: Optional[Dict[str, object]] = None) -> None:
        event = WorkerEvent(kind, self._gpu, video_id, frames, seconds, time.time(), payload or {})
        with self._lock:
            try:
                self._conn.send(event)
            except OSError:
                self._stop.set()


class _TaskFeed:
    """VideoTasks from the parent, handed to the ChunkReader thread one video at a time.

    Remembers the tasks handed out and not yet finished, so a failure can be attributed.
    """

    def __init__(self, tasks, send: _EventSender, stop: threading.Event) -> None:
        self._tasks = tasks
        self._send = send
        self._stop = stop
        self._next: Optional[VideoTask] = None   # taken from the queue, not yet handed out
        self._ended = False
        self._claimed: Dict[str, VideoTask] = {}
        self._lock = threading.Lock()
        self.waiting = False                     # blocked on an empty queue (idle, for the heartbeat)

    def peek(self) -> Optional[VideoTask]:
        """The next task without handing it out; None at the end of the stream or on a stop request."""
        if self._next is None and not self._ended:
            self._next = self._get()
        return self._next

    def videos(self, allow_frame_mismatch: bool) -> Iterator[VideoInfo]:
        """The input of one ChunkReader: videos of consecutive tasks that share `allow_frame_mismatch`
        (a reader has a single setting); runs in the reader's decode thread."""
        while True:
            task = self.peek()
            if task is None or self._stop.is_set() or task.allow_frame_mismatch != allow_frame_mismatch:
                return
            self._next = None
            with self._lock:
                self._claimed[task.video.video_id] = task
            self._send('claimed', task.video.video_id, payload={'attempt': task.attempt})
            yield task.video

    def task(self, video_id: str) -> VideoTask:
        with self._lock:
            return self._claimed[video_id]

    def finish(self, video_id: str) -> None:
        with self._lock:
            self._claimed.pop(video_id, None)

    def unfinished(self) -> List[VideoTask]:
        with self._lock:
            return list(self._claimed.values())

    def _get(self) -> Optional[VideoTask]:
        self.waiting = True
        try:
            while not self._stop.is_set():
                try:
                    task = self._tasks.get(timeout=0.5)
                except queue.Empty:
                    continue
                self._ended = task is None
                return task
            return None
        finally:
            self.waiting = False


class _VideoLoop:
    """The worker's main loop: chunk -> processor -> spill -> commit, one video after another."""

    def __init__(self, settings: Settings, pipeline: Pipeline, send: _EventSender, feed: _TaskFeed,
                 stop: threading.Event) -> None:
        self._settings = settings
        self._pipeline = pipeline
        self._send = send
        self._feed = feed
        self._stop = stop
        self._task: Optional[VideoTask] = None   # video in progress, its spill and start time
        self._spill = None
        self._t0 = 0.0
        self._dropped: set = set()               # failed videos whose remaining chunks are skipped
        self.frames = 0                          # frames appended for the video in progress
        self.activity = 0                        # bumped per chunk and commit (heartbeat liveness)
        self.busy = False                        # inside a chunk

    @property
    def video_id(self) -> Optional[str]:
        task = self._task
        return task.video.video_id if task is not None else None

    @property
    def seconds(self) -> float:
        return time.time() - self._t0 if self._task is not None else 0.0

    def run(self) -> None:
        """Process tasks until the None sentinel or a stop request."""
        while not self._stop.is_set():
            first = self._feed.peek()
            if first is None:
                break
            self._read(first.allow_frame_mismatch)
        self.abort()

    def abort(self) -> None:
        """Drop the video in progress (stop request or fatal error); committed outputs are untouched."""
        spill, self._spill, self._task = self._spill, None, None
        if spill is not None:
            spill.abort()

    def _read(self, allow_frame_mismatch: bool) -> None:
        """One ChunkReader (its own scope, so its pinned slots are freed before a next one)."""
        from .video import ChunkReader
        with ChunkReader(self._feed.videos(allow_frame_mismatch), self._settings,
                         allow_frame_mismatch=allow_frame_mismatch) as reader:
            try:
                self._consume(reader)
            except BaseException:
                self._stop.set()   # unblocks the decode thread's task wait, so closing the reader is quick
                raise

    def _consume(self, reader) -> None:
        while not self._stop.is_set():
            try:
                chunk = next(reader)
            except StopIteration:
                return
            except Exception as exc:
                # The decode thread died; every chunk it decoded before has been handled, so the
                # failure belongs to the videos still open (normally the one being decoded).
                tb = traceback.format_exc()
                for task in self._feed.unfinished():
                    self._fail(task, _describe(exc), tb)
                self._dropped.clear()   # their remaining chunks will never arrive
                return
            self.busy = True
            self.activity += 1
            try:
                self._on_chunk(chunk)
            finally:
                chunk.release()
                self.busy = False

    def _on_chunk(self, chunk: Chunk) -> None:
        vid = chunk.video.video_id
        if vid in self._dropped:
            if chunk.last:
                self._dropped.discard(vid)
            return
        task = self._feed.task(vid)
        if chunk.error is not None:
            self._fail(task, chunk.error, '')
            return
        try:
            if self._task is not task:
                self._begin(task)
            if chunk.num_frames:
                dets, poses = self._pipeline.processor(task.primary_rule).process(chunk)
                self._spill.append(chunk.start, dets, poses)
                self.frames += chunk.num_frames
            if chunk.last:
                self._commit()
        except Exception as exc:
            if is_gpu_error(exc):
                raise
            self._fail(task, _describe(exc), traceback.format_exc())
            if not chunk.last:
                self._dropped.add(vid)

    def _begin(self, task: VideoTask) -> None:
        from . import record
        if self._task is not None:   # the reader always ends a video first; never leave one open
            self._fail(self._task, 'a new video started before this one ended', '')
        self._spill = record.SpillWriter(Path(task.out_root), task.video, task.primary_rule)
        self._task, self._t0, self.frames = task, time.time(), 0
        self._send('started', task.video.video_id, payload={'attempt': task.attempt})

    def _commit(self) -> None:
        task = self._task
        info = self._spill.finalize(self._settings, self._pipeline.provenance(task.primary_rule))
        seconds = self.seconds
        self._task = self._spill = None
        self._feed.finish(task.video.video_id)
        self.activity += 1
        self._send('done', task.video.video_id, info.num_frames, seconds, dataclasses.asdict(info))

    def _fail(self, task: VideoTask, error: str, tb: str) -> None:
        frames, seconds = (self.frames, self.seconds) if task is self._task else (0, 0.0)
        if task is self._task:
            self.abort()
        self._feed.finish(task.video.video_id)
        self._send('failed', task.video.video_id, frames, seconds,
                   dict(error=error, traceback=tb, attempt=task.attempt))


def _describe(exc: BaseException) -> str:
    return f'{type(exc).__name__}: {exc}'


class _Heartbeat:
    """Background 'progress' events for the parent's watchdog.

    A beat goes out every `interval` s while the loop makes progress or waits for work, so a hung
    loop (CUDA hang, stuck decoder) goes silent and the parent restarts the worker, while an idle
    worker stays alive. It also requests a stop when the parent process has died.
    """

    def __init__(self, send: _EventSender, loop: _VideoLoop, feed: _TaskFeed, stop: threading.Event,
                 interval: float) -> None:
        self._send = send
        self._loop = loop
        self._feed = feed
        self._stop = stop
        self._interval = interval
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, name='slp-heartbeat', daemon=True)

    def __enter__(self) -> '_Heartbeat':
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._done.set()
        self._thread.join()

    def _run(self) -> None:
        parent = multiprocessing.parent_process()
        seen = -1
        while not self._done.wait(self._interval):
            if parent is not None and not parent.is_alive():
                self._stop.set()
            activity = self._loop.activity
            if activity != seen or (self._feed.waiting and not self._loop.busy):
                seen = activity
                self._send('progress', self._loop.video_id, self._loop.frames, self._loop.seconds)


@contextlib.contextmanager
def _stop_on_signals(stop: threading.Event) -> Iterator[None]:
    """SIGINT / SIGTERM set `stop` instead of interrupting (a commit must never be cut in half)."""
    previous = {sig: signal.signal(sig, lambda signum, frame: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run_worker(gpu: int, settings: Settings, tasks, events, setup: Callable[[Settings], ContextManager[Pipeline]],
               heartbeat_s: float = HEARTBEAT_S) -> None:
    """The worker body once the CUDA environment is pinned (worker_main without the GPU setup).

    `setup(settings)` is a context manager yielding the Pipeline: the GPU one in worker_main, a
    fake one in tests. Any exception outside the per-video handling (setup, CUDA/TensorRT) sends
    'fatal' and ends the process with os._exit(EXIT_FATAL): the CUDA state may be broken, so no
    Python teardown runs; the event is already in the pipe.
    """
    stop = threading.Event()
    send = _EventSender(events, gpu, stop)
    feed = _TaskFeed(tasks, send, stop)
    loop: Optional[_VideoLoop] = None
    with _stop_on_signals(stop):
        try:
            with setup(settings) as pipeline:
                send('ready', payload=dict(gpu_name=pipeline.gpu_name, extraction_hash=pipeline.extraction_hash))
                loop = _VideoLoop(settings, pipeline, send, feed, stop)
                with _Heartbeat(send, loop, feed, stop, heartbeat_s):
                    loop.run()
        except Exception as exc:
            video_id = loop.video_id if loop is not None else None
            if loop is not None:
                with contextlib.suppress(Exception):
                    loop.abort()
            send('fatal', video_id, payload=dict(error=_describe(exc), traceback=traceback.format_exc()))
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(EXIT_FATAL)
    send('exit', payload={'interrupted': stop.is_set()})


def worker_main(gpu: int, settings: Settings, tasks, events) -> None:
    """Process entry point (multiprocessing 'spawn' context), one per GPU.

    Pins the process to physical GPU `gpu` (CUDA_VISIBLE_DEVICES, PCI bus order) before anything
    can start CUDA, then: device cuda:0, env.strict_fp32(), env.check_environment, cv2
    single-threaded, engines.load_engines, one ChunkProcessor per primary rule with a
    settings.crop_threads crop pool, and one video.ChunkReader fed from `tasks`, so the next video
    is decoded while the current one finishes. Per video: record.SpillWriter -> append per chunk
    -> release -> finalize on the last chunk -> 'done'. An error chunk or any non-CUDA exception
    aborts the spill and sends 'failed'; CUDA/TensorRT errors send 'fatal' and exit with code 3.
    `tasks`: this worker's multiprocessing queue of VideoTask (None ends it); `events`: the write
    end of a multiprocessing Pipe (see the module docstring).
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env.configure_process()
    run_worker(gpu, settings, tasks, events, _gpu_pipeline)
