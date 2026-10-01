"""The extraction run: probe, schedule, resume, supervise workers, merge the CSV (spec §4.4).

Runs in the parent process and never initialises CUDA. The parent owns the schedule: it keeps the
queue of VideoTasks and feeds each GPU's worker through that worker's own task queue, PREFETCH
tasks ahead (the video in progress and the next one, which the worker's reader decodes ahead).
Because the parent always knows which videos a worker holds, the videos of a crashed or hung
worker are requeued exactly. Events come back over one Pipe per worker (see worker.py).

Failure policy: a video gets MAX_ATTEMPTS attempts and every failed attempt is a line in
.state/failed.jsonl. When a worker dies, the video named by its 'fatal' event is charged an
attempt. Without a 'fatal' event (segfault, OOM kill, watchdog) the video it was processing is
charged, and each video its reader had already claimed (decoded ahead, maybe the real culprit)
becomes a suspect and goes to the back of the queue, away from its neighbour; a video suspected
in MAX_ATTEMPTS deaths is failed. The worker's other tasks go back to the front unchanged.

Final rule (spec D14): workers commit with the dataset's per-frame extraction rule; when the
dataset's primary rule differs (e.g. a video-level rule), the parent derives each commit with it on
the CPU before counting the video as done, so `extract` leaves final outputs. A video committed
but not derived yet (a stop or crash in between) has the extraction rule's derivation_hash, so the
next run derives it (record.resume_action -> 'derive').

Host RAM stays small: the parent keeps per-video bookkeeping only, never per-frame data.
"""
from __future__ import annotations

import collections
import contextlib
import csv
import dataclasses
import logging
import multiprocessing
import os
import signal
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import connection as mp_connection
from pathlib import Path
from typing import Callable, Deque, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from . import record
from .datasets import Dataset, Video, index_table, load_videos, schedule
from .settings import Settings
from .types import VideoInfo
from .worker import VideoTask, WorkerEvent, worker_main

log = logging.getLogger(__name__)

PREFETCH = 2       # tasks outstanding per worker: the video in progress + the one decoded ahead
VIDEO_INDEX = 'video_ids.csv'   # <out_root>/video_ids.csv: datasets.index_table of every video
MAX_ATTEMPTS = 2   # a failed video is requeued once
_STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)


@dataclass
class RunSummary:
    """Outcome of one `extract` run (video ids per outcome)."""

    done: List[str] = field(default_factory=list)
    derived: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    frames: int = 0
    seconds: float = 0.0
    pending: List[str] = field(default_factory=list)   # not finished: interrupted, or no GPU left


@dataclass(frozen=True)
class Timing:
    """Supervision intervals (seconds) and limits of run_extract; tests shorten them."""

    watchdog_s: float = 600.0        # a worker silent this long is killed and respawned
    respawn_limit: int = 3           # respawns per GPU within respawn_window_s, then that GPU stops
    respawn_window_s: float = 3600.0
    gpu_sample_s: float = 10.0       # nvidia-smi -> .state/gpu.csv
    progress_s: float = 60.0         # one progress line
    stop_grace_s: float = 120.0      # how long stopping workers may take before they are killed
    merge_every: int = 50            # rewrite video_meta.csv after this many commits


def plan_videos(videos: Sequence[Video], split_order: Sequence[str] = (),
                video_ids: Optional[Sequence[str]] = None, limit: Optional[int] = None,
                probe_threads: int = 8) -> List[VideoInfo]:
    """Probe (video.probe, threaded) and order a dataset's videos (datasets.load_videos).

    Order: datasets.schedule (split_order, then the largest file first). `video_ids` restricts the
    set (unknown ids raise ValueError); `limit` keeps the first N, and only those are probed. Probe
    failures (video.VideoError, e.g. VFR) are raised with the video id.
    """
    from .video import VideoError, probe
    by_id = {video.video_id: video for video in videos}
    if video_ids is not None:
        unknown = sorted(set(video_ids) - set(by_id))
        if unknown:
            raise ValueError(f'unknown video ids {unknown}')
        by_id = {vid: by_id[vid] for vid in video_ids}
    sizes = {vid: os.path.getsize(video.path) for vid, video in by_id.items()}
    order = schedule(by_id.values(), split_order, sizes)[:limit]

    def probe_one(video: Video) -> VideoInfo:
        try:
            return probe(video.path, video.video_id)
        except VideoError as e:
            raise VideoError(f'{video.video_id}: {e}') from e

    with ThreadPoolExecutor(probe_threads) as pool:
        return list(pool.map(probe_one, order))


def run_extract(dataset: Dataset, settings: Settings, out_root: Path, gpus: Sequence[int],
                video_ids: Optional[Sequence[str]] = None, limit: Optional[int] = None,
                allow_frame_mismatch: bool = False, *, timing: Timing = Timing(),
                worker_target: Callable = worker_main) -> RunSummary:
    """Extract `dataset` into `out_root` with one worker process (`worker_target`) per GPU.

    - Lists the dataset's videos (datasets.load_videos: the plugin contract is checked first).
    - Holds an exclusive lock on out_root (root_lock), clears .work/ once, writes
      <out_root>/video_ids.csv (every video of the dataset, datasets.index_table), then per video
      record.resume_action decides skip / derive (record.derive on the CPU here, interleaved with
      supervising the workers) / extract (workers use dataset.extraction_rule; the parent then
      derives the commit with dataset.primary_rule when that differs, see the module docstring).
    - Workers are started on demand (spawn context); see the module docstring for scheduling and
      the failure policy. A GPU whose worker needed more than timing.respawn_limit respawns
      within timing.respawn_window_s takes no more work. A worker silent for timing.watchdog_s
      is killed and respawned.
    - nvidia-smi every timing.gpu_sample_s to <out_root>/.state/gpu.csv; one progress line every
      timing.progress_s (videos, fps per GPU, ETA).
    - Ctrl-C or SIGTERM: stop handing out work, every worker stops after its current chunk and
      deletes its unfinished spill; workers still running after timing.stop_grace_s (or at a
      second Ctrl-C) are killed. Committed videos stay intact (a commit is never interrupted),
      the CSV is merged, then KeyboardInterrupt is re-raised.
    - merge_meta at the end and after every timing.merge_every commits.
    """
    from . import engines
    started = time.time()
    out_root = Path(out_root).resolve()
    gpus = list(gpus)
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError(f'need distinct GPU indices, got {gpus}')
    summary = RunSummary()
    all_videos = load_videos(dataset)
    with root_lock(out_root), _sigterm_as_interrupt():
        record.clear_work(out_root)
        write_video_index(out_root / VIDEO_INDEX, *index_table(all_videos, dataset.data_root))
        model_provenance = engines.model_provenance(settings)
        worker_hash = record.extraction_hash(settings, model_provenance)   # what a worker reports when ready
        extraction_hash = record.extraction_hash(settings, model_provenance, dataset.extraction_rule)
        derivation_hash = record.derivation_hash(settings, dataset.primary_rule)
        videos = plan_videos(all_videos, dataset.split_order, video_ids, limit)
        tasks, derives = [], []
        for video in videos:
            action = record.resume_action(out_root, video.video_id, extraction_hash, derivation_hash)
            if action == 'skip':
                summary.skipped.append(video.video_id)
            elif action == 'derive':
                derives.append(video.video_id)
            else:
                tasks.append(VideoTask(video, str(out_root), dataset.extraction_rule, allow_frame_mismatch))
        log.info('%s -> %s: %d videos, %d to extract (%d frames), %d to derive, %d already done',
                 dataset.name, out_root, len(videos), len(tasks), sum(t.video.nb_frames for t in tasks),
                 len(derives), len(summary.skipped))
        supervisor = _Supervisor(settings, out_root, gpus, timing, worker_target, worker_hash,
                                 dataset.primary_rule, derivation_hash, summary, len(videos))
        try:
            supervisor.run(tasks, derives)
        finally:
            summary.seconds = time.time() - started
            _log_summary(summary)
    return summary


def run_derive(out_root: Path, settings: Settings, primary_rule: str,
               video_ids: Optional[Sequence[str]] = None) -> Tuple[List[str], List[str]]:
    """CPU re-derivation (record.derive, spec D6) of committed videos, then merge_meta.

    `video_ids` defaults to every video with a done marker. Returns (derived, failed) ids; each
    failure is logged and recorded in .state/failed.jsonl.
    """
    out_root = Path(out_root).resolve()
    derived: List[str] = []
    failed: List[str] = []
    with root_lock(out_root), _sigterm_as_interrupt():
        try:
            for vid in committed_videos(out_root) if video_ids is None else video_ids:
                with deferred_interrupt():   # a Ctrl-C held during the derive is raised after the bookkeeping
                    (derived if _derive_video(out_root, vid, settings, primary_rule) else failed).append(vid)
        finally:
            with deferred_interrupt():
                merge_meta(out_root)
    return derived, failed


def write_video_index(path: Path, header: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    """Write a video index CSV (datasets.index_table) atomically: temp file, fsync, rename."""
    tmp = path.with_name(path.name + '.tmp')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(header))
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def merge_meta(out_root: Path) -> Path:
    """Write <out_root>/video_meta.csv from .state/meta/*.json (meta.load_meta_rows +
    meta.write_video_meta); returns the CSV path. Only rows of videos with a done marker."""
    from . import meta
    layout = record.OutputLayout(Path(out_root))
    rows = meta.load_meta_rows(layout.meta_json('*').parent)
    done = set(committed_videos(out_root))
    meta.write_video_meta(layout.video_meta_csv, {vid: row for vid, row in rows.items() if vid in done})
    return layout.video_meta_csv


def committed_videos(out_root: Path) -> List[str]:
    """Ids of the videos with a done marker under `out_root`, sorted."""
    pattern = record.OutputLayout(Path(out_root)).done_marker('*')
    return sorted(path.stem for path in pattern.parent.glob(pattern.name))


@contextlib.contextmanager
def root_lock(out_root: Path) -> Iterator[None]:
    """Exclusive lock on an output root: one extract / derive at a time (a run clears .work/)."""
    import fcntl
    path = Path(out_root) / '.state' / 'lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'{out_root} is in use by another slp_pose run') from None
        yield


_deferring = 0


@contextlib.contextmanager
def deferred_interrupt() -> Iterator[None]:
    """Hold Ctrl-C and SIGTERM until the block ends (then raise KeyboardInterrupt), so a commit
    is never cut in half. Nested blocks and non-main threads are no-ops."""
    global _deferring
    if _deferring or threading.current_thread() is not threading.main_thread():
        yield
        return
    caught: List[int] = []
    previous = {sig: signal.signal(sig, lambda signum, frame: caught.append(signum)) for sig in _STOP_SIGNALS}
    _deferring += 1
    try:
        yield
    finally:
        _deferring -= 1
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if caught:
            raise KeyboardInterrupt


@contextlib.contextmanager
def _sigterm_as_interrupt() -> Iterator[None]:
    """SIGTERM (kill, systemd, a closing tmux) stops the run like Ctrl-C."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _derive_video(out_root: Path, video_id: str, settings: Settings, primary_rule: str) -> bool:
    """record.derive with Ctrl-C held; a failure is logged and recorded (attempt 1, no GPU).

    A Ctrl-C held here is raised when the block ends, before this returns: callers that count the
    result call this inside their own deferred_interrupt block (nested blocks are no-ops), so the
    interrupt comes after their bookkeeping.
    """
    with deferred_interrupt():
        try:
            record.derive(out_root, video_id, settings, primary_rule)
        except Exception as exc:
            error = f'derive: {type(exc).__name__}: {exc}'
            log.error('%s: %s', video_id, error)
            record.record_failure(out_root, video_id, error, traceback.format_exc(), 1, None)
            return False
    log.info('%s: derived with %s', video_id, primary_rule)
    return True


def _duration(seconds: float) -> str:
    days, rest = divmod(int(seconds), 86400)
    text = time.strftime('%H:%M:%S', time.gmtime(rest))
    return f'{days}d {text}' if days else text


def _log_summary(s: RunSummary) -> None:
    log.info('run ended after %s: %d extracted (%d frames), %d derived, %d skipped, %d failed, %d not finished',
             _duration(s.seconds), len(s.done), s.frames, len(s.derived), len(s.skipped), len(s.failed),
             len(s.pending))
    if s.failed:
        log.error('failed videos (see .state/failed.jsonl): %s', ' '.join(s.failed))
    if s.pending:
        log.warning('not finished (the next run picks them up): %s', ' '.join(s.pending))


@dataclass
class _Slot:
    """Parent-side state of one GPU's worker; survives respawns."""

    gpu: int
    proc: Optional[multiprocessing.process.BaseProcess] = None
    tasks: Optional[object] = None                      # the worker's task queue
    conn: Optional[mp_connection.Connection] = None     # read end of its event pipe
    outstanding: 'collections.OrderedDict[str, VideoTask]' = field(default_factory=collections.OrderedDict)
    claimed: Set[str] = field(default_factory=set)      # outstanding videos its reader has taken
    fatal: Optional[WorkerEvent] = None
    last_event: float = 0.0                             # time.monotonic() of the last event or the spawn
    exiting: bool = False                               # sentinel sent
    spawns: int = 0
    respawns: Deque[float] = field(default_factory=collections.deque)
    stopped: bool = False                               # respawn budget exhausted: no more work
    video: Optional[str] = None                         # video in progress and its frames so far
    frames: int = 0
    processed: int = 0                                  # frames processed on this GPU in this run
    reported: int = 0                                   # `processed` at the last progress line


class _Supervisor:
    """The event loop of one extract run: derive, dispatch, supervise the workers, report."""

    def __init__(self, settings: Settings, out_root: Path, gpus: Sequence[int], timing: Timing, target: Callable,
                 worker_hash: str, primary_rule: str, derivation_hash: str, summary: RunSummary,
                 num_videos: int) -> None:
        self._ctx = multiprocessing.get_context('spawn')
        self._settings = settings
        self._root = out_root
        self._timing = timing
        self._target = target
        self._worker_hash = worker_hash
        self._rule = primary_rule
        self._derivation_hash = derivation_hash
        self._summary = summary
        self._num_videos = num_videos
        self._slots = [_Slot(gpu) for gpu in gpus]
        self._queue: Deque[VideoTask] = collections.deque()
        self._derives: Deque[Tuple[str, bool]] = collections.deque()   # (video, just extracted)
        self._todo: Dict[str, int] = {}          # unfinished video -> nb_frames, for the ETA
        self._suspects: Dict[str, int] = collections.Counter()
        self._commits = 0
        self._stopping = False
        self._sampler: Optional[_GpuSampler] = None
        now = time.monotonic()
        self._samples: Deque[Tuple[float, int]] = collections.deque([(now, 0)], maxlen=11)  # ETA window
        self._last_report = now

    def run(self, tasks: Sequence[VideoTask], derives: Sequence[str]) -> None:
        self._queue.extend(tasks)
        self._derives.extend((vid, False) for vid in derives)
        self._todo = {t.video.video_id: t.video.nb_frames for t in tasks}
        try:
            while self._work_left():
                self._step()
            self._stop_workers(signal_workers=False)
        except KeyboardInterrupt:
            log.warning('interrupted: each worker stops after its current chunk (Ctrl-C again kills them)')
            try:
                self._stop_workers(signal_workers=True)
            except KeyboardInterrupt:
                log.warning('second interrupt: killing the workers')
            raise
        finally:
            with deferred_interrupt():
                self._kill_workers()
                if self._sampler is not None:
                    self._sampler.stop()
                self._summary.pending = [t.video.video_id for t in self._queue] + [vid for vid, _ in self._derives]
                merge_meta(self._root)

    # ------------------------------------------------------------------ main loop
    def _work_left(self) -> bool:
        if self._derives or any(s.outstanding for s in self._slots):
            return True
        return bool(self._queue) and any(not s.stopped for s in self._slots)

    def _step(self) -> None:
        if self._derives:
            vid, extracted = self._derives.popleft()
            with deferred_interrupt():   # a Ctrl-C held during the derive is raised after the bookkeeping
                if _derive_video(self._root, vid, self._settings, self._rule):
                    (self._summary.done if extracted else self._summary.derived).append(vid)
                    self._committed()
                else:
                    self._summary.failed.append(vid)
        self._start_workers()
        self._dispatch()
        self._pump(0.0 if self._derives else 1.0)
        self._supervise()
        self._report()

    def _start_workers(self) -> None:
        if not self._queue or self._stopping:
            return
        t = self._timing
        for s in self._slots:
            if s.proc is None and not s.stopped:
                if s.spawns and not self._may_respawn(s):
                    s.stopped = True
                    log.error('GPU%d: %d respawns within %.0f s; this GPU takes no more work',
                              s.gpu, t.respawn_limit, t.respawn_window_s)
                    continue
                self._spawn(s)
        if self._sampler is None and any(s.proc is not None for s in self._slots):
            self._sampler = _GpuSampler(self._root / '.state' / 'gpu.csv', [s.gpu for s in self._slots],
                                        t.gpu_sample_s)

    def _may_respawn(self, s: _Slot) -> bool:
        now = time.monotonic()
        while s.respawns and now - s.respawns[0] >= self._timing.respawn_window_s:
            s.respawns.popleft()
        if len(s.respawns) >= self._timing.respawn_limit:
            return False
        s.respawns.append(now)
        return True

    def _spawn(self, s: _Slot) -> None:
        tasks = self._ctx.Queue()
        reader, writer = self._ctx.Pipe(duplex=False)
        proc = self._ctx.Process(target=self._target, args=(s.gpu, self._settings, tasks, writer),
                                 name=f'slp-worker-gpu{s.gpu}', daemon=True)
        proc.start()
        writer.close()   # only the worker writes; its death then reads as EOF here
        s.proc, s.tasks, s.conn = proc, tasks, reader
        s.last_event = time.monotonic()
        s.spawns += 1
        log.info('GPU%d: worker started (pid %d)%s', s.gpu, proc.pid, ', respawn' if s.spawns > 1 else '')

    def _dispatch(self) -> None:
        if self._stopping:
            return
        for depth in range(1, PREFETCH + 1):   # every worker's first task before anyone's second
            for s in self._slots:
                if self._queue and s.proc is not None and len(s.outstanding) < depth:
                    task = self._queue.popleft()
                    s.outstanding[task.video.video_id] = task
                    s.tasks.put(task)

    def _pump(self, timeout: float) -> None:
        """Wait up to `timeout` s for events or a worker exit; handle every event that arrived."""
        conns = {s.conn: s for s in self._slots if s.conn is not None}
        waitables = list(conns) + [s.proc.sentinel for s in self._slots if s.proc is not None]
        if waitables:
            for ready in mp_connection.wait(waitables, timeout):
                if ready in conns:
                    self._drain(conns[ready])

    def _drain(self, s: _Slot) -> None:
        while s.conn is not None:
            try:
                if not s.conn.poll():
                    return
                event = s.conn.recv()
            except (EOFError, OSError):   # the worker is gone; _supervise handles its exit
                s.conn.close()
                s.conn = None
                return
            s.last_event = time.monotonic()
            self._on_event(s, event)

    def _supervise(self) -> None:
        """Handle dead workers; kill a worker that was silent for longer than the watchdog."""
        now = time.monotonic()
        for s in self._slots:
            if s.proc is None:
                continue
            reason = None
            if s.proc.exitcode is None:
                silent = now - s.last_event
                if silent < self._timing.watchdog_s:
                    continue
                reason = f'watchdog: no event for {silent:.0f} s'
                log.error('GPU%d: %s; killing the worker', s.gpu, reason)
                s.proc.kill()
                s.proc.join(10)
            self._drain(s)
            self._on_death(s, reason)

    # ------------------------------------------------------------------ events
    def _on_event(self, s: _Slot, ev: WorkerEvent) -> None:
        vid = ev.video_id
        if ev.kind == 'ready':
            if ev.payload.get('extraction_hash') != self._worker_hash and not self._stopping:
                raise RuntimeError(f'GPU{s.gpu}: the worker computes a different extraction hash than the parent '
                                   f'(models, engines or settings changed?)')
            log.info('GPU%d: ready (%s)', s.gpu, ev.payload.get('gpu_name'))
        elif ev.kind == 'claimed':
            s.claimed.add(vid)
        elif ev.kind == 'started':
            s.video, s.frames = vid, 0
        elif ev.kind == 'progress':
            self._count_frames(s, vid, ev.frames)
        elif ev.kind == 'done':
            self._count_frames(s, vid, ev.frames)
            self._finish(s, vid)
            self._todo.pop(vid, None)
            self._summary.frames += ev.frames
            log.info('%s: done on GPU%d, %d frames in %.0f s (%.1f fps)', vid, s.gpu, ev.frames, ev.seconds,
                     ev.frames / max(ev.seconds, 1e-9))
            if ev.payload.get('derivation_hash') == self._derivation_hash:
                self._summary.done.append(vid)
                self._committed()
            else:   # committed with the extraction rule: derive the final rule next
                self._derives.append((vid, True))
        elif ev.kind == 'failed':
            task = self._finish(s, vid)
            if task is not None:
                self._failed_attempt(task, str(ev.payload.get('error', '')), str(ev.payload.get('traceback', '')),
                                     s.gpu)
        elif ev.kind == 'fatal':
            s.fatal = ev
            log.error('GPU%d: fatal worker error%s: %s\n%s', s.gpu, f' on {vid}' if vid else '',
                      ev.payload.get('error'), ev.payload.get('traceback'))
        elif ev.kind == 'exit':
            log.info('GPU%d: worker exited%s', s.gpu, ' (interrupted)' if ev.payload.get('interrupted') else '')

    def _count_frames(self, s: _Slot, vid: Optional[str], frames: int) -> None:
        if vid is not None and vid == s.video and frames > s.frames:
            s.processed += frames - s.frames
            s.frames = frames

    def _finish(self, s: _Slot, vid: Optional[str]) -> Optional[VideoTask]:
        if vid == s.video:
            s.video, s.frames = None, 0
        s.claimed.discard(vid)
        return s.outstanding.pop(vid, None)

    def _failed_attempt(self, task: VideoTask, error: str, tb: str, gpu: Optional[int]) -> None:
        vid = task.video.video_id
        with deferred_interrupt():
            record.record_failure(self._root, vid, error, tb, task.attempt, gpu)
        if task.attempt < MAX_ATTEMPTS:
            self._queue.appendleft(dataclasses.replace(task, attempt=task.attempt + 1))
            log.warning('%s: attempt %d failed on GPU%s: %s; requeued', vid, task.attempt, gpu, error)
        else:
            self._todo.pop(vid, None)
            self._summary.failed.append(vid)
            log.error('%s: failed after %d attempts: %s', vid, task.attempt, error)

    def _committed(self) -> None:
        self._commits += 1
        if self._commits % self._timing.merge_every == 0:
            with deferred_interrupt():
                merge_meta(self._root)

    # ------------------------------------------------------------------ worker exits
    def _on_death(self, s: _Slot, reason: Optional[str], charge: bool = True) -> None:
        """Requeue a finished worker's tasks; charge the videos that may have killed it (module docstring)."""
        code, fatal, current = s.proc.exitcode, s.fatal, s.video
        tasks, claimed = list(s.outstanding.values()), set(s.claimed)
        self._release(s)
        died = code != 0 or fatal is not None
        error = tb = ''
        if died:
            error = str(fatal.payload.get('error')) if fatal else (reason or f'worker exited with code {code}')
            tb = str(fatal.payload.get('traceback', '')) if fatal else ''
            log.error('GPU%d: worker exited with code %s', s.gpu, code)
        crashed = died and charge
        culprit = fatal.video_id if fatal is not None else current
        for task in reversed(tasks):   # back to the front of the queue in their original order
            vid = task.video.video_id
            if crashed and vid == culprit:
                self._failed_attempt(task, error, tb, s.gpu)
            elif crashed and fatal is None and vid in claimed:
                self._suspect(task, error, s.gpu)
            else:
                self._queue.appendleft(task)

    def _suspect(self, task: VideoTask, error: str, gpu: int) -> None:
        vid = task.video.video_id
        self._suspects[vid] += 1
        if self._suspects[vid] < MAX_ATTEMPTS:
            self._queue.append(task)
            log.warning('%s: was decoded ahead when the worker died; requeued at the back', vid)
            return
        with deferred_interrupt():
            record.record_failure(self._root, vid, f'decoded ahead in {self._suspects[vid]} worker deaths, '
                                  f'last: {error}', '', task.attempt, gpu)
        self._todo.pop(vid, None)
        self._summary.failed.append(vid)
        log.error('%s: failed, decoded ahead in %d worker deaths', vid, self._suspects[vid])

    def _release(self, s: _Slot) -> None:
        if s.conn is not None:
            s.conn.close()
        s.tasks.cancel_join_thread()   # items for a dead worker must not block the parent's exit
        s.tasks.close()
        if s.proc.exitcode is not None:
            s.proc.close()
        s.proc = s.conn = s.tasks = s.fatal = s.video = None
        s.outstanding.clear()
        s.claimed.clear()
        s.exiting = False
        s.frames = 0

    def _stop_workers(self, signal_workers: bool) -> None:
        """Send every worker the None sentinel (and SIGTERM: stop after the current chunk), then
        wait up to stop_grace_s for them to exit, handling their last events."""
        self._stopping = True
        for s in self._slots:
            if s.proc is not None and s.proc.exitcode is None:
                s.exiting = True
                s.tasks.put(None)
                if signal_workers:
                    s.proc.terminate()
        deadline = time.monotonic() + self._timing.stop_grace_s
        while any(s.proc is not None for s in self._slots) and time.monotonic() < deadline:
            self._pump(0.5)
            for s in self._slots:
                if s.proc is not None and s.proc.exitcode is not None:
                    self._drain(s)
                    self._on_death(s, None)

    def _kill_workers(self) -> None:
        self._stopping = True
        for s in self._slots:
            if s.proc is None:
                continue
            if s.proc.exitcode is None:
                log.warning('GPU%d: killing the worker (pid %s)', s.gpu, s.proc.pid)
                s.proc.kill()
                s.proc.join(10)
            self._drain(s)
            self._on_death(s, 'killed at shutdown', charge=False)

    # ------------------------------------------------------------------ progress
    def _report(self) -> None:
        now = time.monotonic()
        if now - self._last_report < self._timing.progress_s:
            return
        elapsed, self._last_report = now - self._last_report, now
        gpus = []
        for s in self._slots:
            fps = (s.processed - s.reported) / elapsed
            s.reported = s.processed
            gpus.append(f'GPU{s.gpu} ' + ('stopped' if s.stopped else f'{fps:.1f} fps'))
        total = sum(s.processed for s in self._slots)
        self._samples.append((now, total))
        t0, n0 = self._samples[0]
        rate = (total - n0) / (now - t0) if now > t0 else 0.0
        left = max(0, sum(self._todo.values()) - sum(s.frames for s in self._slots))
        s = self._summary
        finished = len(s.done) + len(s.derived) + len(s.skipped) + len(s.failed)
        log.info('progress: %d/%d videos (%d failed) | %s | %d frames left, ETA %s', finished, self._num_videos,
                 len(s.failed), ', '.join(gpus), left, _duration(left / rate) if rate > 0 else '?')


class _GpuSampler:
    """nvidia-smi every `interval` s appended to `path` as CSV: gpu, util_pct, memory_mib, time."""

    def __init__(self, path: Path, gpus: Sequence[int], interval: float) -> None:
        self._path = path
        self._cmd = ['nvidia-smi', '-i', ','.join(str(g) for g in gpus),
                     '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader,nounits']
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name='slp-gpu-sampler', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self._interval + 5)

    def _run(self) -> None:
        warned = False
        new = not self._path.exists()
        with open(self._path, 'a') as f:
            if new:
                f.write('gpu,util_pct,memory_mib,time\n')
            while True:
                try:
                    out = subprocess.run(self._cmd, capture_output=True, text=True, check=True,
                                         timeout=self._interval).stdout
                except FileNotFoundError:
                    log.warning('nvidia-smi not found; no GPU samples')
                    return
                except (OSError, subprocess.SubprocessError) as e:
                    if not warned:
                        log.warning('nvidia-smi failed (%s); sampling continues', e)
                        warned = True
                    out = ''
                stamp = time.strftime('%Y-%m-%dT%H:%M:%S')
                for line in out.strip().splitlines():
                    f.write(','.join(cell.strip() for cell in line.split(',')) + f',{stamp}\n')
                f.flush()
                if self._stop.wait(self._interval):
                    return
