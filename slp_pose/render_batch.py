"""Full-length overlay videos of every committed video (spec §4.5, the `render-done` command).

Read-only with respect to the output root: it lists .state/done/, reads poses/ and persons/ of
committed videos, and probes the lock without creating it, so it can run beside a live extraction.
Each video is rendered by its own spawned process (render.render_video with persons, frame 0 to
the end) at a lowered CPU priority and with the GPUs hidden, into <vis_dir>/<vid>.mp4, followed by
its render record <vis_dir>/.state/rendered/<vid>.json: the stamp of the done marker and the
options it was rendered from, and the mp4's inode and size. An mp4 is up to date only while its
record equals the current marker's stamp, the options and the mp4 itself (never an mtime
comparison, so neither clock skew nor the marker's staging-time mtime can make a stale render look
fresh, and an old record never vouches for a new mp4); otherwise it is rendered again.
One render-done at a time per visualization directory: it holds <vis_dir>/.state/lock.
Importing this module does not import torch or cv2; only the render processes do.
"""
from __future__ import annotations

import contextlib
import json
import logging
import multiprocessing
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from multiprocessing import connection as mp_connection
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from .record import OutputLayout
from .run import committed_videos, root_lock

log = logging.getLogger(__name__)

DEFAULT_JOBS = 3
POLL_S = 300.0     # --follow: seconds between looks for new commits while a render slot is free
NICENESS = 10      # added to each render process's nice value
_JOIN_S = 30.0     # how long a stopped render process may take to clean up before it is killed
_PR_SET_PDEATHSIG = 1   # linux/prctl.h
RECORDS = Path('.state') / 'rendered'   # under the vis dir: <vid>.json render records

Stamp = Tuple[int, int]   # (st_ino, st_mtime_ns) of a done marker: changes whenever it is rewritten


@dataclass(frozen=True)
class RenderTask:
    """One full-length render, sent to its process."""

    video_id: str
    video_path: Path
    root: Path
    out_path: Path
    record_path: Path   # its render record, written once the mp4 is in place (and fsynced)
    persons: bool
    marker: Stamp       # the done marker when the task was made


@dataclass(frozen=True)
class RenderResult:
    """What a render process sends back; `error` None = rendered."""

    video_id: str
    frames: int = 0
    seconds: float = 0.0
    size_bytes: int = 0
    error: Optional[str] = None


@dataclass
class RenderSummary:
    """Outcome of one render_committed run (video ids per outcome)."""

    rendered: List[str] = field(default_factory=list)
    up_to_date: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    frames: int = 0
    size_bytes: int = 0
    seconds: float = 0.0


@dataclass
class _Job:
    task: RenderTask
    proc: multiprocessing.process.BaseProcess
    conn: mp_connection.Connection


def default_vis_dir(out_root: Path) -> Path:
    """<out_root name>_vis next to the output root, e.g. data/BOBSL/dwpose -> data/BOBSL/dwpose_vis."""
    out_root = Path(out_root)
    return out_root.with_name(out_root.name + '_vis')


def check_dirs(out_root: Path, vis_dir: Path) -> None:
    """Raise ValueError unless `vis_dir` lies outside `out_root` (the root is never written)."""
    out_root, vis_dir = Path(out_root).resolve(), Path(vis_dir).resolve()
    if vis_dir == out_root or out_root in vis_dir.parents:
        raise ValueError(f'the visualization directory {vis_dir} must lie outside the output root {out_root}')


def extraction_running(out_root: Path) -> bool:
    """True while an extract or derive run holds <out_root>/.state/lock (run.root_lock).

    Probes with a NON-BLOCKING SHARED flock that is released at once: the probe never waits, never
    creates the lock file and never disturbs a run that already holds the exclusive lock. (A run
    starting within the same microseconds would see the root as busy and refuse to start.)
    """
    import fcntl
    try:
        fd = os.open(Path(out_root) / '.state' / 'lock', os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)   # also releases the shared lock if it was taken
    return False


def render_committed(out_root: Path, vis_dir: Path, video_paths: Mapping[str, Path], jobs: int = DEFAULT_JOBS,
                     video_ids: Optional[Sequence[str]] = None, limit: Optional[int] = None, follow: bool = False,
                     persons: bool = True, poll_s: float = POLL_S) -> RenderSummary:
    """Render every committed video of `out_root` (or of `video_ids`) to <vis_dir>/<vid>.mp4.

    - Holds <vis_dir>/.state/lock for the whole run (RuntimeError if another run holds it) and
      first deletes the .part files that killed renders left behind.
    - Skips a video whose render record matches its current done marker, `persons` and mp4; renders
      at most `limit` videos; up to `jobs` render processes at a time, each of which is stopped
      when this process dies. `video_paths` maps ids to source videos.
    - A failed video is logged and skipped (retried only once its done marker changes); a render
      whose done marker changed meanwhile is discarded and counts as failed.
    - Without `follow`, returns once nothing is left to render. With `follow`, looks for new
      commits every `poll_s` s while a render slot is free and returns once nothing is left and
      no extraction holds the root (extraction_running is sampled before each listing, so a video
      committed just before the extraction ended is still rendered).
    - Ctrl-C or SIGTERM stops the render processes (each kills its ffmpeg and deletes its .part
      file), then KeyboardInterrupt is re-raised.
    """
    if jobs < 1 or poll_s <= 0:
        raise ValueError(f'need jobs >= 1 and poll_s > 0, got {jobs} and {poll_s}')
    check_dirs(out_root, vis_dir)
    out_root, vis_dir = Path(out_root).resolve(), Path(vis_dir).resolve()
    with root_lock(vis_dir), sigterm_as_interrupt():   # the lock file of an output root, on the vis dir
        (vis_dir / RECORDS).mkdir(parents=True, exist_ok=True)
        _remove_stale_parts(vis_dir)
        started = time.monotonic()
        batch = _Batch(out_root, vis_dir, video_paths, jobs, video_ids, limit, persons)
        log.info('render-done: %s -> %s, %d committed videos, %d jobs%s', out_root, vis_dir,
                 len(committed_videos(out_root)), jobs, ', following the extraction' if follow else '')
        try:
            while True:
                active = follow and extraction_running(out_root)
                batch.start(batch.todo())
                if batch.running:
                    batch.collect(poll_s if follow else None)
                elif active and not batch.limit_reached:
                    time.sleep(poll_s)
                else:
                    break
        except KeyboardInterrupt:
            log.warning('interrupted: stopping %d render processes', len(batch.running))
            raise
        finally:
            _stop(batch.running.values())
            batch.summary.seconds = time.monotonic() - started
            _log_summary(batch.summary, batch.layout, video_ids)
    return batch.summary


# --------------------------------------------------------------------------- parent side
class _Batch:
    """Bookkeeping of one render_committed run."""

    def __init__(self, out_root: Path, vis_dir: Path, video_paths: Mapping[str, Path], jobs: int,
                 video_ids: Optional[Sequence[str]], limit: Optional[int], persons: bool) -> None:
        self.layout = OutputLayout(out_root)
        self.summary = RenderSummary()
        self.running: Dict[str, _Job] = {}
        self._vis_dir = vis_dir
        self._video_paths = video_paths
        self._jobs = jobs
        self._video_ids = video_ids
        self._limit = limit
        self._persons = persons
        self._launched = 0
        self._failed_at: Dict[str, Stamp] = {}   # failed video -> its done marker at the time
        self._settled: Set[str] = set()          # rendered or found up to date in this run

    @property
    def limit_reached(self) -> bool:
        return self._limit is not None and self._launched >= self._limit

    def todo(self) -> List[Tuple[str, Stamp]]:
        """(video, done marker) of the committed videos that need a render and have none running;
        records the up-to-date ones."""
        out = []
        for vid in committed_videos(self.layout.root) if self._video_ids is None else self._video_ids:
            marker = _stamp(self.layout.done_marker(vid))
            if vid in self.running or marker is None or self._failed_at.get(vid) == marker:
                continue
            if not self._is_fresh(vid, marker):
                out.append((vid, marker))
            elif vid not in self._settled:
                self._settled.add(vid)
                self.summary.up_to_date.append(vid)
        return out

    def start(self, todo: Sequence[Tuple[str, Stamp]]) -> None:
        """Launch renders of `todo` in order while a slot is free and the limit allows."""
        for vid, marker in todo:
            if len(self.running) >= self._jobs or self.limit_reached:
                return
            self._launched += 1
            if vid not in self._video_paths:
                self._finish(RenderResult(vid, error='no such video in the dataset'), marker)
                continue
            self.running[vid] = _launch(RenderTask(vid, Path(self._video_paths[vid]), self.layout.root,
                                                   self._vis_dir / f'{vid}.mp4', _record_path(self._vis_dir, vid),
                                                   self._persons, marker))

    def collect(self, timeout: Optional[float]) -> None:
        """Wait up to `timeout` s (None = until one ends) and handle every render that ended."""
        owner = {}
        for vid, job in self.running.items():
            owner[job.conn] = owner[job.proc.sentinel] = vid
        for vid in sorted({owner[ready] for ready in mp_connection.wait(list(owner), timeout)}):
            job = self.running.pop(vid)
            self._finish(_collect(job), job.task.marker)

    def _is_fresh(self, vid: str, marker: Stamp) -> bool:
        """<vid>.mp4 exists and its render record says it was rendered from `marker` with these options."""
        try:
            record = json.loads(_record_path(self._vis_dir, vid).read_text())
            mp4 = (self._vis_dir / f'{vid}.mp4').stat()
        except (OSError, ValueError):
            return False
        return record == _render_record(marker, self._persons, mp4)

    def _finish(self, result: RenderResult, marker: Stamp) -> None:
        vid, s = result.video_id, self.summary
        if result.error is not None:
            self._failed_at[vid] = marker
            s.failed.append(vid)
            log.error('%s: render failed: %s', vid, result.error)
            return
        self._settled.add(vid)
        s.rendered.append(vid)
        s.frames += result.frames
        s.size_bytes += result.size_bytes
        log.info('%s: %d frames in %.0f s (%.0f fps), %.1f MB', vid, result.frames, result.seconds,
                 result.frames / max(result.seconds, 1e-9), result.size_bytes / 1e6)


def _stamp(path: Path) -> Optional[Stamp]:
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return st.st_ino, st.st_mtime_ns


def _record_path(vis_dir: Path, vid: str) -> Path:
    return vis_dir / RECORDS / f'{vid}.json'


def _render_record(marker: Stamp, persons: bool, mp4: os.stat_result) -> Dict[str, object]:
    """What <vis_dir>/.state/rendered/<vid>.json holds (as JSON). The mp4's inode is new after each
    render, so an older record never matches a newer mp4."""
    return dict(marker=list(marker), persons=persons, mp4=[mp4.st_ino, mp4.st_size])


def _remove_stale_parts(vis_dir: Path) -> None:
    """Delete the <name>.<pid>.part files in `vis_dir` whose writing process is gone."""
    for path in [*vis_dir.glob('*.part'), *(vis_dir / RECORDS).glob('*.part')]:
        pid = path.stem.rsplit('.', 1)[-1]
        if pid.isdigit() and not _alive(int(pid)):
            path.unlink(missing_ok=True)
            log.info('removed %s, left behind by a killed render', path)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:   # another user's process
        pass
    return True


def _launch(task: RenderTask) -> _Job:
    ctx = multiprocessing.get_context('spawn')
    reader, writer = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_render_process, args=(task, writer), name=f'slp-render-{task.video_id}', daemon=True)
    proc.start()
    writer.close()   # only the child writes; its exit then reads as EOF here
    return _Job(task, proc, reader)


def _collect(job: _Job) -> RenderResult:
    """The job's result once its process has ended; an error result if it died without one."""
    result = None
    try:
        if job.conn.poll(_JOIN_S):
            result = job.conn.recv()
    except (EOFError, OSError):
        pass
    job.conn.close()
    job.proc.join(_JOIN_S)
    if job.proc.exitcode is None:
        job.proc.kill()
        job.proc.join()
    if result is None:
        result = RenderResult(job.task.video_id, error=f'the render process exited with code {job.proc.exitcode}')
    job.proc.close()
    return result


def _stop(jobs: Iterable[_Job]) -> None:
    """SIGTERM every render process still running (each cleans up), wait, kill stragglers."""
    jobs = list(jobs)
    for job in jobs:
        if job.proc.exitcode is None:
            job.proc.terminate()
    for job in jobs:
        job.proc.join(_JOIN_S)
        if job.proc.exitcode is None:
            job.proc.kill()
            job.proc.join()
        job.conn.close()


def _log_summary(summary: RenderSummary, layout: OutputLayout, video_ids: Optional[Sequence[str]]) -> None:
    log.info('render-done ended after %.0f s: %d rendered (%d frames, %.1f MB), %d up to date, %d failed',
             summary.seconds, len(summary.rendered), summary.frames, summary.size_bytes / 1e6,
             len(summary.up_to_date), len(summary.failed))
    if summary.failed:
        log.error('failed videos: %s', ' '.join(summary.failed))
    missing = [vid for vid in video_ids or () if not layout.done_marker(vid).exists()]
    if missing:
        log.warning('not committed, so not rendered: %s', ' '.join(missing))


def _raise_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


@contextlib.contextmanager
def sigterm_as_interrupt() -> Iterator[None]:
    """SIGTERM raises KeyboardInterrupt like Ctrl-C inside the block, so the render clean-up runs
    (a no-op outside the main thread)."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(signal.SIGTERM, _raise_interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


# --------------------------------------------------------------------------- render process
def _render_process(task: RenderTask, conn: mp_connection.Connection) -> None:
    """Child process: render one video and send its RenderResult; exit code 130 when stopped."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''   # before torch is imported: never touch the extraction's GPUs
    os.nice(NICENESS)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _interrupt_once)
    try:
        _stop_with_parent()
        conn.send(_render(task))
    except KeyboardInterrupt:
        sys.exit(130)
    finally:
        conn.close()


def _stop_with_parent() -> None:
    """Have the kernel SIGTERM this process when its parent dies, even by SIGKILL, so no orphaned
    render keeps running beside the next render-done."""
    import ctypes
    if ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'prctl(PR_SET_PDEATHSIG) failed')
    if os.getppid() != multiprocessing.parent_process().pid:   # it died before the prctl
        raise KeyboardInterrupt


def _interrupt_once(signum, frame) -> None:
    """Ctrl-C (to the whole process group) and the parent's SIGTERM usually both arrive; the
    second must not cut short the clean-up (kill ffmpeg, delete the .part file) of the first."""
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal.SIG_IGN)
    raise KeyboardInterrupt


def _render(task: RenderTask) -> RenderResult:
    """Render the mp4, then write its render record."""
    from .render import durable_replace, render_video
    from .video import probe
    started = time.monotonic()
    try:
        video = probe(task.video_path, task.video_id)
        render_video(task.root, video, task.out_path, persons=task.persons)
        if _stamp(OutputLayout(task.root).done_marker(task.video_id)) != task.marker:
            task.out_path.unlink()
            raise RuntimeError('the done marker changed while rendering (re-extracted or derived); '
                               'the video is rendered again once it is committed')
        frames, mp4 = probe(task.out_path).nb_frames, task.out_path.stat()
        tmp = task.record_path.with_name(f'{task.record_path.name}.{os.getpid()}.part')
        tmp.write_text(json.dumps(_render_record(task.marker, task.persons, mp4)) + '\n')
        durable_replace(tmp, task.record_path)
    except Exception as exc:
        return RenderResult(task.video_id, error=f'{type(exc).__name__}: {exc}')
    return RenderResult(task.video_id, frames, time.monotonic() - started, mp4.st_size)
