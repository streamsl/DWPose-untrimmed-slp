"""Full-length overlay videos of every committed video (spec §4.5, the `render-done` command).

Read-only with respect to the output root: it lists .state/done/, reads poses/ and persons/ of
committed videos, and probes the lock without creating it, so it can run beside a live extraction.
Each video is rendered by its own spawned process (render.render_video with persons, frame 0 to
the end) at a lowered CPU priority and with the GPUs hidden, into <vis_dir>/<vid>.mp4, followed by
its render record <vis_dir>/.state/rendered/<vid>.json (version RECORD_VERSION):
  inputs  what the overlay was drawn from (overlay_inputs): the zip CRC-32 and size of every
          persons/ member it draws (OVERLAY_MEMBERS; POSE_MEMBERS with --no-persons) and the size
          of poses/, which is a pure function of kpt_offsets, kpts and kpt_primary (checked bit for
          bit at every commit) once the done marker describes the files;
  files   [st_ino, st_mtime_ns] of the done marker and [st_ino, st_size, st_mtime_ns] of persons/
          and poses/ when `inputs` was last confirmed;
  persons the option; mp4 [st_ino, st_size] of the mp4 (an old record never vouches for a new mp4).
An mp4 is up to date while its record has these options and mp4, and `files` equals the current
files (equality, never an mtime comparison) or, after a commit, `inputs` equals the new files'
inputs; the record then gets the new `files`. So a `derive` that changes only the meta row (or
the provenance) does not re-render; a change of anything drawn does. A video whose record has
these options and mp4 but whose files the done marker does not describe at that look (a commit in
progress, or cut short) is neither rendered nor up to date: it is looked at again at the next pass
(summary.pending). A render whose inputs changed meanwhile, or whose files the done marker did not
describe (a commit in progress), is discarded.
A version-1 record ({marker, persons, mp4}: rendered from the commit of that done marker, written
by render-done before RECORD_VERSION) is upgraded while its marker is unchanged, so run the new
render-done (`--limit 0` renders nothing) before a re-commit; a render task from a still-running
older render-done (no record_version) keeps its version-1 behaviour.
One render-done at a time per visualization directory: it holds <vis_dir>/.state/lock.
Importing this module does not import torch or cv2; only the render processes do.
"""
from __future__ import annotations

import contextlib
import errno
import json
import logging
import multiprocessing
import os
import signal
import sys
import threading
import time
import zipfile
from dataclasses import dataclass, field
from multiprocessing import connection as mp_connection
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .record import OutputLayout
from .run import committed_videos, root_lock

log = logging.getLogger(__name__)

DEFAULT_JOBS = 3
POLL_S = 300.0     # --follow: seconds between looks for new commits while a render slot is free
GRACE_S = 900.0    # --follow: how long the root must stay unlocked before render-done returns
NICENESS = 10      # added to each render process's nice value
_JOIN_S = 30.0     # how long a stopped render process may take to clean up before it is killed
_PR_SET_PDEATHSIG = 1   # linux/prctl.h
RECORDS = Path('.state') / 'rendered'   # under the vis dir: <vid>.json render records
RECORD_VERSION = 2   # render records keyed on the overlay's inputs (version 1: on the done marker)
# persons/ members the overlay reads (render._open_persons, render._draw_people); POSE_MEMBERS
# determine poses/ (record invariant), the only input of a --no-persons render.
POSE_MEMBERS = ('num_frames', 'kpt_offsets', 'kpts', 'kpt_primary')
OVERLAY_MEMBERS = ('frame_size', 'det_offsets', 'det_boxes', 'det_scores', 'det_flags', 'num_persons',
                   'kpt_det') + POSE_MEMBERS
NOT_COMMITTED = ('the done marker does not describe poses/ and persons/ (a commit in progress or cut short); '
                 'the video is rendered once it is committed')

Stamp = Tuple[int, int]   # (st_ino, st_mtime_ns) of a done marker: changes whenever it is rewritten
Inputs = Dict[str, object]           # overlay_inputs: what an overlay is drawn from
FileKey = List[List[int]]            # the files it was read from (module docstring, `files`)


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
    # The render record to write: RECORD_VERSION, or 1 (the class default) for a task pickled by
    # an older render-done parent, which compares version-1 records only.
    record_version: int = 1


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
    # At the last look, a render record vouched for the mp4 but the done marker did not describe
    # the files (a commit in progress or cut short): neither rendered nor checked.
    pending: List[str] = field(default_factory=list)
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
                     persons: bool = True, poll_s: float = POLL_S, grace_s: float = GRACE_S) -> RenderSummary:
    """Render every committed video of `out_root` (or of `video_ids`) to <vis_dir>/<vid>.mp4.

    - Holds <vis_dir>/.state/lock for the whole run (RuntimeError if another run holds it) and
      first deletes the .part files that killed renders left behind.
    - Skips a video whose render record matches `persons`, its mp4 and its current overlay inputs
      (module docstring; a version-1 record is upgraded while its done marker is unchanged), and
      looks again at the next pass at one whose record vouches for its mp4 while the done marker
      does not describe the files (a commit in progress; summary.pending lists those still so at
      the end); renders at most `limit` videos (0: only checks and upgrades the records); up to `jobs`
      render processes at a time, each of which is stopped when this process dies. `video_paths`
      maps ids to source videos.
    - A failed video is logged and skipped (retried only once its done marker changes); a render
      whose inputs changed meanwhile is discarded and counts as failed until a later render
      of the same video in this run succeeds (summary.failed lists the videos still failed).
    - Without `follow`, returns once nothing is left to render. With `follow`, looks for new
      commits every `poll_s` s while a render slot is free and returns once nothing is left and
      no extraction has held the root for `grace_s` s (extraction_running is sampled before each
      listing, so a video committed just before the extraction ended is still rendered; the
      grace keeps it following across a stop and restart of the extraction, SIGTERM then the
      same command). Start it once the extraction holds the root (its log shows
      '<dataset> -> <root>: N videos ...'): when no extraction holds the root at the start, a
      warning says so and it behaves as without `follow` (no grace).
    - Ctrl-C or SIGTERM stops the render processes (each kills its ffmpeg and deletes its .part
      file), then KeyboardInterrupt is re-raised.
    """
    if jobs < 1 or poll_s <= 0 or grace_s < 0:
        raise ValueError(f'need jobs >= 1, poll_s > 0 and grace_s >= 0, got {jobs}, {poll_s} and {grace_s}')
    check_dirs(out_root, vis_dir)
    out_root, vis_dir = Path(out_root).resolve(), Path(vis_dir).resolve()
    with root_lock(vis_dir), sigterm_as_interrupt():   # the lock file of an output root, on the vis dir
        (vis_dir / RECORDS).mkdir(parents=True, exist_ok=True)
        _remove_stale_parts(vis_dir)
        started = time.monotonic()
        batch = _Batch(out_root, vis_dir, video_paths, jobs, video_ids, limit, persons)
        log.info('render-done: %s -> %s, %d committed videos, %d jobs%s', out_root, vis_dir,
                 len(committed_videos(out_root)), jobs, ', following the extraction' if follow else '')
        free_since: Optional[float] = None   # monotonic time the root was first seen free since it was held
        if follow and not extraction_running(out_root):
            log.warning('--follow: no extract or derive run holds %s now, so render-done returns once the '
                        'videos committed so far are rendered (start it after the extraction has started)',
                        out_root)
            free_since = -float('inf')
        try:
            while True:
                held = follow and extraction_running(out_root)
                now = time.monotonic()
                if held:
                    free_since = None
                elif free_since is None:
                    free_since = now
                    if follow:
                        log.info('--follow: no extraction holds %s now; following for %.0f s more', out_root,
                                 grace_s)
                active = held or (follow and now - free_since < grace_s)
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
        records the up-to-date ones and the pending ones (looked at again at the next call)."""
        out, pending = [], []
        for vid in committed_videos(self.layout.root) if self._video_ids is None else self._video_ids:
            marker = _stamp(self.layout.done_marker(vid))
            if vid in self.running or marker is None or self._failed_at.get(vid) == marker:
                continue
            fresh = self._is_fresh(vid, marker)
            if fresh is None:
                pending.append(vid)
            elif not fresh:
                out.append((vid, marker))
            else:
                # Also when settled earlier in this run: an attempt since then may have failed during
                # a commit that changed nothing drawn.
                self._failed_at.pop(vid, None)
                if vid in self.summary.failed:
                    self.summary.failed.remove(vid)
                if vid not in self._settled:
                    self._settled.add(vid)
                    self.summary.up_to_date.append(vid)
        self.summary.pending = pending
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
                                                   self._persons, marker, RECORD_VERSION))

    def collect(self, timeout: Optional[float]) -> None:
        """Wait up to `timeout` s (None = until one ends) and handle every render that ended."""
        owner = {}
        for vid, job in self.running.items():
            owner[job.conn] = owner[job.proc.sentinel] = vid
        for vid in sorted({owner[ready] for ready in mp_connection.wait(list(owner), timeout)}):
            job = self.running.pop(vid)
            self._finish(_collect(job), job.task.marker)

    def _is_fresh(self, vid: str, marker: Stamp) -> Optional[bool]:
        """True when <vid>.mp4 exists and its render record says it was drawn, with these options,
        from what the current files would draw (module docstring; the record is then refreshed or
        upgraded). None when that record has these options and mp4 but the done marker does not
        describe the files now (overlay_inputs None: a commit in progress, or cut short): rendering
        now would redo an mp4 that a meta-only commit leaves up to date, so look again later.
        Else False."""
        path = _record_path(self._vis_dir, vid)
        try:
            record = json.loads(path.read_text())
            mp4 = (self._vis_dir / f'{vid}.mp4').stat()
        except (OSError, ValueError):
            return False
        if not isinstance(record, dict):
            return False
        if record.get('version') == RECORD_VERSION:
            if record.get('persons') != self._persons or record.get('mp4') != [mp4.st_ino, mp4.st_size]:
                return False
            if record.get('files') == _file_key(self.layout, vid):
                return True                     # the very files it was drawn from
            state = overlay_inputs(self.layout, vid, self._persons)
            if state is None:
                return None
            if state[0] != record.get('inputs'):
                return False
        elif record == _render_record(marker, self._persons, mp4):   # version 1: drawn from this commit
            state = overlay_inputs(self.layout, vid, self._persons)
            if state is None:
                return None
            if state[1][0] != list(marker):     # re-committed since `marker` was read
                return False
        else:
            return False
        try:
            _write_json(path, _inputs_record(self._persons, mp4, *state))
        except OSError as exc:   # still up to date; the next look confirms it again
            log.warning('%s: cannot update its render record: %s', vid, exc)
        return True

    def _finish(self, result: RenderResult, marker: Stamp) -> None:
        vid, s = result.video_id, self.summary
        if result.error is not None:
            self._failed_at[vid] = marker
            if vid not in s.failed:
                s.failed.append(vid)
            log.error('%s: render failed: %s', vid, result.error)
            return
        self._settled.add(vid)
        self._failed_at.pop(vid, None)
        if vid in s.failed:   # an earlier attempt in this run failed, e.g. discarded as its marker changed
            s.failed.remove(vid)
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
    """A version-1 render record (as JSON): what render-done wrote before RECORD_VERSION, and what
    a running older render-done parent still compares. The mp4's inode is new after each render,
    so an older record never matches a newer mp4."""
    return dict(marker=list(marker), persons=persons, mp4=[mp4.st_ino, mp4.st_size])


def _inputs_record(persons: bool, mp4: os.stat_result, inputs: Inputs, files: FileKey) -> Dict[str, object]:
    """What <vis_dir>/.state/rendered/<vid>.json holds (as JSON; module docstring)."""
    return dict(version=RECORD_VERSION, persons=persons, mp4=[mp4.st_ino, mp4.st_size], inputs=inputs, files=files)


def overlay_inputs(layout: OutputLayout, video_id: str, persons: bool) -> Optional[Tuple[Inputs, FileKey]]:
    """(inputs, files) of a committed video (module docstring), or None when a file is missing or
    unreadable or the done marker does not describe them (sizes, extraction and derivation hash of
    the persons provenance): a commit in progress or cut short.

    inputs = {'poses_bytes': size of poses/, 'members': {name: [CRC-32, size] of its .npy}} for
    OVERLAY_MEMBERS (POSE_MEMBERS without `persons`), from the zip central directory (no array is
    read). The marker and persons/ are read through the descriptors they are stat'ed with.
    """
    try:
        with open(layout.done_marker(video_id), 'rb') as f:
            marker_st, marker = os.fstat(f.fileno()), json.loads(f.read())
        with open(layout.persons(video_id), 'rb') as f, zipfile.ZipFile(f) as zf:
            persons_st = os.fstat(f.fileno())
            entries = {info.filename: info for info in zf.infolist()}
            with zf.open('provenance.npy') as member:
                provenance = json.loads(str(np.lib.format.read_array(member, allow_pickle=False)[()]))
            members = {name: [entries[name + '.npy'].CRC, entries[name + '.npy'].file_size]
                       for name in (OVERLAY_MEMBERS if persons else POSE_MEMBERS)}
        poses_st = os.stat(layout.poses(video_id))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return None
    if not (isinstance(marker, dict) and isinstance(provenance, dict)
            and marker.get('persons_bytes') == persons_st.st_size and marker.get('poses_bytes') == poses_st.st_size
            and all(marker.get(key) == provenance.get(key) for key in ('extraction_hash', 'derivation_hash'))):
        return None
    files = [[marker_st.st_ino, marker_st.st_mtime_ns], [persons_st.st_ino, persons_st.st_size, persons_st.st_mtime_ns],
             [poses_st.st_ino, poses_st.st_size, poses_st.st_mtime_ns]]
    return dict(poses_bytes=poses_st.st_size, members=members), files


def _file_key(layout: OutputLayout, video_id: str) -> Optional[FileKey]:
    """The `files` of a video's current done marker, persons/ and poses/ (None if one is missing)."""
    try:
        marker, persons, poses = (os.stat(path) for path in (layout.done_marker(video_id), layout.persons(video_id),
                                                             layout.poses(video_id)))
    except OSError:
        return None
    return [[marker.st_ino, marker.st_mtime_ns], [persons.st_ino, persons.st_size, persons.st_mtime_ns],
            [poses.st_ino, poses.st_size, poses.st_mtime_ns]]


def _write_json(path: Path, obj: object) -> None:
    """Write `obj` as JSON to `path` durably: <path>.<pid>.part, fsync, rename, fsync the directory."""
    tmp = path.with_name(f'{path.name}.{os.getpid()}.part')
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(json.dumps(obj) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError as e:
        if e.errno != errno.EINVAL:   # some filesystems cannot fsync a directory; renames stay atomic
            raise
    finally:
        os.close(fd)


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
    if summary.pending:
        log.warning('not checked, as a commit was in progress (or was cut short): %s; run render-done again once '
                    'they are committed', ' '.join(summary.pending))
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
    """Render the mp4, then write its render record (version 1 for an older parent's task)."""
    from .render import durable_replace, render_video
    from .video import probe
    started = time.monotonic()
    layout = OutputLayout(task.root)
    try:
        video = probe(task.video_path, task.video_id)
        if task.record_version < RECORD_VERSION:   # an older render-done parent: its record and its rule
            render_video(task.root, video, task.out_path, persons=task.persons)
            if _stamp(layout.done_marker(task.video_id)) != task.marker:
                task.out_path.unlink()
                raise RuntimeError('the done marker changed while rendering (re-extracted or derived); '
                                   'the video is rendered again once it is committed')
            frames, mp4 = probe(task.out_path).nb_frames, task.out_path.stat()
            tmp = task.record_path.with_name(f'{task.record_path.name}.{os.getpid()}.part')
            tmp.write_text(json.dumps(_render_record(task.marker, task.persons, mp4)) + '\n')
            durable_replace(tmp, task.record_path)
        else:
            before = overlay_inputs(layout, task.video_id, task.persons)
            if before is None:
                raise RuntimeError(NOT_COMMITTED)
            render_video(task.root, video, task.out_path, persons=task.persons)
            after = overlay_inputs(layout, task.video_id, task.persons)
            if after is None or after[0] != before[0]:
                task.out_path.unlink()
                raise RuntimeError('what the overlay draws changed while rendering (re-extracted or derived); '
                                   'the video is rendered again once it is committed')
            frames, mp4 = probe(task.out_path).nb_frames, task.out_path.stat()
            _write_json(task.record_path, _inputs_record(task.persons, mp4, *after))
    except Exception as exc:
        return RenderResult(task.video_id, error=f'{type(exc).__name__}: {exc}')
    return RenderResult(task.video_id, frames, time.monotonic() - started, mp4.st_size)
