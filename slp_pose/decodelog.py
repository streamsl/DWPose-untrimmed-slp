"""Per-worker decoder logs: which video's decoding made libav print errors.

OpenCV's FFmpeg backend lets libav print its errors ('error while decoding MB', 'Invalid NAL unit
size', 'concealing ... errors', 'partial file') to the process's stderr, from C and without a video
id. A damaged source, or a bad read of it, makes such lines, so they must name their video. Each GPU worker therefore
points its file descriptor 2 at a file of its own, <out_root>/.state/logs/decode_gpu<N>_<stamp>.log
(DecodeLog; N = the physical GPU, stamp = the worker's start, so a respawned worker starts a new
file), and writes a marker line around each video its reader decodes and when it commits one:

  BEGIN <video_id> attempt <k> at <time>   the reader starts decoding the video (task attempt k)
  END <video_id> at <time>                 the reader is done with it (its decoder is closed)
  COMMIT <video_id> at <time>              the worker committed the video (after its END)

A line between a video's BEGIN and END was printed while the reader decoded it. Python's
sys.stderr is pointed at the old stderr, so the worker's warnings and tracebacks stay in the
extract's own output, but other C code of the worker can still write to fd 2 during a decode
(TensorRT, CUDA, OpenCV in the crop threads, a crash message: these land in the decode log too).
printed_lines() returns every line of each video's committed decode, and is_decoder_line() tells
the decoder's own lines: libav's, which carry its context prefix '[<component> @ 0x<address>]'
(e.g. '[h264 @ 0x55d0c2a3c4c0] error while decoding MB 6 2'), and OpenCV's videoio warnings.
libav's repeat counter ('Last message repeated N times') is dropped: it is printed at the next
message, which may come after the END. `check` reports the videos whose committed decode printed
decoder errors, unless their dataset declares the source damaged (datasets.Video.known_damage),
and logs other lines as info.
The worker's results never depend on any of this.

Standard library only (the worker imports it before torch; `check` imports no torch for it).
"""
from __future__ import annotations

import ctypes
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Tuple

_NAME = re.compile(r'decode_gpu(\d+)_(\d{8}-\d{6})\.log')
# A marker ends its line; a libav message printed without its newline may precede it on that line.
_MARKER = re.compile(r'(.*?)(BEGIN|END|COMMIT) ([A-Za-z0-9._-]+)(?: attempt (\d+))? at (\S+)')
_REPEATED = re.compile(r'^\s*Last message repeated \d+ times?\s*$')
# libav's context prefix ('[h264 @ 0x55d0c2a3c4c0] ', '[mov,mp4,m4a,3gp,3g2,mj2 @ 0x...] '), or an
# OpenCV log line of its video I/O ('[ WARN:0@1.234] global cap_ffmpeg_impl.hpp:... ').
_DECODER = re.compile(r'\[[^\[\]@]+ @ 0x[0-9A-Fa-f]+\] |\[ ?(?:WARN|ERROR|FATAL)[^\]]*\].*(?:videoio|cap_)')


def log_dir(out_root: Path) -> Path:
    """<out_root>/.state/logs, where the decode logs (and the commands' logs) go."""
    return Path(out_root) / '.state' / 'logs'


_libc: List[object] = []   # the process's C library, loaded once (_flush_c_stdio)


def _flush_c_stdio() -> None:
    """fflush(NULL): the C stdio buffers of this process (libav writes through stderr, unbuffered
    by default; this covers a library that buffers it)."""
    try:
        if not _libc:
            _libc.append(ctypes.CDLL(None))
        _libc[0].fflush(None)
    except (OSError, AttributeError):
        pass


class DecodeLog:
    """A worker's decode log: fd 2 redirected to <out_root>/.state/logs/decode_gpu<gpu>_<stamp>.log
    on the first `open` (C code's stderr: libav, OpenCV), with sys.stderr (Python's) moved to the old
    stderr; the marker lines; and `close` to put both back."""

    def __init__(self, gpu: int) -> None:
        self._gpu = gpu
        self._stamp = time.strftime('%Y%m%d-%H%M%S')
        self._saved: Optional[int] = None   # a duplicate of the old fd 2 while redirected
        self._python: Optional[TextIO] = None   # sys.stderr before `open`, and ours while open
        self._ours: Optional[TextIO] = None
        self.path: Optional[Path] = None

    def open(self, out_root: Path) -> None:
        """Redirect fd 2 to this worker's decode log under `out_root` (appending; only the first
        call of the process acts). Best effort: when the file cannot be opened, stderr stays as it
        is and the markers are skipped (the extraction never depends on the log)."""
        if self.path is not None:
            return
        path = log_dir(out_root) / f'decode_gpu{self._gpu}_{self._stamp}.log'
        self.path = path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        except OSError:
            return
        sys.stderr.flush()
        _flush_c_stdio()
        self._saved = os.dup(2)
        os.dup2(fd, 2)
        os.close(fd)
        self._python = sys.stderr
        self._ours = os.fdopen(os.dup(self._saved), 'w', buffering=1, encoding='utf-8', errors='backslashreplace')
        sys.stderr = self._ours

    def mark(self, kind: str, video_id: str, attempt: Optional[int] = None) -> None:
        """Write one marker line (BEGIN needs `attempt`), flushing what was printed before it; a
        no-op unless fd 2 is redirected."""
        if self._saved is None:
            return
        more = '' if attempt is None else f' attempt {attempt}'
        try:
            _flush_c_stdio()
            os.write(2, f'{kind} {video_id}{more} at {time.strftime("%Y-%m-%dT%H:%M:%S")}\n'.encode())
        except OSError:   # e.g. a full disk: the log loses a marker, the extraction goes on
            pass

    def close(self) -> None:
        """Point fd 2 and sys.stderr back at what they were before `open` (idempotent)."""
        if self._saved is None:
            return
        _flush_c_stdio()
        os.dup2(self._saved, 2)
        os.close(self._saved)
        self._saved = None
        if self._ours is not None:
            if sys.stderr is self._ours:
                sys.stderr = self._python
            self._ours.close()
            self._ours = self._python = None


def printed_lines(out_root: Path) -> Dict[str, List[str]]:
    """{video_id: every line written to the worker's fd 2 (the decoder's and any other C code's)
    while the video was decoded} from the decode logs of `out_root`, for the last decode of each
    video that its worker then committed (the decode behind the outputs). Videos whose decode
    printed nothing are left out. The logs are read in the order of their start stamps, so a later
    run's decode replaces an earlier one."""
    out: Dict[str, List[str]] = {}
    for path in _logs(out_root):
        for vid, lines in _committed_lines(path).items():
            if lines:
                out[vid] = lines
            else:
                out.pop(vid, None)
    return out


def is_decoder_line(line: str) -> bool:
    """Whether a line of a decode log is the decoder's: libav's (its '[<component> @ 0x...]'
    prefix) or an OpenCV video I/O warning."""
    return _DECODER.search(line) is not None


def problem(lines: List[str]) -> str:
    """`check`'s problem line for a video whose committed decode printed `lines`."""
    return f'decoder errors while extracting ({len(lines)} lines): {lines[0]}'


def _logs(out_root: Path) -> List[Path]:
    found: List[Tuple[str, int, Path]] = []
    folder = log_dir(out_root)
    if folder.is_dir():
        for path in folder.iterdir():
            match = _NAME.fullmatch(path.name)
            if match:
                found.append((match.group(2), int(match.group(1)), path))
    return [path for _, _, path in sorted(found)]


def _committed_lines(path: Path) -> Dict[str, List[str]]:
    """{video: the lines of its decode in this log that the worker committed}. A COMMIT can come
    before its decode's END (the worker commits while the reader closes the decoder); it then
    applies at that END, and one after the END applies to the video's last decode."""
    decoded: Dict[str, List[str]] = {}
    committed: Dict[str, List[str]] = {}
    current: Optional[str] = None
    lines: List[str] = []
    commit_at_end = False

    def finish() -> None:
        decoded[current] = lines
        if commit_at_end:
            committed[current] = lines

    with open(path, encoding='utf-8', errors='replace') as f:
        for raw in f:
            text = raw.rstrip('\r\n')
            match = _MARKER.fullmatch(text)
            if match is None:
                if current is not None and text.strip() and not _REPEATED.match(text):
                    lines.append(text)
                continue
            before, kind, vid = match.group(1).strip(), match.group(2), match.group(3)
            if current is not None and before:
                lines.append(before)
            if kind == 'BEGIN':
                if current is not None:   # the decode before never ended (its worker died): it ends here
                    finish()
                current, lines, commit_at_end = vid, [], False
            elif kind == 'END' and vid == current:
                finish()
                current, lines, commit_at_end = None, [], False
            elif kind == 'COMMIT' and vid == current:
                commit_at_end = True
            elif kind == 'COMMIT' and vid in decoded:
                committed[vid] = decoded[vid]
    if current is not None:
        finish()
    return committed
