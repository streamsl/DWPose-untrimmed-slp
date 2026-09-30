"""Per-video record: spill files, streamed poses .npy, persons .npz, commit, resume, derive, check.

CPU only: imports select, meta and env; detpost (which imports torch) only inside `derive`; never engines.

Layout under an output root <root> (spec §4.1, §4.4):
  poses/<vid>.npy          (T,133,3) float32 C-order, npy format v1.0, primary signer only;
                           bytes identical to np.save(path, poses) of the same array.
  persons/<vid>.npz        np.savez (stored, zip64) with exactly PERSONS_MEMBERS.
  .work/<vid>/             spill and staging files of an in-progress video (the whole .work/ is cleared at run start).
  .state/meta/<vid>.json   the video's meta row (meta.py), JSON with allow_nan.
  .state/done/<vid>.json   commit marker: dataclasses.asdict(CommitInfo) as JSON (sort_keys).
  .state/failed.jsonl      one JSON object per failed attempt: {video_id, error, traceback, attempt, gpu, time}.
video_meta.csv is written only by the parent (run.merge_meta via meta.write_video_meta).

persons/<vid>.npz members (spec §4.1; numpy dtypes exact, 0-d arrays for scalars/strings):
  schema_version int64 0-d = types.SCHEMA_VERSION; video_id str 0-d; frame_size (2,) int64 (W,H);
  fps (2,) int64 (num,den); num_frames int64 0-d = T; det_offsets (T+1,) int64; det_boxes (N,4)
  float32; det_scores (N,) float32; det_flags (N,) uint8; num_persons (T,) uint8; kpt_offsets
  (T+1,) int64; kpt_det (M,) int32 VIDEO-GLOBAL det row index; kpts (M,133,3) float32;
  kpt_primary (T,) int8; primary_rule str 0-d; meta str 0-d (JSON row); provenance str 0-d (JSON).
Invariants (PersonsRecord.problems, run at finalize and by `check`):
  poses[t] == kpts[kpt_offsets[t] + kpt_primary[t]] bit-for-bit, zeros when kpt_primary[t] == -1;
  kpt_primary[t] == -1 exactly where the frame has no POSE_SET row (equivalently num_persons[t] == 0
  while count_score_thr == pose_score_thr); num_persons[t] == number of COUNT_SET rows of frame t;
  kpt_det rows of frame t are exactly its POSED rows, ascending; POSED implies POSE_SET; det rows
  within a frame have non-increasing scores. Also checked: dtypes, shapes and CSR indices, the
  primary is the `primary_rule` pick on the frame's pose set, and `meta` is the row computed
  from the record (poses/ and the meta row are pure functions of the record).
Commit (finalize and derive): every file is staged and fsynced in .work/<vid>/, then os.replace
  persons/<vid>.npz -> poses/<vid>.npy -> .state/meta/<vid>.json, fsync those directories, then
  os.replace .state/done/<vid>.json and fsync its directory, so a done marker always describes
  complete files. finalize first deletes an older done marker of the video (a crash mid-commit
  then reads as 'extract'); derive keeps it (a crash mid-derive reads as 'derive' again, which
  is correct because derive never changes its own inputs).
Memory: a SpillWriter holds one chunk; finalize, derive and check read the big columns through
  memmaps and handle kpts/poses in blocks of _BLOCK_FRAMES frames, so RAM is O(T + N) small
  arrays plus one block, for any video length.
"""
from __future__ import annotations

import dataclasses
import datetime
import errno
import hashlib
import io
import json
import math
import os
import shutil
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from . import env
from .meta import compute_meta_row
from .select import PRIMARY_RULES
from .settings import Settings
from .types import COUNT_SET, NUM_KEYPOINTS, POSE_SET, POSED, SCHEMA_VERSION, ChunkDets, ChunkPoses, VideoInfo

PERSONS_MEMBERS = ('schema_version', 'video_id', 'frame_size', 'fps', 'num_frames', 'det_offsets', 'det_boxes',
                   'det_scores', 'det_flags', 'num_persons', 'kpt_offsets', 'kpt_det', 'kpts', 'kpt_primary',
                   'primary_rule', 'meta', 'provenance')
_ROW = (NUM_KEYPOINTS, 3)   # one person's keypoints in the saved encoding
_BLOCK_FRAMES = 4096        # frames per block when streaming or comparing poses (~6.5 MB)
# Spill columns in .work/<vid>/<name>.bin: dtype and per-row shape. *_ends are the CSR offsets
# without their leading 0; kpt_det is already video-global.
_SPILL_COLUMNS = {
    'det_ends': (np.int64, ()), 'det_boxes': (np.float32, (4,)), 'det_scores': (np.float32, ()),
    'det_flags': (np.uint8, ()), 'num_persons': (np.uint8, ()), 'kpt_ends': (np.int64, ()),
    'kpt_det': (np.int32, ()), 'kpts': (np.float32, _ROW), 'kpt_primary': (np.int8, ()),
}
# Files staged in .work/<vid>/ and renamed into place by the commit.
_STAGED_POSES, _STAGED_PERSONS, _STAGED_META, _STAGED_DONE = 'poses.npy', 'persons.npz', 'meta.json', 'done.json'


@dataclass(frozen=True)
class OutputLayout:
    """Paths of one output root (see module docstring); `root` may be given as a str."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, 'root', Path(self.root))

    def poses(self, video_id: str) -> Path:
        return self.root / 'poses' / f'{video_id}.npy'

    def persons(self, video_id: str) -> Path:
        return self.root / 'persons' / f'{video_id}.npz'

    @property
    def work_root(self) -> Path:
        return self.root / '.work'

    def work(self, video_id: str) -> Path:
        return self.work_root / video_id

    def meta_json(self, video_id: str) -> Path:
        return self.root / '.state' / 'meta' / f'{video_id}.json'

    def done_marker(self, video_id: str) -> Path:
        return self.root / '.state' / 'done' / f'{video_id}.json'

    @property
    def failed_log(self) -> Path:
        return self.root / '.state' / 'failed.jsonl'

    @property
    def video_meta_csv(self) -> Path:
        return self.root / 'video_meta.csv'


@dataclass(frozen=True)
class CommitInfo:
    """What the done marker records (also the payload of a worker 'done' event)."""

    video_id: str
    num_frames: int
    extraction_hash: str
    derivation_hash: str
    poses_bytes: int
    persons_bytes: int


# --------------------------------------------------------------------------- persons record
@dataclass
class PersonsRecord:
    """In-memory mirror of persons/<vid>.npz (field meanings and dtypes as in the module docstring).

    Array fields may be read-only memmaps (load_persons(..., mmap_mode='r')); the methods read
    `kpts` in blocks, so they never load it whole.
    """

    video_id: str
    frame_size: Tuple[int, int]
    fps: Tuple[int, int]
    num_frames: int
    det_offsets: np.ndarray
    det_boxes: np.ndarray
    det_scores: np.ndarray
    det_flags: np.ndarray
    num_persons: np.ndarray
    kpt_offsets: np.ndarray
    kpt_det: np.ndarray
    kpts: np.ndarray
    kpt_primary: np.ndarray
    primary_rule: str
    meta: Dict[str, object]
    provenance: Dict[str, object]

    def poses(self) -> np.ndarray:
        """(T,133,3) float32: the primary's kpts row per frame, zeros where kpt_primary == -1."""
        return self._pose_block(0, self.num_frames)

    def problems(self, poses: Optional[np.ndarray] = None) -> List[str]:
        """Invariant violations as readable lines (empty = OK); `poses` is checked if given."""
        out = self._structure_problems()
        if out:
            return out
        t = self.num_frames
        flags, scores = np.asarray(self.det_flags), np.asarray(self.det_scores)
        det_frame = np.repeat(np.arange(t), np.diff(self.det_offsets))
        rising = (det_frame[1:] == det_frame[:-1]) & (scores[1:] > scores[:-1])
        _report(out, 'detector scores increase within the frame', det_frame[1:][rising])
        posed = (flags & POSED) != 0
        _report(out, 'POSED rows outside the pose set', det_frame[posed & ((flags & POSE_SET) == 0)])
        count = np.bincount(det_frame[(flags & COUNT_SET) != 0], minlength=t)
        _report(out, 'num_persons is not the COUNT_SET size', np.flatnonzero(self.num_persons != count))
        kpt_count = np.diff(self.kpt_offsets)
        _report(out, 'kpt rows are not the POSED rows (count)',
                np.flatnonzero(kpt_count != np.bincount(det_frame[posed], minlength=t)))
        posed_rows = np.flatnonzero(posed)
        if len(posed_rows) == len(self.kpt_det):
            kpt_frame = np.repeat(np.arange(t), kpt_count)
            _report(out, 'kpt rows are not the POSED rows (index)',
                    np.unique(kpt_frame[np.asarray(self.kpt_det) != posed_rows]))
        position, not_posed = _rule_primary(self, self.primary_rule)
        _report(out, f'the {self.primary_rule!r} pick is not posed', np.flatnonzero(not_posed))
        _report(out, f'kpt_primary is not the {self.primary_rule!r} pick (-1 exactly without a pose set)',
                np.flatnonzero(~not_posed & (position != self.kpt_primary)))
        row = compute_meta_row(self.video_id, self.frame_size, self.fps, self.num_persons, self.kpt_offsets,
                               self.kpts, self.kpt_primary)
        if _canonical(row) != _canonical(self.meta):
            out.append('meta is not the row computed from the record')
        if poses is not None:
            out.extend(self._poses_problems(poses))
        return out

    def _structure_problems(self) -> List[str]:
        """dtype / shape / index problems; any of them makes the other checks meaningless."""
        t, n, m = self.num_frames, len(self.det_scores), len(self.kpt_det)
        out = [] if t >= 1 else ['no frames']
        for name, dtype, shape in (('det_offsets', np.int64, (t + 1,)), ('det_boxes', np.float32, (n, 4)),
                                   ('det_scores', np.float32, (n,)), ('det_flags', np.uint8, (n,)),
                                   ('num_persons', np.uint8, (t,)), ('kpt_offsets', np.int64, (t + 1,)),
                                   ('kpt_det', np.int32, (m,)), ('kpts', np.float32, (m,) + _ROW),
                                   ('kpt_primary', np.int8, (t,))):
            value = getattr(self, name)
            if value.dtype != dtype or value.shape != shape:
                out.append(f'{name} is {value.dtype} {value.shape}, expected {np.dtype(dtype)} {shape}')
        if out:
            return out
        for name, rows in (('det_offsets', n), ('kpt_offsets', m)):
            offsets = np.asarray(getattr(self, name))
            if offsets[0] != 0 or offsets[-1] != rows or np.any(np.diff(offsets) < 0):
                out.append(f'{name} is not a CSR index over {rows} rows')
        if out:
            return out
        if m and not (self.kpt_det.min() >= 0 and self.kpt_det.max() < n):
            out.append('kpt_det points outside det_*')
        primary = np.asarray(self.kpt_primary)
        _report(out, 'kpt_primary outside the posed list',
                np.flatnonzero((primary < -1) | (primary >= np.diff(self.kpt_offsets))))
        if self.primary_rule not in PRIMARY_RULES:
            out.append(f'unknown primary rule {self.primary_rule!r}')
        if len(self.frame_size) != 2 or min(self.frame_size) < 1 or len(self.fps) != 2 or min(self.fps) < 1:
            out.append(f'bad frame_size {self.frame_size} or fps {self.fps}')
        return out

    def _poses_problems(self, poses: np.ndarray) -> List[str]:
        if poses.dtype != np.float32 or poses.shape != (self.num_frames,) + _ROW:
            return [f'poses is {poses.dtype} {poses.shape}, expected float32 {(self.num_frames,) + _ROW}']
        bad = [s + np.flatnonzero((np.asarray(poses[s:s + len(block)]).view(np.uint32) != block.view(np.uint32))
                                  .reshape(len(block), -1).any(axis=1))
               for s, block in self._pose_blocks()]
        out: List[str] = []
        _report(out, 'poses rows are not the primary keypoints (bitwise; zeros without a primary)', np.concatenate(bad))
        return out

    def _pose_block(self, start: int, stop: int) -> np.ndarray:
        primary = np.asarray(self.kpt_primary[start:stop])
        out = np.zeros((len(primary),) + _ROW, np.float32)
        has = primary >= 0
        out[has] = self.kpts[np.asarray(self.kpt_offsets[start:stop])[has] + primary[has]]
        return out

    def _pose_blocks(self) -> Iterator[Tuple[int, np.ndarray]]:
        for start in range(0, self.num_frames, _BLOCK_FRAMES):
            yield start, self._pose_block(start, min(start + _BLOCK_FRAMES, self.num_frames))


def _report(out: List[str], what: str, frames) -> None:
    """Append 'what: n frame(s), first ...' when `frames` (frame indices) is not empty."""
    frames = np.asarray(frames)
    if len(frames):
        out.append(f'{what}: {len(frames)} frame(s), first ' + ', '.join(str(int(t)) for t in frames[:5]))


def _canonical(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, allow_nan=True)


def _rule_primary(record: PersonsRecord, rule_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Per frame: the position of the rule's pick in the posed list (-1 without a pose set), and
    whether the pick is missing from the posed rows (that rule would need re-extraction)."""
    rule = PRIMARY_RULES[rule_name]
    det_offsets, kpt_offsets = np.asarray(record.det_offsets), np.asarray(record.kpt_offsets)
    boxes, scores, kpt_det = np.asarray(record.det_boxes), np.asarray(record.det_scores), np.asarray(record.kpt_det)
    pose = (np.asarray(record.det_flags) & POSE_SET) != 0
    position = np.full(record.num_frames, -1, np.int64)
    not_posed = np.zeros(record.num_frames, bool)
    det_frame = np.repeat(np.arange(record.num_frames), np.diff(det_offsets))
    for t in np.unique(det_frame[pose]):
        rows = det_offsets[t] + np.flatnonzero(pose[det_offsets[t]:det_offsets[t + 1]])
        pick = rows[rule(boxes[rows], scores[rows])]
        hit = np.flatnonzero(kpt_det[kpt_offsets[t]:kpt_offsets[t + 1]] == pick)
        if len(hit):
            position[t] = hit[0]
        else:
            not_posed[t] = True
    return position, not_posed


def load_persons(path: Path, mmap_mode: Optional[str] = None) -> PersonsRecord:
    """Read persons/<vid>.npz (all members; raises on a missing member or schema_version != 1).

    With mmap_mode ('r'), the array members are memory-mapped from the uncompressed zip instead
    of read (np.load ignores mmap_mode for .npz files).
    """
    arrays = _read_npz(Path(path), mmap_mode)
    missing = [name for name in PERSONS_MEMBERS if name not in arrays]
    if missing:
        raise ValueError(f'{path}: missing members {missing}')
    if int(arrays['schema_version']) != SCHEMA_VERSION:
        raise ValueError(f'{path}: schema_version {int(arrays["schema_version"])}, expected {SCHEMA_VERSION}')

    def text(name: str) -> str:
        return str(arrays[name][()])

    def pair(name: str) -> Tuple[int, int]:
        return int(arrays[name][0]), int(arrays[name][1])

    return PersonsRecord(
        video_id=text('video_id'), frame_size=pair('frame_size'), fps=pair('fps'),
        num_frames=int(arrays['num_frames']), det_offsets=arrays['det_offsets'], det_boxes=arrays['det_boxes'],
        det_scores=arrays['det_scores'], det_flags=arrays['det_flags'], num_persons=arrays['num_persons'],
        kpt_offsets=arrays['kpt_offsets'], kpt_det=arrays['kpt_det'], kpts=arrays['kpts'],
        kpt_primary=arrays['kpt_primary'], primary_rule=text('primary_rule'), meta=json.loads(text('meta')),
        provenance=json.loads(text('provenance')))


def save_persons(path: Path, record: PersonsRecord) -> None:
    """np.savez (uncompressed) of PERSONS_MEMBERS to exactly `path`, fsynced (callers write to a
    temp name and os.replace it during the commit). np.savez streams memmapped members in 16 MiB
    pieces, and its output is deterministic (members are stamped 1980-01-01)."""
    problems = record._structure_problems()
    if problems:
        raise ValueError(f'{record.video_id}: cannot save the persons record: ' + '; '.join(problems))
    members = dict(
        schema_version=np.array(SCHEMA_VERSION, np.int64), video_id=np.array(record.video_id),
        frame_size=np.array(record.frame_size, np.int64), fps=np.array(record.fps, np.int64),
        num_frames=np.array(record.num_frames, np.int64), det_offsets=record.det_offsets,
        det_boxes=record.det_boxes, det_scores=record.det_scores, det_flags=record.det_flags,
        num_persons=record.num_persons, kpt_offsets=record.kpt_offsets, kpt_det=record.kpt_det,
        kpts=record.kpts, kpt_primary=record.kpt_primary, primary_rule=np.array(record.primary_rule),
        meta=np.array(_meta_json(record.meta)), provenance=np.array(json.dumps(record.provenance, sort_keys=True)))
    with open(path, 'wb') as f:
        np.savez(f, **members)
        f.flush()
        os.fsync(f.fileno())


def _meta_json(row: Dict[str, object]) -> str:
    return json.dumps(row, allow_nan=True)


def _read_npz(path: Path, mmap_mode: Optional[str]) -> Dict[str, np.ndarray]:
    out = {}
    with zipfile.ZipFile(path) as zf, open(path, 'rb') as raw:
        for info in zf.infolist():
            name = info.filename[:-len('.npy')] if info.filename.endswith('.npy') else info.filename
            array = None
            if mmap_mode is not None and info.compress_type == zipfile.ZIP_STORED:
                array = _memmap_member(path, raw, info, mmap_mode)
            if array is None:
                with zf.open(info) as member:
                    array = np.lib.format.read_array(member, allow_pickle=False)
            out[name] = array
    return out


def _memmap_member(path: Path, raw, info: zipfile.ZipInfo, mmap_mode: str) -> Optional[np.ndarray]:
    """Memmap of one stored .npy member, or None for 0-d / empty / unusual members."""
    raw.seek(info.header_offset)
    local = raw.read(30)  # zip local file header; the data follows its name and extra field
    if local[:4] != b'PK\x03\x04':
        raise zipfile.BadZipFile(f'{path}: bad local header for {info.filename}')
    name_len, extra_len = struct.unpack('<HH', local[26:30])
    raw.seek(info.header_offset + 30 + name_len + extra_len)
    version = np.lib.format.read_magic(raw)
    if version not in ((1, 0), (2, 0)):
        return None
    read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, fortran_order, dtype = read_header(raw)
    if dtype.hasobject or not shape or 0 in shape:
        return None
    return np.memmap(path, dtype=dtype, mode=mmap_mode, offset=raw.tell(), shape=shape,
                     order='F' if fortran_order else 'C')


# --------------------------------------------------------------------------- streamed poses .npy
def _npy_header(num_rows: int) -> bytes:
    """The exact header np.save writes for a (num_rows,133,3) float32 C-order array (v1.0)."""
    buf = io.BytesIO()
    np.lib.format.write_array_header_1_0(buf, {'descr': np.lib.format.dtype_to_descr(np.dtype(np.float32)),
                                               'fortran_order': False, 'shape': (int(num_rows),) + _ROW})
    return buf.getvalue()


class NpyStreamWriter:
    """Append (n,133,3) float32 rows to a .npy of unknown final length.

    Writes a v1.0 header for shape (0,133,3) up front (numpy pads v1.0 headers so the shape can
    grow in place) and rewrites it at close with the final T; raises if the header length would
    change. The closed file equals np.save of the same array byte-for-byte.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._file = open(self.path, 'wb')
        self._rows = 0
        header = _npy_header(0)
        self._header_len = len(header)
        self._file.write(header)

    def append(self, rows: np.ndarray) -> None:
        """Append (n,133,3) float32 rows (n may be 0)."""
        if rows.dtype != np.float32 or rows.shape[1:] != _ROW:
            raise ValueError(f'expected (n,133,3) float32 rows, got {rows.dtype} {rows.shape}')
        self._file.write(np.ascontiguousarray(rows).data)
        self._rows += len(rows)

    def close(self) -> int:
        """Rewrite the header, flush + fsync, return T (idempotent)."""
        if not self._file.closed:
            try:
                header = _npy_header(self._rows)
                if len(header) != self._header_len:
                    raise ValueError(f'{self.path}: the npy header would change length '
                                     f'({self._header_len} -> {len(header)} bytes)')
                self._file.seek(0)
                self._file.write(header)
                self._file.flush()
                os.fsync(self._file.fileno())
            finally:
                self._file.close()
        return self._rows

    def discard(self) -> None:
        """Close without finishing and delete the file (idempotent)."""
        self._file.close()
        self.path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- spill and commit
class SpillWriter:
    """Streams one video's chunk results to column files under <root>/.work/<vid>/.

    Host RAM stays O(chunk): det/kpt columns are appended to raw binary files and the primary
    poses go straight into an NpyStreamWriter at .work/<vid>/poses.npy.
    """

    def __init__(self, root: Path, video: VideoInfo, primary_rule: str) -> None:
        """Clear and create .work/<vid>/ (a previous crash may have left files there)."""
        _check_rule(primary_rule)
        self.video = video
        self.primary_rule = primary_rule
        self._layout = OutputLayout(root)
        self._dir = self._layout.work(video.video_id)
        _fresh_dir(self._dir)
        self._files = {name: open(self._dir / f'{name}.bin', 'wb') for name in _SPILL_COLUMNS}
        self._poses = NpyStreamWriter(self._dir / _STAGED_POSES)
        self._frames = self._dets = self._kpts = 0
        self._state = 'open'   # -> 'closed' (finalize started) -> 'done' (committed or aborted)

    @property
    def frames_written(self) -> int:
        return self._frames

    def append(self, start: int, dets: ChunkDets, poses: ChunkPoses) -> None:
        """Append one chunk. `start` (video frame index of the chunk's first frame) must equal
        frames_written; dets/poses are chunk-local (poses.det_index indexes dets rows) and are
        converted to video-global det rows here; num_persons comes from dets.num_persons()."""
        vid = self.video.video_id
        if self._state != 'open':
            raise RuntimeError(f'{vid}: the spill is closed')
        if start != self._frames:
            raise ValueError(f'{vid}: chunk starts at frame {start}, expected {self._frames}')
        if poses.num_frames != dets.num_frames:
            raise ValueError(f'{vid}: {dets.num_frames} det frames but {poses.num_frames} pose frames')
        if not (np.array_equal(poses.det_index, np.flatnonzero(dets.flags & POSED))
                and np.array_equal(np.diff(poses.offsets), dets.count_per_frame(POSED))):
            raise ValueError(f'{vid}: pose rows must be the POSED det rows of each frame (select.select_chunk)')
        if self._dets + len(dets.scores) > np.iinfo(np.int32).max:
            raise ValueError(f'{vid}: too many candidates for int32 kpt_det')
        self._write('det_ends', dets.offsets[1:] + self._dets)
        self._write('det_boxes', dets.boxes)
        self._write('det_scores', dets.scores)
        self._write('det_flags', dets.flags)
        self._write('num_persons', dets.num_persons())
        self._write('kpt_ends', poses.offsets[1:] + self._kpts)
        self._write('kpt_det', poses.det_index.astype(np.int64) + self._dets)
        self._write('kpts', poses.kpts)
        self._write('kpt_primary', poses.primary)
        self._poses.append(poses.primary_poses())
        self._frames += dets.num_frames
        self._dets += len(dets.scores)
        self._kpts += len(poses.det_index)

    def finalize(self, settings: Settings, provenance: Dict[str, object]) -> CommitInfo:
        """Close the spill, build the meta row (meta.compute_meta_row with T = frames_written),
        write persons/poses/meta/done via temp files, check invariants (raise on any problem),
        commit in the documented order and remove .work/<vid>/. `provenance` comes from
        make_provenance; its hashes (checked against `settings` and the rule) go into the done
        marker. On an exception, call abort()."""
        vid = self.video.video_id
        if self._state != 'open':
            raise RuntimeError(f'{vid}: the spill is closed')
        _check_provenance(provenance, settings, self.primary_rule)
        self._state = 'closed'
        for f in self._files.values():
            f.close()
        if self._poses.close() == 0:
            raise ValueError(f'{vid}: no frames to commit')
        info = self._stage(provenance)
        _publish(self._layout, self._dir, vid, drop_old_marker=True)
        self._state = 'done'
        return info

    def abort(self) -> None:
        """Delete .work/<vid>/ without touching committed outputs (idempotent)."""
        if self._state == 'done':
            return
        for f in self._files.values():
            f.close()
        self._poses.discard()
        shutil.rmtree(self._dir, ignore_errors=True)
        self._state = 'done'

    def _write(self, name: str, values: np.ndarray) -> None:
        self._files[name].write(np.ascontiguousarray(values, _SPILL_COLUMNS[name][0]).data)

    def _stage(self, provenance: Dict[str, object]) -> CommitInfo:
        """Build the record from memmaps of the spill columns, validate it, stage its files."""
        columns = {name: _read_column(self._dir / f'{name}.bin', dtype, shape)
                   for name, (dtype, shape) in _SPILL_COLUMNS.items()}
        zero = np.zeros(1, np.int64)
        v = self.video
        record = PersonsRecord(
            video_id=v.video_id, frame_size=(int(v.width), int(v.height)), fps=(int(v.fps_num), int(v.fps_den)),
            num_frames=self._frames, det_offsets=np.concatenate([zero, columns['det_ends']]),
            det_boxes=columns['det_boxes'], det_scores=columns['det_scores'], det_flags=columns['det_flags'],
            num_persons=columns['num_persons'], kpt_offsets=np.concatenate([zero, columns['kpt_ends']]),
            kpt_det=columns['kpt_det'], kpts=columns['kpts'], kpt_primary=columns['kpt_primary'],
            primary_rule=self.primary_rule, meta={}, provenance=provenance)
        record.meta = compute_meta_row(v.video_id, record.frame_size, record.fps, record.num_persons,
                                       record.kpt_offsets, record.kpts, record.kpt_primary)
        _raise_problems(record, np.load(self._dir / _STAGED_POSES, mmap_mode='r'))
        return _stage_files(self._dir, record)


def _read_column(path: Path, dtype, row_shape: Tuple[int, ...]) -> np.ndarray:
    row_bytes = np.dtype(dtype).itemsize * math.prod(row_shape)
    size = path.stat().st_size
    if size % row_bytes:
        raise ValueError(f'{path}: {size} bytes is not a whole number of rows')
    if size == 0:
        return np.zeros((0,) + row_shape, dtype)   # np.memmap cannot map an empty file
    return np.memmap(path, dtype=dtype, mode='r', shape=(size // row_bytes,) + row_shape)


def _raise_problems(record: PersonsRecord, poses: np.ndarray) -> None:
    problems = record.problems(poses)
    if problems:
        raise ValueError(f'{record.video_id}: invalid record: ' + '; '.join(problems))


def _stage_files(work: Path, record: PersonsRecord) -> CommitInfo:
    """Write persons.npz, meta.json and done.json (fsynced) next to the staged poses.npy."""
    save_persons(work / _STAGED_PERSONS, record)
    _write_text(work / _STAGED_META, _meta_json(record.meta) + '\n')
    info = CommitInfo(video_id=record.video_id, num_frames=record.num_frames,
                      extraction_hash=str(record.provenance['extraction_hash']),
                      derivation_hash=str(record.provenance['derivation_hash']),
                      poses_bytes=(work / _STAGED_POSES).stat().st_size,
                      persons_bytes=(work / _STAGED_PERSONS).stat().st_size)
    _write_text(work / _STAGED_DONE, json.dumps(dataclasses.asdict(info), sort_keys=True) + '\n')
    return info


def _publish(layout: OutputLayout, work: Path, video_id: str, drop_old_marker: bool) -> None:
    """Rename the staged files into place in the commit order (module docstring), remove `work`."""
    moves = ((work / _STAGED_PERSONS, layout.persons(video_id)), (work / _STAGED_POSES, layout.poses(video_id)),
             (work / _STAGED_META, layout.meta_json(video_id)))
    marker = layout.done_marker(video_id)
    for directory in [dst.parent for _, dst in moves] + [marker.parent]:
        directory.mkdir(parents=True, exist_ok=True)
    if drop_old_marker and marker.exists():
        marker.unlink()
        _fsync_dir(marker.parent)
    for src, dst in moves:
        os.replace(src, dst)
    for _, dst in moves:
        _fsync_dir(dst.parent)
    os.replace(work / _STAGED_DONE, marker)
    _fsync_dir(marker.parent)
    shutil.rmtree(work)


def _check_provenance(provenance: Dict[str, object], settings: Settings, primary_rule: str) -> None:
    """The done marker takes its hashes from `provenance`; they must describe this run."""
    want = dict(extraction_hash=extraction_hash(settings, provenance.get('model_provenance', {})),
                derivation_hash=derivation_hash(settings, primary_rule))
    for key, value in want.items():
        if provenance.get(key) != value:
            raise ValueError(f'provenance {key} does not match these settings and rule {primary_rule!r} '
                             f'(build it with make_provenance)')


def _check_rule(name: str) -> None:
    if name not in PRIMARY_RULES:
        raise KeyError(f'unknown primary rule {name!r}; known: {sorted(PRIMARY_RULES)}')


def _fresh_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _write_text(path: Path, text: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError as e:
        if e.errno != errno.EINVAL:   # some filesystems cannot fsync a directory; renames stay atomic
            raise
    finally:
        os.close(fd)


# --------------------------------------------------------------------------- hashes and provenance
def _sha256_json(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def extraction_hash(settings: Settings, model_provenance: Dict[str, object]) -> str:
    """sha256 hex of json.dumps({schema_version, settings.extraction_fields(), every *_sha256
    entry of engines.model_provenance(settings)}, sort_keys=True), as one flat dict."""
    fields = dict(schema_version=SCHEMA_VERSION, **settings.extraction_fields())
    fields.update({k: v for k, v in model_provenance.items() if k.endswith('_sha256')})
    return _sha256_json(fields)


def derivation_hash(settings: Settings, primary_rule: str) -> str:
    """sha256 hex of json.dumps(settings.derivation_fields(primary_rule), sort_keys=True)."""
    return _sha256_json(settings.derivation_fields(primary_rule))


def make_provenance(settings: Settings, primary_rule: str, model_provenance: Dict[str, object],
                    gpu_name: str) -> Dict[str, object]:
    """The `provenance` JSON: git_sha (env.git_sha), extraction_hash, derivation_hash,
    model_provenance (engines.model_provenance), libraries (env.library_versions), gpu_name,
    settings (dataclasses.asdict with repo_root as str)."""
    config = dataclasses.asdict(settings)
    config['repo_root'] = str(settings.repo_root)
    return dict(git_sha=env.git_sha(settings.repo_root), extraction_hash=extraction_hash(settings, model_provenance),
                derivation_hash=derivation_hash(settings, primary_rule), model_provenance=dict(model_provenance),
                libraries=env.library_versions(include_trt=settings.backend == 'trt'), gpu_name=gpu_name,
                settings=config)


# --------------------------------------------------------------------------- resume and failures
def resume_action(root: Path, video_id: str, extraction_hash_: str, derivation_hash_: str) -> str:
    """'skip' if the done marker matches both hashes and poses/persons exist with the recorded
    sizes; 'derive' if only derivation_hash differs (or the .state/meta row is missing);
    otherwise 'extract'."""
    layout = OutputLayout(root)
    try:
        marker = json.loads(layout.done_marker(video_id).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return 'extract'
    sizes_ok = all(_size(path) == marker.get(key) for path, key in
                   ((layout.poses(video_id), 'poses_bytes'), (layout.persons(video_id), 'persons_bytes')))
    if marker.get('extraction_hash') != extraction_hash_ or not sizes_ok:
        return 'extract'
    if marker.get('derivation_hash') != derivation_hash_ or not layout.meta_json(video_id).is_file():
        return 'derive'
    return 'skip'


def _size(path: Path) -> Optional[int]:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return None


def clear_work(root: Path) -> None:
    """Remove <root>/.work/ entirely (called once by the parent at run start)."""
    work = OutputLayout(root).work_root
    if work.exists():
        shutil.rmtree(work)


def record_failure(root: Path, video_id: str, error: str, traceback_text: str, attempt: int,
                   gpu: Optional[int]) -> None:
    """Append one line to .state/failed.jsonl (parent process only; flush + fsync)."""
    log = OutputLayout(root).failed_log
    log.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(video_id=video_id, error=error, traceback=traceback_text, attempt=attempt, gpu=gpu,
                           time=datetime.datetime.now().astimezone().isoformat(timespec='seconds')))
    with open(log, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
        f.flush()
        os.fsync(f.fileno())


# --------------------------------------------------------------------------- derive
def derive(root: Path, video_id: str, settings: Settings, primary_rule: str) -> CommitInfo:
    """CPU re-derivation from persons/<vid>.npz (spec D6, §4.4), no GPU.

    Recomputes COUNT_SET bits per frame with detpost.count_set_mask(det_boxes, det_scores,
    settings.count_score_thr, settings.count_nms_thr), num_persons, kpt_primary with
    select.PRIMARY_RULES[primary_rule] on each frame's POSE_SET rows (the pick must be a POSED
    row, else raise ValueError: the rule needs re-extraction), then poses, the meta row and the
    provenance's derivation_hash and count thresholds (in provenance['settings']); POSE_SET,
    POSED, det_* and kpts are unchanged. Commits in the documented order. With unchanged
    settings it reproduces every output file byte-for-byte.
    """
    _check_rule(primary_rule)
    layout = OutputLayout(root)
    work = layout.work(video_id)
    _fresh_dir(work)
    try:
        info = _stage_derived(layout, work, video_id, settings, primary_rule)
        _publish(layout, work, video_id, drop_old_marker=False)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return info


def _stage_derived(layout: OutputLayout, work: Path, video_id: str, settings: Settings,
                   primary_rule: str) -> CommitInfo:
    """Stage the re-derived files in `work`; the memmaps of the old record die with this call."""
    from .detpost import count_set_mask   # detpost imports torch; keep `record` importable without it
    old = load_persons(layout.persons(video_id), mmap_mode='r')
    damaged = old._structure_problems() + ([] if old.video_id == video_id else [f'holds video {old.video_id!r}'])
    if damaged:
        raise ValueError(f'{layout.persons(video_id)}: ' + '; '.join(damaged))
    t = old.num_frames
    offsets = np.asarray(old.det_offsets)
    boxes, scores = np.asarray(old.det_boxes), np.asarray(old.det_scores)
    flags = np.asarray(old.det_flags) & np.uint8(0xFF ^ COUNT_SET)
    for f in np.flatnonzero(np.diff(offsets)):
        rows = slice(offsets[f], offsets[f + 1])
        keep = count_set_mask(boxes[rows], scores[rows], settings.count_score_thr, settings.count_nms_thr)
        flags[offsets[f] + np.flatnonzero(keep)] |= COUNT_SET
    count = np.bincount(np.repeat(np.arange(t), np.diff(offsets))[(flags & COUNT_SET) != 0], minlength=t)
    if count.max(initial=0) > 255:
        raise ValueError(f'{video_id}: more than 255 people in the count set of one frame')
    position, not_posed = _rule_primary(old, primary_rule)
    if not_posed.any():
        frames = np.flatnonzero(not_posed)
        raise ValueError(f'{video_id}: rule {primary_rule!r} picks a person that was not posed in {len(frames)} '
                         f'frame(s), first {frames[0]}; this rule needs re-extraction')
    provenance = json.loads(json.dumps(old.provenance))
    provenance['derivation_hash'] = derivation_hash(settings, primary_rule)
    provenance['settings'].update(count_score_thr=settings.count_score_thr, count_nms_thr=settings.count_nms_thr)
    record = dataclasses.replace(old, det_flags=flags, num_persons=count.astype(np.uint8),
                                 kpt_primary=position.astype(np.int8), primary_rule=primary_rule,
                                 provenance=provenance)
    record.meta = compute_meta_row(video_id, record.frame_size, record.fps, record.num_persons,
                                   record.kpt_offsets, record.kpts, record.kpt_primary)
    writer = NpyStreamWriter(work / _STAGED_POSES)
    for _, block in record._pose_blocks():
        writer.append(block)
    writer.close()
    _raise_problems(record, np.load(work / _STAGED_POSES, mmap_mode='r'))
    return _stage_files(work, record)


# --------------------------------------------------------------------------- check
@dataclass
class CheckReport:
    """Result of `check` for one video."""

    video_id: str
    problems: List[str]                       # invariant violations; empty = OK
    primary_switches: List[int]               # frames t where the primary box jumps from t-1
    suggested_windows: List[Tuple[int, int]]  # [start, end) frame windows to render

    @property
    def ok(self) -> bool:
        return not self.problems


def check(root: Path, video_id: str, switch_iou: float = 0.3, window_s: float = 2.0) -> CheckReport:
    """Verify the §4.1 invariants for a committed video and flag primary switches.

    Besides PersonsRecord.problems: the poses file header and size, the done marker (hashes,
    sizes, T) and the .state/meta row against the record. A switch is a frame t where both t-1
    and t have a primary and IoU(primary box t-1, primary box t) < switch_iou. Suggested windows
    are [t - window_s*fps, t + window_s*fps) (rounded up to whole frames) clipped to [0, T) and
    merged when they overlap or touch.
    """
    layout = OutputLayout(root)
    paths = (layout.persons(video_id), layout.poses(video_id), layout.meta_json(video_id), layout.done_marker(video_id))
    missing = [f'missing {path}' for path in paths if not path.is_file()]
    if missing:
        return CheckReport(video_id, missing, [], [])
    try:
        record = load_persons(layout.persons(video_id), mmap_mode='r')
        poses = np.load(layout.poses(video_id), mmap_mode='r')
        problems = _file_problems(layout, video_id, record)
    except (OSError, ValueError, zipfile.BadZipFile) as e:
        return CheckReport(video_id, [f'unreadable output: {e}'], [], [])
    structural = record._structure_problems()
    if structural:
        return CheckReport(video_id, problems + structural, [], [])
    switches = _primary_switches(record, switch_iou)
    half = math.ceil(window_s * record.fps[0] / record.fps[1])
    return CheckReport(video_id, problems + record.problems(poses), switches,
                       _merge_windows(switches, half, record.num_frames))


def _file_problems(layout: OutputLayout, video_id: str, record: PersonsRecord) -> List[str]:
    out = []
    if record.video_id != video_id:
        out.append(f'the persons file holds video {record.video_id!r}')
    poses_path, persons_path = layout.poses(video_id), layout.persons(video_id)
    header = _npy_header(record.num_frames)
    with open(poses_path, 'rb') as f:
        if f.read(len(header)) != header:
            out.append('the poses file header is not the np.save v1.0 header of (T,133,3) float32')
    if poses_path.stat().st_size != len(header) + record.num_frames * NUM_KEYPOINTS * 3 * 4:
        out.append('the poses file size does not match T')
    expected = CommitInfo(video_id, record.num_frames, record.provenance.get('extraction_hash'),
                          record.provenance.get('derivation_hash'), poses_path.stat().st_size,
                          persons_path.stat().st_size)
    if json.loads(layout.done_marker(video_id).read_text(encoding='utf-8')) != dataclasses.asdict(expected):
        out.append('the done marker does not describe the committed files')
    if _canonical(json.loads(layout.meta_json(video_id).read_text(encoding='utf-8'))) != _canonical(record.meta):
        out.append('the .state/meta row differs from the persons meta')
    return out


def _primary_switches(record: PersonsRecord, switch_iou: float) -> List[int]:
    primary = np.asarray(record.kpt_primary)
    frames = np.flatnonzero(primary >= 0)
    rows = np.asarray(record.kpt_det)[np.asarray(record.kpt_offsets)[frames] + primary[frames]]
    boxes = np.asarray(record.det_boxes)[rows].astype(np.float64)
    pairs = np.flatnonzero(frames[1:] == frames[:-1] + 1)   # (frames[i], frames[i + 1]) adjacent
    iou = _iou(boxes[pairs], boxes[pairs + 1])
    return frames[pairs + 1][iou < switch_iou].tolist()


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise IoU of two (n,4) xyxy arrays (0 where the union is empty)."""
    w = np.clip(np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    h = np.clip(np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = w * h
    union = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _merge_windows(frames: List[int], half: int, num_frames: int) -> List[Tuple[int, int]]:
    windows: List[Tuple[int, int]] = []
    for t in frames:
        start, stop = max(0, t - half), min(num_frames, t + half)
        if windows and start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], stop)
        else:
            windows.append((start, stop))
    return windows
