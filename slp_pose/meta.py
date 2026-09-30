"""video_meta rows and the CSV writer (spec §4.1 `video_meta.csv`).

CPU only, no torch.

A meta row is a plain dict with exactly the keys of META_FIELDS and these Python types:
  video_id str; duration_s float = T * den / num (exact rational fps); width int; height int
  (original frame size); caption_source None (blank until the subtitles design);
  multi_person_ratio float = mean(num_persons >= 2); undetected_ratio float =
  mean(num_persons == 0) (num_persons from the COUNT set); extra_person_motion float, NaN when
  the misaligned-slt function would return None, 0.0 when there are no extra posed people.
The same row is stored as JSON (json.dumps(allow_nan=True), NaN token) in
`<root>/.state/meta/<vid>.json` and in the `meta` member of persons/<vid>.npz.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

# Byte-for-byte the header of misaligned-slt poses/pose_io.py::META_FIELDS.
META_FIELDS = ('video_id', 'duration_s', 'width', 'height', 'caption_source',
               'multi_person_ratio', 'undetected_ratio', 'extra_person_motion')
# COCO-WholeBody indices standing in for OpenPose-18 _OP_SHOULDERS (2, 5) and _OP_ARMS (3, 6, 4, 7).
COCO_SHOULDERS = (6, 5)        # right, left shoulder
COCO_ARMS = (8, 7, 10, 9)      # right elbow, left elbow, right wrist, left wrist
JOINT_SCORE_THR = 0.3          # a joint is valid only when its raw score > this
_MIN_OBSERVATIONS = 10         # per arm joint, as in the original
_SLOT_JOINTS = np.array(COCO_SHOULDERS + COCO_ARMS)


def duration_s(num_frames: int, fps_num: int, fps_den: int) -> float:
    """T * den / num as a Python float (written with repr)."""
    return int(num_frames) * int(fps_den) / int(fps_num)


def extra_person_motion(kpt_offsets: np.ndarray, kpts: np.ndarray, kpt_primary: np.ndarray) -> Optional[float]:
    """Port of misaligned-slt poses/signverse.py::extra_person_motion to the stored record.

    Args:
        kpt_offsets: (T+1,) int64 CSR offsets into kpts.
        kpts: (M,133,3) float32 saved encoding (x/W, y/H, raw score); may be a memmap (only the
            six shoulder and arm joints of the extra rows are read).
        kpt_primary: (T,) int8 primary position per frame, -1 = none.
    Slots: in frame t the non-primary posed rows, in stored order (= descending detector score),
    are slot 1, slot 2, ... (a frame without a primary has no extra slots). Per slot the joint
    coordinates are float64 (x/W, y/H), NaN where raw score <= JOINT_SCORE_THR; right/left
    shoulder = COCO_SHOULDERS, arm joints = COCO_ARMS; everything after that (shoulder-width
    normalisation with the 0.5 * median floor, >= 10 observations per joint, nanstd, max over
    joints and slots) is identical to the original.
    Returns:
        The largest per-slot arm variation; None when extra posed people exist but none is
        measurable; 0.0 when there are no extra posed people.
    """
    offsets = np.asarray(kpt_offsets, np.int64)
    frame = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    position = np.arange(offsets[-1]) - offsets[:-1][frame]   # position in the frame's posed list
    primary = np.asarray(kpt_primary, np.int64)[frame]
    extra = np.flatnonzero((primary >= 0) & (position != primary))
    if not len(extra):
        return 0.0
    slot = position[extra] + (position[extra] < primary[extra])
    joints = kpts[extra[:, None], _SLOT_JOINTS]               # (n, 6, 3) float32
    xy = joints[..., :2].astype(np.float64)
    # Compared in float32, like every numpy consumer of the saved scores: float32(0.3) is invalid.
    xy[~(joints[..., 2] > JOINT_SCORE_THR)] = np.nan
    motions = [m for m in (_slot_motion(xy[slot == s]) for s in np.unique(slot)) if m is not None]
    return max(motions) if motions else None


def _slot_motion(xy: np.ndarray) -> Optional[float]:
    """The original's per-slot computation, vectorised over the slot's frames.

    xy: (n, 6, 2) float64 in frame order: right shoulder, left shoulder, then COCO_ARMS. Every step
    matches the original element for element except the shoulder-width norm, which is vectorised
    here and a per-row np.linalg.norm (a BLAS dot) there; the two agree bit for bit on this
    machine's OpenBLAS, and tests/test_meta.py checks the final value for exact equality.
    """
    right, left = xy[:, 0], xy[:, 1]
    finite = np.flatnonzero(np.isfinite(right).all(axis=-1) & np.isfinite(left).all(axis=-1))
    width = np.linalg.norm(right[finite] - left[finite], axis=-1)
    wide = width > 1e-6
    rows, width = finite[wide], width[wide]
    if not len(rows):
        return None
    arms = xy[rows, 2:] - ((right[rows] + left[rows]) / 2.0)[:, None, :]
    # Each frame is scaled by its own shoulder width, floored at half the median width.
    arms = arms / np.maximum(width, 0.5 * float(np.median(width)))[:, None, None]
    seen = np.isfinite(arms).all(axis=-1)
    keep = seen.sum(axis=0) >= _MIN_OBSERVATIONS
    if not keep.any():
        return None
    sd = np.nanstd(np.where(seen[..., None], arms, np.nan)[:, keep], axis=0)
    return float(np.nanmax(sd))


def compute_meta_row(video_id: str, frame_size: Tuple[int, int], fps: Tuple[int, int], num_persons: np.ndarray,
                     kpt_offsets: np.ndarray, kpts: np.ndarray, kpt_primary: np.ndarray) -> Dict[str, object]:
    """The meta row of one video (see module docstring); T = len(num_persons) must be > 0.

    frame_size = (W, H), fps = (num, den), num_persons (T,) uint8 count-set sizes.
    """
    num_persons = np.asarray(num_persons)
    t = len(num_persons)
    if t == 0:
        raise ValueError(f'{video_id}: a meta row needs at least one frame')
    if len(kpt_offsets) != t + 1 or len(kpt_primary) != t:
        raise ValueError(f'{video_id}: kpt_offsets / kpt_primary do not cover {t} frames')
    motion = extra_person_motion(kpt_offsets, kpts, kpt_primary)
    return dict(video_id=str(video_id), duration_s=duration_s(t, fps[0], fps[1]),
                width=int(frame_size[0]), height=int(frame_size[1]), caption_source=None,
                multi_person_ratio=int(np.count_nonzero(num_persons >= 2)) / t,
                undetected_ratio=int(np.count_nonzero(num_persons == 0)) / t,
                extra_person_motion=math.nan if motion is None else float(motion))


def _cell(value: object) -> str:
    # What csv.writer writes for a raw value: '' for None, else str() (== repr for floats).
    return '' if value is None else str(value)


def csv_cells(row: Mapping[str, object]) -> List[str]:
    """The 8 CSV cells of a row, formatted exactly like misaligned-slt `save_video_meta`:
    duration_s via str(float) (== repr), width/height as int ('' if None), caption_source ''
    if None, the two ratios f'{x:.4f}', extra_person_motion repr(float(x)) ('nan' for NaN).
    """
    def ratio(key: str) -> str:
        value = row.get(key)
        return '' if value is None else f'{float(value):.4f}'

    motion = row.get('extra_person_motion')
    return [str(row['video_id']), _cell(row.get('duration_s')), _cell(row.get('width')), _cell(row.get('height')),
            str(row.get('caption_source') or ''), ratio('multi_person_ratio'), ratio('undetected_ratio'),
            '' if motion is None else repr(float(motion))]


def write_video_meta(path: Path, rows: Mapping[str, Mapping[str, object]]) -> None:
    """Write `<root>/video_meta.csv` atomically (tmp file + fsync + os.replace).

    Output bytes equal misaligned-slt `save_video_meta(path, rows)`: csv.writer, header
    META_FIELDS, rows sorted by video_id, '\\r\\n' line ends, utf-8. Only the parent process calls this.
    """
    path = Path(path)
    lines = []
    for video_id in sorted(rows):
        # save_video_meta writes the dict key as the id; a row must agree with its key.
        if rows[video_id].get('video_id') != video_id:
            raise ValueError(f'meta row keyed {video_id!r} has video_id {rows[video_id].get("video_id")!r}')
        lines.append(csv_cells(rows[video_id]))
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(META_FIELDS)
        writer.writerows(lines)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_meta_rows(state_meta_dir: Path) -> Dict[str, Dict[str, object]]:
    """{video_id: row} from every `<root>/.state/meta/<vid>.json` (the CSV merge input)."""
    rows = {}
    for path in sorted(Path(state_meta_dir).glob('*.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        if row.get('video_id') != path.stem:
            raise ValueError(f'{path}: video_id {row.get("video_id")!r} does not match the file name')
        rows[path.stem] = row
    return rows
