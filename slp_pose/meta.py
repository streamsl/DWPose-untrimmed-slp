"""video_meta rows and the CSV writer.

CPU only, no torch.

A meta row is a plain dict with exactly these keys, in this order:
  video_id str; duration_s float = T * den / num (exact rational fps); width int; height int
  (original frame size); caption_source None (run.merge_meta fills it from .state/captions.json
  once `slp-pose subs` has run);
  undetected_ratio float = share of frames with nobody in the count set (num_persons == 0, the
  whole frame; misaligned-slt's audit value, no rule reads it);
  masked_runs, empty_runs, handless_runs: lists of [a, b] (Python ints), half-open frame runs on
  the poses/ grid, ascending and disjoint; [] = measured, no run.
The three run kinds are misaligned-slt prepare_yt25.py's masked_runs / empty_runs /
handless_runs at commit MSLT_COMMIT, with the same algorithm and constants, applied to our record.
The constants are seconds, as there; each video converts them to frames of its poses/ grid with
seconds_to_frames at the rate misaligned-slt's loader reads for it, T / duration_s (loader_fps;
0.5 / 1 / 2 s = 13 / 25 / 50 frames at 25 fps, 15 / 30 / 60 at 29.97 fps):
  the kept body = the primary (the poses/ row, kpt_primary; none where it is -1);
  a person = a posed row; its body = COCO joints 0-16 plus a neck (the shoulders' midpoint,
    visible when both shoulders are), the OpenPose-18 body of upstream DWPose; a joint is visible
    when its raw score > JOINT_SCORE_THR and it lies on screen (x/W and y/H in [-0.25, 1.25]);
    a real body has >= REAL_BODY_JOINTS visible joints;
  masked_runs: frames with >= 2 real bodies among the frame's candidates, scored by the arm
    motion of the real candidates other than the kept body (in the kept body's shoulder widths),
    each followed as a person across the frames (by its shoulder midpoint, not its detector slot);
    the candidates are every posed row for a per-frame primary rule, and for a video-level rule
    (signer.SignerChoice.region) the posed rows whose box centre lies in the frame's
    interpreter region, plus the primary (see people_in_region);
  empty_runs: frames whose kept body is not a real body (no primary, or too few visible joints);
  handless_runs: frames whose poses/ row shows no hand.
A GPU worker's commit holds the whole-frame row of its per-frame extraction rule; the parent's
`derive` replaces it with the row of the dataset's primary rule. The row is stored as JSON
(json.dumps(allow_nan=True), NaN token) in `<root>/.state/meta/<vid>.json` and in the `meta` member
of persons/<vid>.npz.

The CSV (video_meta.csv, and one video_meta_<name>.csv per other split file: subs.write_meta_files)
is misaligned-slt's poses/pose_io.py contract: header META_FIELDS, bytes equal to its
`save_video_meta`. Its columns:
  the meta row's (csv_row), without masked_runs: misaligned-slt does not read it (it stays in the
    meta row, inside the record);
  switch_runs, default_signing_runs: always []. They mark misaligned-slt's own signer switch, which
    never runs on these outputs (the dataset's signer rule picks the signer), so no unit is in doubt;
  copy_of: written by misaligned-slt's copy check; slp-pose never computes it and keeps it;
  unreliable_spans: [start_s, end_s] spans in seconds where the file's subtitle track does not hold
    the signing (Dataset.unreliable_spans; merge_spans); [] = checked, none; blank = unknown;
  any column that META_FIELDS does not name: kept, after them (rewrite_video_meta).
"""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Byte-for-byte misaligned-slt poses/pose_io.py RUN_FIELDS, JSON_FIELDS and META_FIELDS (the CSV header).
RUN_FIELDS = ('empty_runs', 'handless_runs', 'switch_runs', 'default_signing_runs')
JSON_FIELDS = RUN_FIELDS + ('copy_of', 'unreliable_spans')
META_FIELDS = ('video_id', 'duration_s', 'width', 'height', 'caption_source', 'undetected_ratio') + JSON_FIELDS
RETIRED_FIELDS = ('masked_runs',)   # columns of older files that misaligned-slt does not read: dropped on reading
# The CSV columns that a meta row gives (csv_row); switch_runs and default_signing_runs are always [].
ROW_COLUMNS = ('video_id', 'duration_s', 'width', 'height', 'caption_source', 'undetected_ratio', 'empty_runs',
               'handless_runs')
# Version of the meta definitions: stored as provenance['meta_version'] by `derive` and hashed into
# every rule's derivation_hash (record.py); a GPU worker's commit has none. Bump META_VERSION whenever a
# definition below changes. The CSV columns are not stored in the record, so a change of them alone needs
# no bump (`slp-pose meta` rewrites the files).
META_VERSION = 5
# The misaligned-slt commit whose prepare_yt25.py run definitions this module follows; `derive` stores
# it as provenance['mslt_commit'].
MSLT_COMMIT = '3eb85af'
JOINT_SCORE_THR = 0.3          # a joint is visible / valid only when its raw score > this (float32 compare)
# misaligned-slt prepare_yt25.py constants (times in seconds, as there; seconds_to_frames converts
# them per video) and configs/data.yaml poses.min_extra_person_motion.
REAL_BODY_JOINTS = 8           # _REAL_BODY_JOINTS: visible body joints (of 18) of a real body
COORD_TOLERANCE = 0.25         # _COORD_TOLERANCE: x/W or y/H outside [-tol, 1 + tol] is off screen
MASK_RUN_S = 0.5               # _MASK_RUN_S: runs closer than this merge; shorter masked runs drop
EMPTY_RUN_S = 1.0              # _EMPTY_RUN_S: shorter empty runs drop
MOTION_BLOCK_S = 2.0           # _MOTION_BLOCK_S: masked runs are scored per window this long, half-window steps
HAND_CONF = 0.3                # _HAND_CONF: a hand keypoint counts when its score is above this
HAND_MIN_POINTS = 10           # _HAND_MIN_POINTS: of a hand's 21 keypoints
HANDLESS_RUN_S = MASK_RUN_S    # _HANDLESS_RUN_S: shorter handless runs drop (never merged)
SLOT_JUMP = 0.10               # _SLOT_JUMP: an extra body joins the nearest person this close (frame heights)
MIN_EXTRA_PERSON_MOTION = 0.7  # data.yaml poses.min_extra_person_motion: a window moving more is masked
_MIN_OBSERVATIONS = 10         # per arm joint (_arm_motion)
_MIN_SHOULDER_WIDTH = 1e-6     # _shoulders: a narrower shoulder pair is no shoulder pair
# COCO-WholeBody indices standing in for OpenPose-18 _OP_SHOULDERS (2, 5) and _OP_ARMS (3, 6, 4, 7).
COCO_SHOULDERS = (6, 5)        # right, left shoulder
COCO_ARMS = (8, 7, 10, 9)      # right elbow, left elbow, right wrist, left wrist
COCO_BODY = 17                 # COCO joints 0-16 = OpenPose-18 without the neck (OPENPOSE18_TO_COCO17)
COCO_HANDS = (slice(91, 112), slice(112, 133))   # left, right hand
_SLOT_JOINTS = np.array(COCO_SHOULDERS + COCO_ARMS)
_KEPT_COLUMNS = np.r_[0:COCO_BODY, 91:133]   # what empty_runs / handless_runs read of the kept body
_BLOCK_ROWS = 16384            # posed rows read from kpts at a time

Runs = List[List[int]]


# --------------------------------------------------------------------------- meta row
def meta_row(video_id: str, frame_size: Tuple[int, int], fps: Tuple[int, int], num_persons: np.ndarray,
             kpt_offsets: np.ndarray, kpts: np.ndarray, kpt_primary: np.ndarray,
             candidates: Optional[np.ndarray] = None) -> Dict[str, object]:
    """The meta row of one video (module docstring); T = len(num_persons) must be > 0.

    frame_size = (W, H), fps = (num, den), num_persons (T,) uint8 count-set sizes; kpt_offsets,
    kpts (may be a memmap, read in blocks) and kpt_primary as in the persons record. `candidates`:
    None (per-frame rule, the whole frame) or the (M,) bool posed rows that take part in masked_runs
    (the primary rows always do).
    """
    num_persons = np.asarray(num_persons)
    t = len(num_persons)
    if t == 0:
        raise ValueError(f'{video_id}: a meta row needs at least one frame')
    if len(kpt_offsets) != t + 1 or len(kpt_primary) != t:
        raise ValueError(f'{video_id}: kpt_offsets / kpt_primary do not cover {t} frames')
    real, hand = _kept_body(kpt_offsets, kpts, kpt_primary)
    rate = loader_fps(t, fps)
    handless_run = seconds_to_frames(HANDLESS_RUN_S, rate)
    return dict(video_id=str(video_id), duration_s=duration_s(t, fps[0], fps[1]),
                width=int(frame_size[0]), height=int(frame_size[1]), caption_source=None,
                undetected_ratio=int(np.count_nonzero(num_persons == 0)) / t,
                masked_runs=masked_runs(kpt_offsets, kpts, kpt_primary, rate, frame_size, candidates),
                empty_runs=empty_runs(~real, rate),
                handless_runs=[[a, b] for a, b in _edges(~hand) if b - a >= handless_run])


def seconds_to_frames(seconds: float, fps: float) -> int:
    """misaligned-slt utils.seconds_to_frames at one rate: `seconds` as a frame count at `fps`, at
    least 1, rounded up with its 0.01-frame slack."""
    return max(1, math.ceil(float(seconds) * float(fps) - 0.01))


def loader_fps(num_frames: int, fps: Tuple[int, int]) -> float:
    """The frame rate misaligned-slt reads for a video of our output, T / duration_s (poses/pose_io.py
    build_pose_index on our video_meta row), at which its run rules convert seconds to frames."""
    return int(num_frames) / duration_s(num_frames, fps[0], fps[1])


def visible_joints(joints: np.ndarray) -> np.ndarray:
    """(..., 3) saved encoding -> (...) bool: raw score > JOINT_SCORE_THR and on screen. What
    misaligned-slt's _part_arrays keeps of an upstream DWPose body joint (score index code ->
    1.0 when the DWPose confidence > 0.3, off-screen coordinates -> score 0). Compared in float32."""
    x, y = joints[..., 0], joints[..., 1]
    off = (x < -COORD_TOLERANCE) | (x > 1 + COORD_TOLERANCE) | (y < -COORD_TOLERANCE) | (y > 1 + COORD_TOLERANCE)
    return (joints[..., 2] > JOINT_SCORE_THR) & ~off


def real_bodies(body: np.ndarray) -> np.ndarray:
    """(n, >=17, 3) COCO joints (the first 17 are read) -> (n,) bool real body: at least
    REAL_BODY_JOINTS visible of the 17 COCO body joints plus the neck. The neck is upstream DWPose's:
    the float32 midpoint of the shoulders (COCO 5, 6), visible when both shoulders score > 0.3 and
    the midpoint is on screen."""
    body = np.asarray(body)
    left, right = body[:, 5], body[:, 6]
    neck = np.empty((len(body), 3), np.float32)
    neck[:, :2] = (left[:, :2] + right[:, :2]) / np.float32(2)
    neck[:, 2] = np.where((left[:, 2] > JOINT_SCORE_THR) & (right[:, 2] > JOINT_SCORE_THR), 1, 0)
    seen = visible_joints(body[:, :COCO_BODY]).sum(axis=1) + visible_joints(neck)
    return seen >= REAL_BODY_JOINTS


def masked_runs(kpt_offsets: np.ndarray, kpts: np.ndarray, kpt_primary: np.ndarray, fps: float,
                frame_size: Tuple[int, int], candidates: Optional[np.ndarray] = None,
                min_motion: float = MIN_EXTRA_PERSON_MOTION) -> Runs:
    """Port of misaligned-slt prepare_yt25.masked_runs to the record; `fps` is the rate the run
    constants are converted at (loader_fps), frame_size = (W, H).

    Frame t's payload is its candidate rows (`candidates` None = every posed row; else those rows
    plus the primary row) in stored order (= descending detector score), slot k = the k-th of them;
    the kept slot is the primary's (none where kpt_primary == -1: every real candidate is then an
    extra body and no kept shoulder width is known). A frame qualifies with >= 2 real bodies
    (real_bodies); qualifying frames at most MASK_RUN_S apart form one run [first, last + 1), runs
    shorter than MASK_RUN_S drop; each run is scored in MOTION_BLOCK_S windows at half-window
    steps (block // 2 frames), the last ending at the run end, a window unmeasurable on its own
    takes the whole run's motion, and a window is cut when that is None or > min_motion; cut
    windows that overlap or touch merge. Motion (_arm_motion) as there: the elbows and wrists
    (COCO_ARMS) of every real extra body with both shoulders visible, minus its shoulder midpoint,
    over the kept body's shoulder width in that frame (or the median width, floored at half the
    median), pooled per person (_person_tracks); >= 10 observations per joint; the largest nanstd.
    """
    mask_run, block = seconds_to_frames(MASK_RUN_S, fps), seconds_to_frames(MOTION_BLOCK_S, fps)
    people = _candidates(kpt_offsets, kpts, kpt_primary, candidates, frame_size)
    qualifying = np.flatnonzero(people.real_count >= 2)
    runs: Runs = []
    if not len(qualifying):
        return runs
    cut = np.flatnonzero(np.diff(qualifying) > mask_run) + 1
    starts = qualifying[np.concatenate([[0], cut])].tolist()
    ends = (qualifying[np.concatenate([cut - 1, [len(qualifying) - 1]])] + 1).tolist()
    for a, b in zip(starts, ends):
        if b - a < mask_run:
            continue
        whole = _arm_motion(people, a, b)
        for lo in list(range(a, b - block, block // 2)) + [max(a, b - block)]:
            hi = min(lo + block, b)
            motion = _arm_motion(people, lo, hi)
            if motion is None:
                motion = whole
            if motion is None or motion > min_motion:
                if runs and runs[-1][1] >= lo:
                    runs[-1][1] = max(runs[-1][1], hi)
                else:
                    runs.append([lo, hi])
    return runs


def empty_runs(empty: np.ndarray, fps: float) -> Runs:
    """Port of misaligned-slt prepare_yt25.empty_runs on a (T,) bool 'no real body' per frame, with
    the constants converted at `fps`: runs of empty frames closer than MASK_RUN_S merge (before the
    length filter), then runs shorter than EMPTY_RUN_S drop. The record's empty frames are those
    whose kept body is not a real body."""
    mask_run, empty_run = seconds_to_frames(MASK_RUN_S, fps), seconds_to_frames(EMPTY_RUN_S, fps)
    runs: Runs = []
    for a, b in _edges(np.asarray(empty, bool)):
        if runs and a - runs[-1][1] < mask_run:
            runs[-1][1] = b
        else:
            runs.append([a, b])
    return [run for run in runs if run[1] - run[0] >= empty_run]


def _edges(mask: np.ndarray) -> List[Tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _kept_body(kpt_offsets: np.ndarray, kpts: np.ndarray, kpt_primary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per frame: (the kept body is a real body, the poses/ row shows a hand); False without a primary."""
    offsets, primary = np.asarray(kpt_offsets, np.int64), np.asarray(kpt_primary, np.int64)
    frames = np.flatnonzero(primary >= 0)
    rows = offsets[:-1][frames] + primary[frames]
    real, hand = np.zeros(len(primary), bool), np.zeros(len(primary), bool)
    for s in range(0, len(rows), _BLOCK_ROWS):
        block = np.asarray(kpts[rows[s:s + _BLOCK_ROWS, None], _KEPT_COLUMNS])   # (n, 17 + 42, 3)
        left, right = block[:, COCO_BODY:COCO_BODY + 21, 2], block[:, COCO_BODY + 21:, 2]
        real[frames[s:s + _BLOCK_ROWS]] = real_bodies(block)
        hand[frames[s:s + _BLOCK_ROWS]] = (((left > HAND_CONF).sum(axis=1) >= HAND_MIN_POINTS)
                                           | ((right > HAND_CONF).sum(axis=1) >= HAND_MIN_POINTS))
    return real, hand


@dataclass(frozen=True)
class _Candidates:
    """What masked_runs reads of each frame's candidates (prepare_yt25._real_bodies / _extra_arms)."""

    real_count: np.ndarray   # (T,) int64 real bodies among the candidates
    extra: np.ndarray        # (T,) bool: a real candidate other than the kept body that shows an arm joint
    kept_width: np.ndarray   # (T,) float64 the kept body's shoulder width, NaN = unknown
    arm_frame: np.ndarray    # (A,) int64 frame of each measurable extra body, ascending (then slot order)
    arm_centre: np.ndarray   # (A, 2) float64 its shoulder midpoint in frame heights: (x/W * W/H, y/H)
    arms: np.ndarray         # (A, 4, 2) float64 its COCO_ARMS minus its shoulder midpoint (NaN = not visible)


def _candidates(kpt_offsets: np.ndarray, kpts: np.ndarray, kpt_primary: np.ndarray,
                candidates: Optional[np.ndarray], frame_size: Tuple[int, int]) -> _Candidates:
    offsets, primary = np.asarray(kpt_offsets, np.int64), np.asarray(kpt_primary, np.int64)
    t, m = len(primary), int(offsets[-1])
    frame = np.repeat(np.arange(t), np.diff(offsets))
    kept = np.zeros(m, bool)
    has = primary >= 0
    kept[offsets[:-1][has] + primary[has]] = True
    if candidates is None:
        rows = np.arange(m)
    else:
        candidates = np.asarray(candidates, bool)
        if candidates.shape != (m,):
            raise ValueError(f'candidates has shape {candidates.shape}, expected ({m},)')
        rows = np.flatnonzero(candidates | kept)
    row_frame = frame[rows]
    aspect = frame_size[0] / frame_size[1]   # W/H, as _extra_arms reads it from the payload's frame size
    real = np.zeros(len(rows), bool)
    armed = np.zeros(len(rows), bool)   # an elbow or wrist is visible (_extra_arms)
    width = np.full(len(rows), np.nan)
    measured: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for s in range(0, len(rows), _BLOCK_ROWS):
        block = slice(s, s + _BLOCK_ROWS)
        joints = np.asarray(kpts[rows[block, None], np.arange(COCO_BODY)])   # (n, 17, 3) float32
        real[block] = real_bodies(joints)
        armed[block] = visible_joints(joints[:, list(COCO_ARMS)]).any(axis=1)
        xy = joints[:, _SLOT_JOINTS, :2].astype(np.float64)
        xy[~(visible_joints(joints[:, _SLOT_JOINTS]) & np.isfinite(xy).all(axis=-1))] = np.nan
        right, left = xy[:, 0], xy[:, 1]
        both = np.flatnonzero(np.isfinite(right).all(axis=-1) & np.isfinite(left).all(axis=-1))
        norm = np.linalg.norm(right[both] - left[both], axis=-1)
        wide = both[norm > _MIN_SHOULDER_WIDTH]
        width[s + wide] = norm[norm > _MIN_SHOULDER_WIDTH]
        extra = wide[real[s + wide] & armed[s + wide] & ~kept[rows[s + wide]]]
        middle = (right[extra] + left[extra]) / 2.0
        measured.append((s + extra, xy[extra, 2:] - middle[:, None, :], middle * (aspect, 1.0)))
    is_kept = kept[rows]
    kept_width = np.full(t, np.nan)
    kept_width[row_frame[is_kept]] = width[is_kept]
    index = np.concatenate([i for i, _, _ in measured] + [np.zeros(0, np.int64)])
    return _Candidates(real_count=np.bincount(row_frame[real], minlength=t).astype(np.int64),
                       extra=np.bincount(row_frame[real & armed & ~is_kept], minlength=t) > 0, kept_width=kept_width,
                       arm_frame=row_frame[index],
                       arm_centre=np.concatenate([c for _, _, c in measured] + [np.zeros((0, 2))]),
                       arms=np.concatenate([a for _, a, _ in measured] + [np.zeros((0, 4, 2))]))


def _arm_motion(people: _Candidates, lo: int, hi: int) -> Optional[float]:
    """prepare_yt25._arm_motion over the frames [lo, hi): None when extra bodies exist but cannot be
    measured (or no kept width is known), 0.0 without extra bodies. The arms are pooled per person,
    linked over these frames only (_person_tracks)."""
    has_extra = bool(people.extra[lo:hi].any())
    widths = people.kept_width[lo:hi]
    known = widths[~np.isnan(widths)]
    if not len(known):
        return None if has_extra else 0.0
    median = float(np.median(known))
    first, last = np.searchsorted(people.arm_frame, [lo, hi])
    scale = np.maximum(np.where(np.isnan(widths), median, widths), 0.5 * median)
    arms = people.arms[first:last] / scale[people.arm_frame[first:last] - lo][:, None, None]
    person = _person_tracks(people.arm_frame[first:last], people.arm_centre[first:last])
    scores = []
    for k in range(int(person.max(initial=-1)) + 1):
        joints = arms[person == k]
        seen = np.isfinite(joints).all(axis=-1)
        keep = seen.sum(axis=0) >= _MIN_OBSERVATIONS
        if not keep.any():
            continue
        sd = np.nanstd(np.where(seen[..., None], joints, np.nan)[:, keep], axis=0)
        scores.append(float(np.nanmax(sd)))
    return max(scores) if scores else (None if has_extra else 0.0)


def _person_tracks(frame: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """prepare_yt25._person_tracks: the person (0, 1, ... in order of appearance) of each arm row,
    the rows in frame order, then slot order. A row joins the nearest person whose last centre is
    within SLOT_JUMP and whom no earlier row of its frame joined (ties: the earlier person), else it
    starts a new person; a person never expires. Detector slots re-order every frame, so pooling by
    slot would make two still people who swap ranks look like one moving person."""
    person = np.empty(len(frame), np.int64)
    last = np.empty((len(frame), 2))   # each person's last centre
    taken = np.zeros(len(frame), bool)
    count = known = 0                 # people so far / at the frame's start (only those can be joined)
    for i in range(len(frame)):
        if i == 0 or frame[i] != frame[i - 1]:
            known = count
            taken[:known] = False
        # float64 norms of 2-vectors, equal to the scalar np.linalg.norm prepare_yt25 takes per person
        distance = np.where(taken[:known], np.inf, np.linalg.norm(last[:known] - centre[i], axis=-1))
        k = int(np.argmin(distance)) if known else -1
        if k >= 0 and distance[k] <= SLOT_JUMP:
            taken[k] = True
        else:
            k, count = count, count + 1
        last[k] = centre[i]
        person[i] = k
    return person


def people_in_region(region: np.ndarray, det_offsets: np.ndarray, det_boxes: np.ndarray,
                     kpt_det: np.ndarray) -> np.ndarray:
    """(M,) bool: the posed rows (kpts order) inside their frame's interpreter region, masked_runs'
    candidates for a video-level rule. A person is inside when the centre of their box,
    ((x1 + x2) / 2, (y1 + y2) / 2) in float64, lies in the region box, edges included.

    Args:
        region: (T,4) xyxy original pixels per frame, NaN rows = no region (signer.SignerChoice.region).
        det_offsets, det_boxes: the record's candidates.
        kpt_det: (M,) video-global det row of each posed row.
    """
    region = np.asarray(region, np.float64)
    offsets = np.asarray(det_offsets, np.int64)
    num_frames = len(region)
    if region.shape != (num_frames, 4) or len(offsets) != num_frames + 1:
        raise ValueError(f'region {region.shape} and det_offsets {offsets.shape} do not cover the same frames')
    frame = np.repeat(np.arange(num_frames), np.diff(offsets))
    rows = np.flatnonzero(~np.isnan(region).any(axis=1)[frame])   # candidates of frames with a region
    boxes, box = np.asarray(det_boxes[rows], np.float64), region[frame[rows]]
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    inside = np.zeros(int(offsets[-1]), bool)
    inside[rows] = (cx >= box[:, 0]) & (cx <= box[:, 2]) & (cy >= box[:, 1]) & (cy <= box[:, 3])
    return inside[np.asarray(kpt_det, np.int64)]


def duration_s(num_frames: int, fps_num: int, fps_den: int) -> float:
    """T * den / num as a Python float (written with repr)."""
    return int(num_frames) * int(fps_den) / int(fps_num)


# --------------------------------------------------------------------------- CSV
def _cell(value: object) -> str:
    # What csv.writer writes for a raw value: '' for None, else str() (== repr for floats).
    return '' if value is None else str(value)


def csv_cells(row: Mapping[str, object]) -> List[str]:
    """The META_FIELDS cells of a row, formatted exactly like misaligned-slt `save_video_meta`:
    duration_s via str(float) (== repr), width/height as int, caption_source as text,
    undetected_ratio f'{x:.4f}', each JSON column json.dumps; '' for None or a missing key.
    """
    ratio = row.get('undetected_ratio')
    return [str(row['video_id']), _cell(row.get('duration_s')), _cell(row.get('width')), _cell(row.get('height')),
            str(row.get('caption_source') or ''), '' if ratio is None else f'{float(ratio):.4f}',
            *('' if row.get(key) is None else json.dumps(row[key]) for key in JSON_FIELDS)]


def csv_row(row: Mapping[str, object]) -> Dict[str, object]:
    """The CSV columns that a video's meta row gives (`row`: a meta row, or a row of read_video_meta):
    ROW_COLUMNS, and switch_runs and default_signing_runs = [] (module docstring). The other columns
    are not in it, so rewrite_video_meta keeps them."""
    return dict({key: row.get(key) for key in ROW_COLUMNS}, switch_runs=[], default_signing_runs=[])


def merge_spans(spans: Iterable[Tuple[object, object]]) -> List[List[float]]:
    """unreliable_spans as misaligned-slt reads it: [start_s, end_s] float seconds, sorted, with
    spans that overlap or touch merged and empty ones (end <= start) left out."""
    out: List[List[float]] = []
    for start, end in sorted((float(a), float(b)) for a, b in spans):
        if end <= start:
            continue
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def write_video_meta(path: Path, rows: Mapping[str, Mapping[str, object]], extra: Sequence[str] = ()) -> None:
    """Write a video_meta CSV atomically (tmp file + fsync + os.replace): header META_FIELDS and then
    the `extra` columns (each row's text there, '' where it has none), rows sorted by video_id.

    Without extra columns the bytes equal misaligned-slt `save_video_meta(path, rows)`: csv.writer,
    '\\r\\n' line ends, utf-8. Only the parent process calls this.
    """
    path = Path(path)
    lines = []
    for video_id in sorted(rows):
        # save_video_meta writes the dict key as the id; a row must agree with its key.
        if rows[video_id].get('video_id') != video_id:
            raise ValueError(f'meta row keyed {video_id!r} has video_id {rows[video_id].get("video_id")!r}')
        lines.append(csv_cells(rows[video_id]) + [str(rows[video_id].get(key) or '') for key in extra])
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(META_FIELDS + tuple(extra))
        writer.writerows(lines)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def rewrite_video_meta(path: Path, rows: Mapping[str, Mapping[str, object]]) -> None:
    """write_video_meta of `rows` that keeps what slp-pose does not compute: a column that a row lacks
    (copy_of, which misaligned-slt's copy check writes; unreliable_spans where it is unknown here; a
    column that META_FIELDS does not name) takes the value of the same video's row in the file at
    `path`, and is blank for a video the file does not list. The file then lists exactly `rows`."""
    old, extra = read_video_meta(path)
    write_video_meta(path, {vid: dict(old.get(vid, {}), **row) for vid, row in rows.items()}, extra)


def read_video_meta(path: Path) -> Tuple[Dict[str, Dict[str, object]], List[str]]:
    """({video_id: row}, extra columns) of a video_meta CSV of any version, written by slp-pose or by
    misaligned-slt; ({}, []) when there is no file. A row holds every cell of its line: the
    META_FIELDS ones as the values that csv_cells writes back to the same text (None for a blank
    cell), any other as its text. extra: the file's columns outside META_FIELDS, in file order.
    RETIRED_FIELDS are left out of both."""
    path = Path(path)
    if not path.is_file():
        return {}, []
    with open(path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        extra = [key for key in reader.fieldnames or () if key not in META_FIELDS + RETIRED_FIELDS]
        rows = {}
        for line in reader:
            row = {key: _value(key, text or '') for key, text in line.items()
                   if key is not None and key not in RETIRED_FIELDS}
            if row.get('video_id'):
                rows[row['video_id']] = row
    return rows, extra


def _value(key: str, text: str) -> object:
    """One cell of a video_meta CSV as read_video_meta gives it."""
    if key == 'video_id' or key not in META_FIELDS:
        return text
    if not text:
        return None
    if key in JSON_FIELDS:
        return json.loads(text)
    if key in ('width', 'height'):
        return int(float(text))
    return float(text) if key in ('duration_s', 'undetected_ratio') else text


def load_meta_rows(state_meta_dir: Path) -> Dict[str, Dict[str, object]]:
    """{video_id: row} from every `<root>/.state/meta/<vid>.json` (the CSV merge input)."""
    rows = {}
    for path in sorted(Path(state_meta_dir).glob('*.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        if row.get('video_id') != path.stem:
            raise ValueError(f'{path}: video_id {row.get("video_id")!r} does not match the file name')
        rows[path.stem] = row
    return rows
