"""Posed-slot selection and per-frame primary-signer rules (spec §4.2 select.py, assumption 4, D4, D14).

Pure numpy on the CPU; must not import torch.

Terms (all per frame, rows in det order = descending detector score, see types.ChunkDets):
- pose set S: rows with the POSE_SET bit.
- posed rows: at most K rows of S that get keypoints. Priority list
  [S[rule(S)], S[largest_bbox(S)], S[highest_score(S)]] + the remaining S rows in det order,
  de-duplicated keeping the first occurrence, truncated to K, then sorted ascending.
  So the primary is always posed, and both built-in rules' picks are posed whenever K >= 3,
  which is what lets `record.derive` switch between per-frame rules, or apply a video-level
  rule (signer.py) to the posed people, on the CPU.
- primary position: index of the primary row inside the sorted posed list (int8), -1 = none.
Frames with an empty pose set have no posed rows and primary -1.

A per-frame rule is rule(boxes (n,4) float32 xyxy px, scores (n,) float32, frame_size (W, H) or
None) -> index in 0..n-1 (n >= 1). Only rules that look at positions need frame_size.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .types import POSE_SET, POSED, ChunkDets

FrameSize = Optional[Tuple[int, int]]
PrimaryRule = Callable[..., int]
RIGHT_MIN_X = 0.6   # right_largest: a box whose centre x is at or right of this fraction of W


def largest_bbox(boxes: np.ndarray, scores: np.ndarray, frame_size: FrameSize = None) -> int:
    """Index of the largest box: area (x2 - x1) * (y2 - y1) in float32, ties -> lowest index
    (np.argmax). `scores` and `frame_size` are unused.
    """
    return int(np.argmax(_areas(boxes)))


def highest_score(boxes: np.ndarray, scores: np.ndarray, frame_size: FrameSize = None) -> int:
    """Index of the highest detector score, ties -> lowest index (np.argmax)."""
    return int(np.argmax(scores))


def right_largest(boxes: np.ndarray, scores: np.ndarray, frame_size: FrameSize = None) -> int:
    """Index of the largest box (as largest_bbox) among those whose centre x is >= RIGHT_MIN_X * W,
    or of the largest box when none is there. Auslan-Daily news shows its interpreter in a panel on
    the right, beside news footage or a studio newsreader who may be larger. Needs frame_size.
    """
    if frame_size is None:
        raise ValueError('right_largest needs the frame size')
    b = np.asarray(boxes, np.float32)
    area = _areas(b)
    right = (b[:, 0] + b[:, 2]) / 2 >= RIGHT_MIN_X * frame_size[0]
    return int(np.argmax(np.where(right, area, -np.inf))) if right.any() else int(np.argmax(area))


def _areas(boxes: np.ndarray) -> np.ndarray:
    b = np.asarray(boxes, np.float32)
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])


PRIMARY_RULES: Dict[str, PrimaryRule] = {'largest_bbox': largest_bbox, 'highest_score': highest_score,
                                         'right_largest': right_largest}
# Rules whose pick is always the largest or the highest-score box, which are posed anyway (K >= 2):
# extracting with either poses the same rows, so they need no entry in the extraction hash.
DEFAULT_POSING_RULES = frozenset({'largest_bbox', 'highest_score'})


def _rule(name: str) -> PrimaryRule:
    if name not in PRIMARY_RULES:
        raise KeyError(f'unknown per-frame primary rule {name!r}; known: {sorted(PRIMARY_RULES)}')
    return PRIMARY_RULES[name]


def select_frame(boxes: np.ndarray, scores: np.ndarray, pose_mask: np.ndarray, k: int,
                 rule: str, frame_size: FrameSize = None) -> Tuple[np.ndarray, int]:
    """Posed rows and primary position for one frame.

    Args:
        boxes: (n,4) float32 xyxy original pixels of the frame's candidates, det order.
        scores: (n,) float32 detector scores (non-increasing).
        pose_mask: (n,) bool, True for pose-set rows.
        k: Settings.max_posed (>= 1).
        rule: key of PRIMARY_RULES; the rule sees only the pose-set rows (boxes[S], scores[S]).
        frame_size: (W, H) of the frame, passed to the rule (the default rules ignore it).
    Returns:
        (posed, primary): posed is (p,) int64 ascending row indices into the frame's rows with
        p = min(k, pose_mask.sum()); primary is the position of the primary inside `posed`
        (0 <= primary < p), or (empty int64 array, -1) when the pose set is empty.
    Raises:
        KeyError for an unknown rule.
    """
    pick = _rule(rule)
    if k < 1:
        raise ValueError(f'k must be >= 1, got {k}')
    pose_rows = np.flatnonzero(pose_mask)
    if not len(pose_rows):
        return np.zeros(0, np.int64), -1
    b, s = boxes[pose_rows], scores[pose_rows]
    primary = int(pose_rows[pick(b, s, frame_size)])
    priority = [primary, int(pose_rows[largest_bbox(b, s)]), int(pose_rows[highest_score(b, s)])]
    posed: List[int] = []
    for row in priority + pose_rows.tolist():
        if row not in posed:
            posed.append(row)
            if len(posed) == k:
                break
    posed_rows = np.sort(np.array(posed, np.int64))
    return posed_rows, int(np.searchsorted(posed_rows, primary))


def select_chunk(dets: ChunkDets, k: int, rule: str, frame_size: FrameSize = None) -> Tuple[np.ndarray, np.ndarray]:
    """Apply `select_frame` to every frame of a chunk (`frame_size` = the video's (W, H)).

    Clears, then sets, the POSED bit in `dets.flags` IN PLACE (POSED is always a subset of POSE_SET).
    Returns:
        posed_rows: (M,) int64 chunk-local det row indices, frame-major and ascending (so the rows
            of frame t are contiguous). Maps 1:1 onto ChunkPoses.det_index (cast to int32); the
            ChunkPoses offsets are the cumulative per-frame posed counts.
        primary: (F,) int8 primary position within each frame's posed list, -1 where none
            (becomes ChunkPoses.primary).
    """
    _rule(rule)
    dets.flags &= np.uint8(0xFF ^ POSED)
    pose = (dets.flags & POSE_SET) != 0
    primary = np.full(dets.num_frames, -1, np.int8)
    parts = []
    for t in range(dets.num_frames):
        lo, hi = int(dets.offsets[t]), int(dets.offsets[t + 1])
        if pose[lo:hi].any():
            posed, primary[t] = select_frame(dets.boxes[lo:hi], dets.scores[lo:hi], pose[lo:hi], k, rule,
                                             frame_size)
            parts.append(posed + lo)
    posed_rows = np.concatenate(parts) if parts else np.zeros(0, np.int64)
    dets.flags[posed_rows] |= POSED
    return posed_rows, primary
