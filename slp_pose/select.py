"""Posed-slot selection and the framework's per-frame primary-signer rules (spec §4.2 select.py,
assumption 4, D4, D14, D22).

Pure numpy on the CPU; must not import torch.

Terms (all per frame, rows in det order = descending detector score, see types.ChunkDets):
- pose set S: rows with the POSE_SET bit.
- posed rows: at most K rows of S that get keypoints. Priority list
  [S[rule(S)], S[largest_bbox(S)], S[highest_score(S)]] + the remaining S rows in det order,
  de-duplicated keeping the first occurrence, truncated to K, then sorted ascending.
  So the primary is always posed, and both framework rules' picks are posed whenever K >= 3,
  which is what lets `record.derive` switch between per-frame rules, or apply a video-level
  rule (signer.py) to the posed people, on the CPU.
- primary position: index of the primary row inside the sorted posed list (int8), -1 = none.
Frames with an empty pose set have no posed rows and primary -1.

A per-frame rule is a FrameRule: a name and pick(boxes (n,4) float32 xyxy px, scores (n,) float32,
frame_size (W, H) or None) -> index in 0..n-1 (n >= 1); only rules that look at positions need
frame_size. The framework's rules (PRIMARY_RULES) are generic; a rule for one dataset only (e.g.
Auslan News' `right_largest`) is defined in that dataset's file and listed in its Dataset.rules
(D22). Functions here take a FrameRule or the name of a framework rule.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

from .types import POSE_SET, POSED, ChunkDets

FrameSize = Optional[Tuple[int, int]]
PrimaryRule = Callable[..., int]


@dataclass(frozen=True)
class FrameRule:
    """A named per-frame primary rule (module docstring), called like its `pick`.

    name: what records (persons/ primary_rule), hashes, logs and the CLI call it.
    pick: pick(boxes, scores, frame_size) -> index of the primary among the frame's pose-set rows;
        any callable (a function, functools.partial, an instance with __call__), hashable or not:
        the rule's Python hash() is its name's (the GPU workers cache a processor per rule).
    params: None, or a JSON-able dict of the rule's parameters. Hashed with the name: into the
        extraction hash of a dataset whose workers pose with the rule (posing_rule_params; a change
        re-extracts its videos) and into the derivation hash of one whose primary rule it is. None:
        the name alone stands for the rule (give it a new name when its definition changes).
    """

    name: str
    pick: PrimaryRule = field(hash=False)
    params: Optional[Mapping[str, object]] = field(default=None, hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f'a rule name must be a non-empty str, got {self.name!r}')
        if not callable(self.pick):
            raise ValueError(f'rule {self.name!r}: pick must be callable, got {self.pick!r}')
        if self.params is not None:
            object.__setattr__(self, 'params', _json_params(self.name, self.params))

    def __call__(self, boxes: np.ndarray, scores: np.ndarray, frame_size: FrameSize = None) -> int:
        return self.pick(boxes, scores, frame_size)


def _json_params(name: str, params: Mapping[str, object]) -> Dict[str, object]:
    """A copy of a rule's params, checked to be a JSON-able dict with str keys (they are hashed)."""
    if not isinstance(params, Mapping) or not all(isinstance(key, str) for key in params):
        raise ValueError(f'rule {name!r}: params must be a dict with str keys, got {params!r}')
    try:
        json.dumps(params, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'rule {name!r}: params must be JSON-able (they are hashed): {exc}') from None
    return dict(params)


def largest_bbox(boxes: np.ndarray, scores: np.ndarray, frame_size: FrameSize = None) -> int:
    """Index of the largest box: area (x2 - x1) * (y2 - y1) in float32, ties -> lowest index
    (np.argmax). `scores` and `frame_size` are unused.
    """
    return int(np.argmax(_areas(boxes)))


def highest_score(boxes: np.ndarray, scores: np.ndarray, frame_size: FrameSize = None) -> int:
    """Index of the highest detector score, ties -> lowest index (np.argmax)."""
    return int(np.argmax(scores))


def _areas(boxes: np.ndarray) -> np.ndarray:
    b = np.asarray(boxes, np.float32)
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])


# The framework's per-frame rules, generic for any dataset.
PRIMARY_RULES: Dict[str, FrameRule] = {name: FrameRule(name, pick) for name, pick in
                                       (('largest_bbox', largest_bbox), ('highest_score', highest_score))}
# Rules whose pick is always the largest or the highest-score box, which are posed anyway (K >= 2):
# extracting with either poses the same rows, so they need no entry in the extraction hash.
DEFAULT_POSING_RULES = frozenset({'largest_bbox', 'highest_score'})


def frame_rule(rule: Union[str, FrameRule]) -> FrameRule:
    """`rule` itself, or the framework's per-frame rule of that name (KeyError for another name)."""
    if isinstance(rule, FrameRule):
        return rule
    if rule not in PRIMARY_RULES:
        raise KeyError(f"unknown per-frame primary rule {rule!r}; the framework's: {sorted(PRIMARY_RULES)} "
                       f"(a dataset's own rules come from its Dataset.rule)")
    return PRIMARY_RULES[rule]


def select_frame(boxes: np.ndarray, scores: np.ndarray, pose_mask: np.ndarray, k: int,
                 rule: Union[str, FrameRule], frame_size: FrameSize = None) -> Tuple[np.ndarray, int]:
    """Posed rows and primary position for one frame.

    Args:
        boxes: (n,4) float32 xyxy original pixels of the frame's candidates, det order.
        scores: (n,) float32 detector scores (non-increasing).
        pose_mask: (n,) bool, True for pose-set rows.
        k: Settings.max_posed (>= 1).
        rule: a FrameRule or a framework rule name (PRIMARY_RULES); the rule sees only the
            pose-set rows (boxes[S], scores[S]).
        frame_size: (W, H) of the frame, passed to the rule (the framework's rules ignore it).
    Returns:
        (posed, primary): posed is (p,) int64 ascending row indices into the frame's rows with
        p = min(k, pose_mask.sum()); primary is the position of the primary inside `posed`
        (0 <= primary < p), or (empty int64 array, -1) when the pose set is empty.
    Raises:
        KeyError for an unknown rule name.
    """
    pick = frame_rule(rule)
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


def select_chunk(dets: ChunkDets, k: int, rule: Union[str, FrameRule],
                 frame_size: FrameSize = None) -> Tuple[np.ndarray, np.ndarray]:
    """Apply `select_frame` to every frame of a chunk (`frame_size` = the video's (W, H)).

    Clears, then sets, the POSED bit in `dets.flags` IN PLACE (POSED is always a subset of POSE_SET).
    Returns:
        posed_rows: (M,) int64 chunk-local det row indices, frame-major and ascending (so the rows
            of frame t are contiguous). Maps 1:1 onto ChunkPoses.det_index (cast to int32); the
            ChunkPoses offsets are the cumulative per-frame posed counts.
        primary: (F,) int8 primary position within each frame's posed list, -1 where none
            (becomes ChunkPoses.primary).
    """
    rule = frame_rule(rule)
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
