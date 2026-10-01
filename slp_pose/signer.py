"""Video-level primary-signer rules: the signer's spot on screen (spec D14).

Pure numpy on the CPU; must not import torch.

A video-level rule sees a whole persons record and picks, per frame, one POSED person or none
(-1). It runs in `record.derive` on the CPU after extraction; extraction itself poses with a
per-frame rule (select.PRIMARY_RULES) chosen so that the signer is among the posed people.

`signer_track` (parameters in SignerTrackParams; seconds use the video's exact frame rate):
1. Tracks: the posed boxes of consecutive frames are linked into tracks, greedily by IoU with each
   track's last box (highest IoU first, IoU >= link_iou). A track survives up to max_gap_s without
   a box.
2. Spots: a spot is a place on screen, such as the interpreter's panel. Tracks shorter than
   min_visit_s (people passing through a footage shot) belong to no spot; in a video shorter than
   min_visit_s / max_visit_share (an isolated-sign clip) the threshold is max_visit_share of its
   frames instead, so its signer still founds a spot. The others are visited
   longest first; a track whose median box overlaps a spot's box (IoU >= link_iou) joins the best
   such spot, otherwise its median box founds a new spot. A box counts for its track's spot only
   if it overlaps the spot's box (IoU >= link_iou); in a frame, a spot keeps the box overlapping
   it most. So an interpreter switch and every return after an absence stay at one spot, and a
   box that drifts away (two people merged into one detection) leaves it.
3. Segments: a spot's presence is cut where the spot stays empty for more than merge_gap_s (an
   interpreter leaving near the end of a broadcast ends a segment).
4. The main spot is where the interpreter sits: among the spots present in at least main_share of
   all frames, the one favoured by the position prior ('right': the larger median box centre x;
   None: no preference), then the one present in most frames. While one of its segments spans t
   (first to last box), that segment is the signer: its box at t, or -1 while the spot is empty
   (the interpreter is off screen; a programme person elsewhere is never picked instead).
5. Outside the main spot's segments (or in a video without a main spot), the signer is the other
   segment present in most frames of [t - window_s, t + window_s], among the segments whose span
   contains t, that are present in at least min_track_s in all (people in programme footage
   change with the cuts), and that are steady: present in at least min_density of their span
   inside that window (a centred full-screen interpreter is; people coming and going in footage
   are not). Ties go to the prior, then to the longer presence, then to the earlier segment.
   -1 when there is no such segment, or it has no box at t.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

PRIORS = (None, 'right')


@dataclass(frozen=True)
class SignerTrackParams:
    """Parameters of `signer_track` (module docstring); seconds are converted with the video's
    exact frame rate. The first five fields keep their order (positional construction)."""

    link_iou: float = 0.5      # same person, same spot: IoU >= this
    max_gap_s: float = 1.0     # a track survives this long without a box
    min_track_s: float = 20.0  # a segment outside the main spot needs this much presence
    window_s: float = 60.0     # half-width of the presence window
    prior: Optional[str] = None
    merge_gap_s: float = 60.0  # a spot empty for longer starts a new segment
    min_visit_s: float = 1.0   # shorter tracks belong to no spot (people passing through a shot)
    main_share: float = 0.5    # the main spot is present in at least this share of all frames
    min_density: float = 0.8   # steady: present in at least this share of the span in the window
    max_visit_share: float = 0.5   # min_visit_s is at most this share of the video's frames (short clips)

    def __post_init__(self) -> None:
        if self.prior not in PRIORS:
            raise ValueError(f'unknown prior {self.prior!r}; known: {PRIORS}')
        shares_ok = (0.0 < self.main_share <= 1.0 and 0.0 <= self.min_density <= 1.0
                     and 0.0 < self.max_visit_share <= 1.0)
        if (not 0.0 < self.link_iou <= 1.0 or not shares_ok
                or min(self.max_gap_s, self.min_track_s, self.window_s, self.merge_gap_s, self.min_visit_s) < 0):
            raise ValueError(f'bad signer_track parameters {self}')


VIDEO_RULES: Dict[str, SignerTrackParams] = {
    'signer_track': SignerTrackParams(),
    'signer_track_right': SignerTrackParams(prior='right'),
}


def video_rule(name: str) -> SignerTrackParams:
    """The parameters of a video-level rule; KeyError for an unknown name."""
    if name not in VIDEO_RULES:
        raise KeyError(f'unknown video-level rule {name!r}; known: {sorted(VIDEO_RULES)}')
    return VIDEO_RULES[name]


def rule_params(name: str) -> Dict[str, object]:
    """The parameters of a video-level rule as a plain dict (hashed into derivation_hash)."""
    return dataclasses.asdict(video_rule(name))


def apply_video_rule(name: str, boxes: np.ndarray, offsets: np.ndarray, fps: Tuple[int, int]) -> np.ndarray:
    """Primary position per frame under the video-level rule `name`.

    Args:
        boxes: (M,4) xyxy original pixels of the posed people, frame-major (record.kpt_det order).
        offsets: (T+1,) int64 CSR offsets of the posed rows per frame (record.kpt_offsets).
        fps: exact frame rate (num, den).
    Returns:
        (T,) int64 position of the primary inside each frame's posed list, -1 = none.
    """
    return signer_track(boxes, offsets, fps, video_rule(name))


def signer_track(boxes: np.ndarray, offsets: np.ndarray, fps: Tuple[int, int],
                 params: SignerTrackParams) -> np.ndarray:
    """The `signer_track` rule (module docstring); arguments and result as in apply_video_rule."""
    offsets = np.asarray(offsets, np.int64)
    num_frames = len(offsets) - 1
    out = np.full(num_frames, -1, np.int64)
    if not len(boxes):
        return out
    rate = fps[0] / fps[1]
    boxes = np.asarray(boxes, np.float64)
    frame = np.repeat(np.arange(num_frames), np.diff(offsets))
    tracks = link_tracks(boxes, offsets, params.link_iou, round(params.max_gap_s * rate))
    min_visit = min(round(params.min_visit_s * rate), max(1, math.floor(params.max_visit_share * num_frames)))
    segment, spot = spot_segments(boxes, frame, tracks, params.link_iou, round(params.merge_gap_s * rate), min_visit)
    if not len(spot):
        return out
    rows = np.flatnonzero(segment >= 0)
    rows = rows[np.lexsort((frame[rows], segment[rows]))]          # by segment, then frame
    cuts = np.searchsorted(segment[rows], np.arange(len(spot) + 1))
    frames = [frame[rows[cuts[j]:cuts[j + 1]]] for j in range(len(spot))]
    prior = _prior_keys(boxes[rows], np.repeat(spot, np.diff(cuts)), params.prior)
    main = _main_spot(spot, np.diff(cuts), prior, params.main_share * num_frames)
    chosen = np.full(num_frames, -1, np.int64)
    others = [j for j in range(len(spot)) if spot[j] != main and len(frames[j]) >= round(params.min_track_s * rate)]
    if others:
        best = _signer_per_frame([frames[j] for j in others], prior[spot[others]], round(params.window_s * rate),
                                 params.min_density, num_frames)
        chosen[best >= 0] = np.asarray(others)[best[best >= 0]]
    for j in np.flatnonzero(spot == main):
        chosen[frames[j][0]:frames[j][-1] + 1] = j
    hit = np.flatnonzero(segment == chosen[frame])
    hit = hit[segment[hit] >= 0]
    out[frame[hit]] = hit - offsets[frame[hit]]
    return out


def link_tracks(boxes: np.ndarray, offsets: np.ndarray, link_iou: float, max_gap: int) -> np.ndarray:
    """(M,) int64 track id of every row; ids are numbered in order of the tracks' first frame.

    Per frame, (track, row) pairs with IoU >= link_iou are matched greedily, highest IoU first
    (ties: lower track id, then lower row); unmatched rows start new tracks. A track takes part
    while at most `max_gap` frames have passed since its last box.
    """
    rows: List[List[float]] = np.asarray(boxes, np.float64).tolist()
    ids = np.empty(len(rows), np.int64)
    active: List[list] = []   # [track id, box, last frame]
    next_id = 0
    for t in range(len(offsets) - 1):
        lo, hi = int(offsets[t]), int(offsets[t + 1])
        if lo == hi:
            continue
        active = [a for a in active if t - a[2] - 1 <= max_gap]
        pairs = []
        for i, a in enumerate(active):
            for r in range(lo, hi):
                iou = _iou(a[1], rows[r])
                if iou >= link_iou:
                    pairs.append((-iou, i, r))
        taken_tracks, taken_rows = set(), set()
        for _, i, r in sorted(pairs):
            if i not in taken_tracks and r not in taken_rows:
                taken_tracks.add(i)
                taken_rows.add(r)
                ids[r] = active[i][0]
                active[i][1:] = [rows[r], t]
        for r in range(lo, hi):
            if r not in taken_rows:
                ids[r] = next_id
                active.append([next_id, rows[r], t])
                next_id += 1
    return ids


def spot_segments(boxes: np.ndarray, frame: np.ndarray, tracks: np.ndarray, link_iou: float,
                  merge_gap: int, min_visit: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Spots and their segments (steps 2-3 of the module docstring).

    Args:
        boxes: (M,4) float64 posed boxes; frame: (M,) frame of each row; tracks: (M,) link_tracks
            ids; merge_gap: frames a spot may stay empty inside one segment; min_visit: tracks with
            fewer boxes belong to no spot.
    Returns:
        segment: (M,) int64 segment of each row, -1 for a row outside every segment (a box of a
            short track, a box that does not overlap its track's spot, or a spot's second box in
            a frame). Segments hold at
            most one row per frame and are numbered in order of their first frame.
        spot: (S,) int64 spot of each segment; spots are numbered in founding order.
    """
    order = np.argsort(tracks, kind='stable')
    cuts = np.searchsorted(tracks[order], np.arange(tracks.max() + 2))
    median = np.array([np.median(boxes[order[cuts[j]:cuts[j + 1]]], axis=0) for j in range(len(cuts) - 1)])
    length = np.diff(cuts)
    spot_of_track = np.full(len(median), -1, np.int64)
    spot_box = np.zeros_like(median)   # rows past `spots` stay zero: spot_box[-1] is read for spot-less tracks
    spots = 0
    for j in np.argsort(-length, kind='stable'):   # longest track first
        if length[j] < min_visit:
            break
        fit = _ious(np.broadcast_to(median[j], (spots, 4)), spot_box[:spots])
        if spots and fit.max() >= link_iou:
            spot_of_track[j] = int(np.argmax(fit))
        else:
            spot_box[spots] = median[j]
            spot_of_track[j] = spots
            spots += 1
    spot = spot_of_track[tracks]
    fit = _ious(spot_box[spot], boxes)
    rows = np.flatnonzero((fit >= link_iou) & (spot >= 0))
    if not len(rows):
        return np.full(len(boxes), -1, np.int64), np.zeros(0, np.int64)
    rows = rows[np.lexsort((-fit[rows], frame[rows], spot[rows]))]   # by spot, frame, best fit first
    rows = rows[np.r_[True, (spot[rows][1:] != spot[rows][:-1]) | (frame[rows][1:] != frame[rows][:-1])]]
    starts = np.r_[True, (spot[rows][1:] != spot[rows][:-1]) | (np.diff(frame[rows]) - 1 > merge_gap)]
    rank = np.empty(int(starts.sum()), np.int64)
    rank[np.argsort(frame[rows][starts], kind='stable')] = np.arange(len(rank))
    segment = np.full(len(boxes), -1, np.int64)
    segment[rows] = rank[np.cumsum(starts) - 1]
    segment_spot = np.empty(len(rank), np.int64)
    segment_spot[rank] = spot[rows][starts]
    return segment, segment_spot


def _iou(a: List[float], b: List[float]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _ious(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of a[i] with b[i] for (n,4) boxes (0 for empty boxes)."""
    w = np.clip(np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    h = np.clip(np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = w * h
    union = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return np.divide(inter, union, out=np.zeros(len(a)), where=union > 0)


def _prior_keys(boxes: np.ndarray, spot: np.ndarray, prior: Optional[str]) -> np.ndarray:
    """(spots,) float64 prior key per spot from its segment rows (larger wins); zeros without a
    prior, 'right': the median box centre x."""
    keys = np.zeros(spot.max() + 1)
    if prior == 'right':
        centre = (boxes[:, 0] + boxes[:, 2]) / 2
        for k in np.unique(spot):
            keys[k] = np.median(centre[spot == k])
    return keys


def _main_spot(spot: np.ndarray, presence: np.ndarray, prior_key: np.ndarray, min_frames: float) -> int:
    """The main spot (step 4): among spots present in at least min_frames frames, the prior's
    favourite, then the most frames, then the lower id; -1 when no spot is present that often."""
    frames = np.bincount(spot, weights=presence)
    big = np.flatnonzero(frames >= min_frames)
    if not len(big):
        return -1
    return int(big[np.lexsort((-big, frames[big], prior_key[big]))[-1]])


def _signer_per_frame(frames: List[np.ndarray], prior_key: np.ndarray, window: int, min_density: float,
                      num_frames: int) -> np.ndarray:
    """(T,) int64 index of the signer segment (into `frames`) per frame, -1 where none (step 5).

    frames[j]: ascending frames where segment j has a box; segments ascend with their first frame.
    Score of a steady segment at t: its frames inside [t - window, t + window], then the prior key,
    then its presence, then the earlier segment; one int64 per (segment, frame) from dense ranks.
    """
    count = len(frames)
    prior_rank = np.unique(prior_key, return_inverse=True)[1].astype(np.int64)
    presence_rank = np.unique([len(f) for f in frames], return_inverse=True)[1].astype(np.int64)
    tie = (prior_rank * count + presence_rank) * count + (count - 1 - np.arange(count))
    best_score = np.full(num_frames, -1, np.int64)
    best = np.full(num_frames, -1, np.int64)
    for j, f in enumerate(frames):
        t = np.arange(f[0], f[-1] + 1)
        cover = np.searchsorted(f, t + window, 'right') - np.searchsorted(f, t - window, 'left')
        span = np.minimum(f[-1], t + window) - np.maximum(f[0], t - window) + 1
        score = np.where(cover >= min_density * span, cover * count ** 3 + tie[j], -1)
        better = score > best_score[t]
        best_score[t[better]] = score[better]
        best[t[better]] = j
    return best
