"""Video-level primary-signer rules: the framework's generic signer spot (spec D14, D24), its building
blocks, and the signing-motion measure (D21) that dataset rules can build on (D22).

Pure numpy on the CPU; must not import torch.

A video-level rule (VideoRule) sees the posed people of a whole video (PosedPeople) and picks, per
frame, one POSED person or none (-1), plus the interpreter's region (SignerChoice). It runs in
`record.derive` on the CPU after extraction; extraction itself poses with a per-frame rule
(select.py) chosen so that the signer is among the posed people. The framework's own rule is
`signer_track` (VIDEO_RULES; BOBSL's primary rule); a rule for one dataset only is defined in that
dataset's file from the building blocks below and listed in its Dataset.rules (D22): Auslan News'
`signer_track_right` (slp_pose/datasets/auslan_news.py) adds a right-hand position prior, a
signing-motion test in step 5 and a step 6 that fills the panel's gaps with an on-site interpreter.

`signer_track` (parameters in SignerTrackParams; seconds use the video's exact frame rate):
1. Tracks: the posed boxes of consecutive frames are linked into tracks, greedily by IoU with each
   track's last box (highest IoU first, IoU >= link_iou). A track survives up to max_gap_s without
   a box.
2. Spots: a spot is a place on screen, such as the interpreter's panel. Tracks shorter than
   min_visit_s (people passing through a footage shot) belong to no spot; in a video shorter than
   min_visit_s / max_visit_share (an isolated-sign clip) the threshold is max_visit_share of its
   frames instead, so its signer still founds a spot. The others are visited longest first; a
   track whose median box overlaps a spot's box (IoU >= link_iou) joins the best such spot if it
   fits that spot; any other track founds a new spot: the spot's box and scale are its median box
   and median shoulder width, and it is the spot's first anchor. A track fits a spot (D24) when its
   median box size (the square root of the area) and median shoulder width are within a factor
   fit_scale of the spot's, and it is at least anchor_s long (then it is an anchor) or its median
   box overlaps an anchor's by IoU >= fit_iou: a programme person standing at the spot for a shot
   rarely has the interpreter's size and exact place. A box counts for its track's spot only if it
   overlaps the spot's box (IoU >= link_iou; for an anchor whose median box overlaps the spot's box
   by less than fit_iou, the interpreter having moved, its own median box) and, where its track (an
   anchor: a second interpreter may be broader) or else its spot has a scale, the median shoulder
   width of its track's boxes within +- scale_window_s / 2 is measured, at most max_scale times
   that scale and at least min_scale times it (a track that drifts over a cut onto a close-up, or
   a close-up without shoulders or with the junk width of barely seen ones, leaves the spot); in a
   frame, a spot keeps the box overlapping it most. So an interpreter switch and every return
   after an absence stay at one spot, and a box that drifts away (two people merged into one
   detection) leaves it. Shoulder widths (shoulder_widths: COCO 5, 6 both seen) come from the
   keypoints; without them (width=None) only box sizes are compared.
3. Segments: a spot's presence is cut where the spot stays empty for more than merge_gap_s (an
   interpreter leaving near the end of a broadcast ends a segment).
4. The main spot is where the interpreter sits: among the spots present in at least main_share of
   all frames, the one favoured by the position prior (a key per spot, larger wins: spot_keys with
   a prior_key function that SignerTrackParams.prior names; None: no preference), then the one
   present in most frames. While one of its segments spans t (first to last box), that segment is
   the signer: its box at t, or -1 while the spot is empty (the interpreter is off screen).
5. Outside the main spot's segments (or in a video without a main spot), the signer is the other
   segment present in most frames of [t - window_s, t + window_s], among the segments whose span
   contains t, that are present in at least min_track_s in all (people in programme footage
   change with the cuts), that are steady: present in at least min_density of their span inside
   that window (a centred full-screen interpreter is; people coming and going in footage are
   not), and, when the caller gives a per-box activity test (`active`, `min_active` > 0: a dataset
   rule's signing motion), that are active in at least min_active of their boxes inside that
   window. Ties go to the prior, then to the longer presence, then to the earlier segment. -1 when
   there is no such segment, or it has no box at t.
Steps 1-3 are spot_layout (a SpotLayout), steps 4-5 signer_segments, and SpotLayout.choice turns
the chosen segment per frame into the SignerChoice; a dataset rule may change the chosen segments
in between (Auslan News' step 6).

Signing motion (D21), a measure for dataset rules (the framework's `signer_track` has no motion
test: BOBSL's programme people move their wrists as much as its interpreter). wrist_points gives
every posed row its left and right wrist (COCO 9, 10, seen when scoring > point_conf) minus the
shoulder midpoint and its shoulder width (COCO 5, 6, both seen), in original pixels.
signing_motion, per segment: the wrists are divided by the scale, the median shoulder width of the
segment's boxes within +- motion_window_s / 2, at least width_floor of the box height (a jittering
width of a narrow or turned pair does not move the scale); a wrist farther than max_reach is a
glitch and unseen; each wrist coordinate is median-filtered over the segment's boxes within +-
smooth_s / 2 (a one-frame glitch moves no median); a box's motion is the larger, over the two
wrists, spread sqrt(var x + var y) of the filtered wrist over its segment's boxes within +-
motion_window_s / 2 (a wrist seen in fewer than min_seen of the window's frames is not measured,
NaN). A box moves at signing level when its motion >= min_motion, and signs (signing_boxes) when at
least min_sustain of the measured boxes of its segment within +- motion_window_s / 2 do too: the
window spreads a short movement over half a window on each side (an itch scratched for 1 s would
count for 3 s), and this takes most of that back (about 2 s; 1 s with min_sustain 1), while short
pauses of a signer stay bridged. A signer's wrists keep moving, with pauses; a speaker, officer or
newsreader beside them moves theirs only now and then (a head shake moves no wrist). Wrists, not
hand-keypoint centroids: the centroid of a hand's confident keypoints jumps with the subset that is
confident (clasped or occluded hands of a still speaker).

The interpreter's region (SignerChoice.region) of frame t is the box of the spot whose segment the
rule used at t (the main spot inside its segments' spans, step 4, unless a dataset step took
another segment there; else the accepted segment of step 5, such as a centred full-screen
interpreter), whether or not the signer has a box at t; a frame outside every accepted segment has
no region. A spot's box is the median box of the track that founded it (step 2). meta.masked_runs
counts the people inside it (D18, D19).
"""
from __future__ import annotations

import dataclasses
import functools
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, NamedTuple, Optional, Tuple

import numpy as np

from .select import _json_params

# wrist_points reads COCO left / right shoulder (5, 6) and wrist (9, 10).
WRIST_COLUMNS = np.array([5, 6, 9, 10])
_MIN_SHOULDER_WIDTH = 1e-6   # px; a narrower shoulder pair is no shoulder pair
_BLOCK_ROWS = 16384          # posed rows read from kpts at a time
_MEDIAN_CELLS = 1 << 20      # window cells sorted at a time by _window_median
PriorKey = Callable[[np.ndarray], float]   # a spot's boxes (n,4) float64 -> its prior key (larger wins)


@dataclass(frozen=True)
class SignerTrackParams:
    """Parameters of `signer_track` (module docstring, steps 1-5); seconds are converted with the
    video's exact frame rate. The first five fields keep their order (positional construction).
    The defaults are D24's (BOBSL); D14's box-only spots are fit_iou=0, fit_scale=0, max_scale=0,
    min_scale=0.

    prior names the position prior that the caller passes as a prior_key function (None: no
    preference); only the name is hashed (VideoRule.params), so a changed key function needs a new
    name."""

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
    anchor_s: float = 20.0     # a track this long that fits a spot is one of its anchors (D24)
    fit_iou: float = 0.8       # a shorter track joins only with its median box this close to an anchor's (0: none)
    fit_scale: float = 1.25    # joining: box size and shoulder width within this factor of the spot's (0: none)
    max_scale: float = 1.3     # a box counts while its shoulder width is at most this many of its anchor's (0: none)
    scale_window_s: float = 1.0    # ... the median width of its track's boxes within +- half this
    min_scale: float = 0.25    # ... and at least this many (0: none; a close-up's barely seen shoulders: junk widths)
    point_conf: float = 0.3    # a shoulder is seen when its score > this

    def __post_init__(self) -> None:
        if self.prior is not None and (not isinstance(self.prior, str) or not self.prior):
            raise ValueError(f'prior must be None or the name of a prior key function, got {self.prior!r}')
        shares_ok = (0.0 < self.main_share <= 1.0 and 0.0 <= self.min_density <= 1.0
                     and 0.0 < self.max_visit_share <= 1.0 and 0.0 <= self.fit_iou <= 1.0
                     and 0.0 <= self.point_conf < 1.0)
        scales_ok = ((self.fit_scale == 0 or self.fit_scale >= 1) and (self.max_scale == 0 or self.max_scale >= 1)
                     and 0.0 <= self.min_scale < 1.0)
        if (not 0.0 < self.link_iou <= 1.0 or not shares_ok or not scales_ok
                or min(self.max_gap_s, self.min_track_s, self.window_s, self.merge_gap_s, self.min_visit_s,
                       self.anchor_s, self.scale_window_s) < 0):
            raise ValueError(f'bad signer_track parameters {self}')

    @property
    def uses_scale(self) -> bool:
        """Whether the rule compares shoulder widths (fit_scale, max_scale, min_scale), so needs the keypoints."""
        return self.fit_scale > 0 or self.max_scale > 0 or self.min_scale > 0


@dataclass(frozen=True)
class MotionParams:
    """Parameters of the signing-motion measure (module docstring, 'Signing motion'; D21). The
    defaults are a starting point, calibrated on one news dataset (D21): a dataset's rule spells
    out its own values, as datasets/auslan_news.py does."""

    motion_window_s: float = 2.0   # a box's motion: its segment's wrists within +- half this
    min_motion: float = 0.12       # signing level: motion >= this many shoulder widths
    point_conf: float = 0.3        # a keypoint is seen when its score > this
    max_reach: float = 3.0         # a wrist farther from the shoulder midpoint (shoulder widths) is a glitch
    min_seen: float = 0.25         # a wrist is measured in a window when seen in this share of its frames
    smooth_s: float = 0.1          # wrist coordinates are median-filtered over +- half this
    width_floor: float = 0.1       # the shoulder-width scale is at least this share of the box height
    min_sustain: float = 0.75      # a box signs when this share of its segment's boxes around it move too

    def __post_init__(self) -> None:
        if not (self.motion_window_s > 0 and self.min_motion >= 0 and 0.0 <= self.point_conf < 1.0
                and self.max_reach > 0 and 0.0 < self.min_seen <= 1.0 and self.smooth_s >= 0
                and self.width_floor >= 0 and 0.0 <= self.min_sustain <= 1.0):
            raise ValueError(f'bad signing-motion parameters {self}')


class Wrists(NamedTuple):
    """The posed rows' wrists and shoulder widths (wrist_points), original pixels.

    points: (M,2,2) float32 left, right wrist (x, y) minus the shoulder midpoint, NaN = unseen (or
        no shoulder pair); width: (M,) float32 shoulder width, NaN = no shoulder pair.
    """

    points: np.ndarray
    width: np.ndarray


@dataclass(frozen=True)
class SignerChoice:
    """What a video-level rule chose for each frame of a video.

    primary: (T,) int64 position of the signer inside the frame's posed list, -1 = none.
    region: (T,4) float64 xyxy original pixels, the interpreter's region of each frame (module
        docstring): the box of the spot whose segment the rule used, NaN in a frame outside every
        accepted segment.
    """

    primary: np.ndarray
    region: np.ndarray


class PosedPeople(NamedTuple):
    """What a video-level rule sees of one video: its posed people, frame-major (the order of a
    persons record's kpt_det / kpts rows).

    boxes: (M,4) float32 xyxy original pixels; scores: (M,) float32 detector scores; offsets: (T+1,)
    int64 CSR offsets of each frame's posed rows; kpts: (M,133,3) float32 keypoints (x / W, y / H,
    score), often a memmap of the persons file: read only what the rule needs (wrist_points reads 4
    columns in blocks); fps: exact frame rate (num, den); frame_size: (W, H).
    """

    boxes: np.ndarray
    scores: np.ndarray
    offsets: np.ndarray
    kpts: np.ndarray
    fps: Tuple[int, int]
    frame_size: Tuple[int, int]

    @property
    def num_frames(self) -> int:
        return len(self.offsets) - 1


@dataclass(frozen=True)
class VideoRule:
    """A named video-level primary rule (module docstring; a dataset's own in Dataset.rules, D22).

    name: what records (persons/ primary_rule), hashes, logs and the CLI call it.
    choose: choose(PosedPeople) -> SignerChoice: per frame the signer's position in the posed list
        (-1 = none) and the interpreter's region (NaN rows = none; D18, meta.masked_runs counts the
        people inside it). Any callable, hashable or not (the rule's Python hash() is its name's).
    params: a JSON-able dict of everything that changes the choice (numbers, and the names of
        functions such as a position prior); hashed into the derivation hash (primary_rule_params),
        so the committed videos of a dataset re-derive on the CPU when it changes.
    """

    name: str
    choose: Callable[[PosedPeople], SignerChoice] = field(hash=False)
    params: Mapping[str, object] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f'a rule name must be a non-empty str, got {self.name!r}')
        if not callable(self.choose):
            raise ValueError(f'rule {self.name!r}: choose must be callable, got {self.choose!r}')
        object.__setattr__(self, 'params', _json_params(self.name, self.params))


@dataclass(frozen=True)
class SpotLayout:
    """Steps 1-3 of the module docstring for one video (spot_layout): what spot-based rules build on.

    offsets: (T+1,) int64 posed-row offsets per frame; rate: frames per second;
    boxes: (M,4) float64 posed boxes; frame: (M,) int64 frame of each row;
    tracks: (M,) int64 link_tracks id of each row;
    track_spot: (M,) int64 spot of each row's track (-1 for a short track), whether or not the row
        itself overlaps that spot;
    segment: (M,) int64 segment of each row (-1: none; spot_segments); spot: (S,) int64 spot of each
        segment; spot_box: (spots,4) float64 box of each spot (the median box of its founding track);
    rows: per segment, its rows in frame order; frames: per segment, the frames of those rows.
    """

    offsets: np.ndarray
    rate: float
    boxes: np.ndarray
    frame: np.ndarray
    tracks: np.ndarray
    track_spot: np.ndarray
    segment: np.ndarray
    spot: np.ndarray
    spot_box: np.ndarray
    rows: Tuple[np.ndarray, ...]
    frames: Tuple[np.ndarray, ...]

    @property
    def num_frames(self) -> int:
        return len(self.offsets) - 1

    @property
    def presence(self) -> np.ndarray:
        """(S,) int64 frames in which each segment has a box."""
        return np.array([len(f) for f in self.frames], np.int64)

    def choice(self, chosen: np.ndarray) -> SignerChoice:
        """The SignerChoice of `chosen`, the (T,) segment used per frame (-1: none): the segment's
        box where it has one (else -1) and its spot's box as the region (NaN where none)."""
        num_frames = self.num_frames
        out = np.full(num_frames, -1, np.int64)
        region = np.full((num_frames, 4), np.nan)
        if not len(self.spot):
            return SignerChoice(out, region)
        segment, frame = self.segment, self.frame
        hit = np.flatnonzero(segment == chosen[frame])
        hit = hit[segment[hit] >= 0]
        out[frame[hit]] = hit - self.offsets[frame[hit]]
        used = chosen >= 0
        region[used] = self.spot_box[self.spot[chosen[used]]]
        return SignerChoice(out, region)


def spot_layout(boxes: np.ndarray, offsets: np.ndarray, fps: Tuple[int, int],
                params: SignerTrackParams, width: Optional[np.ndarray] = None) -> SpotLayout:
    """Tracks, spots and segments (steps 1-3) of a video's posed boxes (M,4) xyxy original pixels,
    frame-major, with (T+1,) CSR `offsets` and exact frame rate `fps` (num, den); width: (M,) the
    boxes' shoulder widths (shoulder_widths, NaN = no pair) for the scale tests of step 2, None:
    no scale test (box sizes are still compared)."""
    offsets = np.asarray(offsets, np.int64)
    num_frames = len(offsets) - 1
    rate = fps[0] / fps[1]
    frame = np.repeat(np.arange(num_frames), np.diff(offsets))
    if width is not None and np.shape(width) != (len(boxes),):
        raise ValueError(f'width has shape {np.shape(width)}; expected ({len(boxes)},)')
    if not len(boxes):
        none = np.zeros(0, np.int64)
        return SpotLayout(offsets, rate, np.zeros((0, 4)), frame, none, none, none, none, np.zeros((0, 4)), (), ())
    boxes = np.asarray(boxes, np.float64)
    tracks = link_tracks(boxes, offsets, params.link_iou, round(params.max_gap_s * rate))
    min_visit = min(round(params.min_visit_s * rate), max(1, math.floor(params.max_visit_share * num_frames)))
    segment, spot, spot_box, track_spot = _spot_segments(
        boxes, frame, tracks, params.link_iou, round(params.merge_gap_s * rate), min_visit, width=width,
        anchor=round(params.anchor_s * rate), fit_iou=params.fit_iou, fit_scale=params.fit_scale,
        max_scale=params.max_scale, min_scale=params.min_scale, half=round(params.scale_window_s * rate / 2))
    rows = np.flatnonzero(segment >= 0)
    rows = rows[np.lexsort((frame[rows], segment[rows]))]          # by segment, then frame
    cuts = np.searchsorted(segment[rows], np.arange(len(spot) + 1))
    by_segment = tuple(rows[cuts[j]:cuts[j + 1]] for j in range(len(spot)))
    return SpotLayout(offsets, rate, boxes, frame, tracks, track_spot, segment, spot, spot_box, by_segment,
                      tuple(frame[r] for r in by_segment))


def spot_keys(layout: SpotLayout, prior_key: Optional[PriorKey] = None) -> np.ndarray:
    """(spots with a segment,) float64 position-prior key per spot id (larger wins): zeros without a
    prior, else prior_key(the boxes of every segment row of the spot)."""
    keys = np.zeros(layout.spot.max() + 1) if len(layout.spot) else np.zeros(0)
    if prior_key is not None:
        for k in np.unique(layout.spot):
            rows = np.concatenate([layout.rows[j] for j in np.flatnonzero(layout.spot == k)])
            keys[k] = prior_key(layout.boxes[rows])
    return keys


def signer_segments(layout: SpotLayout, params: SignerTrackParams, keys: np.ndarray,
                    active: Optional[np.ndarray] = None, min_active: float = 0.0) -> Tuple[np.ndarray, int]:
    """Steps 4-5: the (T,) int64 segment used per frame (-1: none) and the main spot (-1: none).

    keys: spot_keys; active: (M,) bool per posed row (e.g. signing_boxes; None = every row), which
    step 5 asks of min_active of a segment's boxes in the window (0 = no activity test).
    """
    num_frames, spot, frames = layout.num_frames, layout.spot, layout.frames
    chosen = np.full(num_frames, -1, np.int64)
    if not len(spot):
        return chosen, -1
    if not 0.0 <= min_active <= 1.0:
        raise ValueError(f'min_active must lie in [0, 1], got {min_active}')
    rate = layout.rate
    active = np.ones(len(layout.boxes), bool) if active is None else np.asarray(active, bool)
    main = _main_spot(spot, layout.presence, keys, params.main_share * num_frames)
    others = [j for j in range(len(spot)) if spot[j] != main and len(frames[j]) >= round(params.min_track_s * rate)]
    if others:
        best = _signer_per_frame([frames[j] for j in others], [active[layout.rows[j]] for j in others],
                                 keys[spot[others]], round(params.window_s * rate), params.min_density, min_active,
                                 num_frames)
        chosen[best >= 0] = np.asarray(others)[best[best >= 0]]
    for j in np.flatnonzero(spot == main):
        chosen[frames[j][0]:frames[j][-1] + 1] = j
    return chosen, main


def signer_track_choice(boxes: np.ndarray, offsets: np.ndarray, fps: Tuple[int, int],
                        params: SignerTrackParams = SignerTrackParams(),
                        prior_key: Optional[PriorKey] = None, width: Optional[np.ndarray] = None) -> SignerChoice:
    """The `signer_track` rule (steps 1-5, no activity test) with the region it used per frame.

    boxes: (M,4) xyxy original pixels of the posed people, frame-major (record.kpt_det order);
    offsets: (T+1,) int64 CSR offsets of the posed rows per frame (record.kpt_offsets); fps: exact
    frame rate (num, den); prior_key: the position prior that params.prior names (both or neither);
    width: (M,) the posed rows' shoulder_widths (None: no scale test).
    """
    if (params.prior is None) != (prior_key is None):
        raise ValueError(f'params.prior {params.prior!r} names the prior_key function: give both or neither')
    layout = spot_layout(boxes, offsets, fps, params, width)
    chosen, _ = signer_segments(layout, params, spot_keys(layout, prior_key))
    return layout.choice(chosen)


def signer_track(boxes: np.ndarray, offsets: np.ndarray, fps: Tuple[int, int],
                 params: SignerTrackParams = SignerTrackParams(), prior_key: Optional[PriorKey] = None,
                 width: Optional[np.ndarray] = None) -> np.ndarray:
    """(T,) int64 primary position per frame under `signer_track` (signer_track_choice.primary)."""
    return signer_track_choice(boxes, offsets, fps, params, prior_key, width).primary


def track_rule(name: str, params: SignerTrackParams = SignerTrackParams(),
               prior_key: Optional[PriorKey] = None) -> VideoRule:
    """`signer_track` with `params` (and the prior_key that params.prior names) as a VideoRule named
    `name`; its hashed params are dataclasses.asdict(params). Of the keypoints it reads only the
    shoulders (shoulder_widths), and only when params.uses_scale."""
    if (params.prior is None) != (prior_key is None):
        raise ValueError(f'params.prior {params.prior!r} names the prior_key function: give both or neither')
    return VideoRule(name, functools.partial(_track_people, params=params, prior_key=prior_key),
                     dataclasses.asdict(params))


def _track_people(people: PosedPeople, params: SignerTrackParams, prior_key: Optional[PriorKey]) -> SignerChoice:
    width = shoulder_widths(people.kpts, people.frame_size, params.point_conf) if params.uses_scale else None
    return signer_track_choice(people.boxes, people.offsets, people.fps, params, prior_key, width)


# The framework's video-level rule: the spot rule without a position prior (BOBSL's primary rule),
# D14's with D24's spot identity (anchors, scale tests: SignerTrackParams' defaults).
VIDEO_RULES: Dict[str, VideoRule] = {'signer_track': track_rule('signer_track')}


def shoulder_widths(kpts: np.ndarray, frame_size: Tuple[int, int], point_conf: float) -> np.ndarray:
    """(M,) float32 shoulder width of every posed row in original pixels: the distance between COCO
    5 and 6 when both score > point_conf, else NaN. kpts: (M,133,3) (x / W, y / H, score; may be a
    memmap, read in blocks of those two columns); frame_size: (W, H)."""
    out = np.full(len(kpts), np.nan, np.float32)
    scale = np.array(frame_size, np.float64)
    for s in range(0, len(kpts), _BLOCK_ROWS):
        block = np.asarray(kpts[s:s + _BLOCK_ROWS, WRIST_COLUMNS[:2]])   # (n, 2, 3): left, right shoulder
        xy = block[..., :2].astype(np.float64) * scale
        pair = np.hypot(xy[:, 0, 0] - xy[:, 1, 0], xy[:, 0, 1] - xy[:, 1, 1])
        seen = (block[..., 2] > np.float32(point_conf)).all(axis=1) & (pair > _MIN_SHOULDER_WIDTH)
        out[s:s + len(block)] = np.where(seen, pair, np.nan)
    return out


def wrist_points(kpts: np.ndarray, frame_size: Tuple[int, int], params: MotionParams) -> Wrists:
    """The wrists and shoulder widths of the posed rows (module docstring, 'Signing motion'), in
    original pixels (x * W, y * H).

    kpts: (M,133,3) the record's posed keypoints (x / W, y / H, score; may be a memmap, read in
    blocks of the columns WRIST_COLUMNS); frame_size: (W, H).
    """
    m = len(kpts)
    points = np.full((m, 2, 2), np.nan, np.float32)
    width = np.full(m, np.nan, np.float32)
    scale = np.array(frame_size, np.float64)
    conf = np.float32(params.point_conf)
    for s in range(0, m, _BLOCK_ROWS):
        block = np.asarray(kpts[s:s + _BLOCK_ROWS, WRIST_COLUMNS])   # (n, 4, 3): shoulders, wrists
        xy = block[..., :2].astype(np.float64) * scale
        seen = block[..., 2] > conf
        left, right = xy[:, 0], xy[:, 1]
        pair = np.hypot(left[:, 0] - right[:, 0], left[:, 1] - right[:, 1])
        pair = np.where(seen[:, 0] & seen[:, 1] & (pair > _MIN_SHOULDER_WIDTH), pair, np.nan)
        width[s:s + len(block)] = pair
        rel = xy[:, 2:] - ((left + right) / 2)[:, None, :]
        points[s:s + len(block)] = np.where((seen[:, 2:] & np.isfinite(pair)[:, None])[..., None], rel, np.nan)
    return Wrists(points, width)


def signing_motion(segment: np.ndarray, frame: np.ndarray, boxes: np.ndarray, wrists: Wrists, rate: float,
                   params: MotionParams) -> np.ndarray:
    """(M,) float64 motion of every row of a segment in shoulder widths, NaN for rows outside every
    segment and rows without a measured wrist (module docstring, 'Signing motion').

    segment, frame: (M,) as in spot_segments; boxes: (M,4) the posed boxes; wrists: wrist_points
    of the same rows; rate: frames per second. Windows hold a row's segment's rows within +- h
    frames: h = round(motion_window_s * rate / 2) for the scale and the spread, round(smooth_s *
    rate / 2) for the median filter. A wrist is measured when seen (after the filter) in at least
    max(2, ceil(min_seen * (2 * h + 1))) of the spread's window; its spread is sqrt(var x + var y)
    (population variances, float64 running sums).
    """
    rows, _, _, motion = _sorted_motion(segment, frame, boxes, wrists, rate, params)
    out = np.full(len(segment), np.nan)
    out[rows] = motion
    return out


def signing_boxes(segment: np.ndarray, frame: np.ndarray, boxes: np.ndarray, wrists: Wrists, rate: float,
                  params: MotionParams) -> np.ndarray:
    """(M,) bool: the row signs (module docstring, 'Signing motion'): its motion (signing_motion)
    is >= min_motion, and so is that of at least min_sustain of the measured rows of its segment
    within +- round(motion_window_s * rate / 2) frames. False outside every segment."""
    rows, key, half, motion = _sorted_motion(segment, frame, boxes, wrists, rate, params)
    out = np.zeros(len(segment), bool)
    lo = np.searchsorted(key, key - half, 'left')
    hi = np.searchsorted(key, key + half, 'right')
    active = motion >= params.min_motion   # NaN (not measured) is not signing
    measured = np.concatenate([[0], np.cumsum(np.isfinite(motion))])
    moving = np.concatenate([[0], np.cumsum(active)])
    out[rows] = active & (moving[hi] - moving[lo] >= params.min_sustain * (measured[hi] - measured[lo]))
    return out


def _sorted_motion(segment: np.ndarray, frame: np.ndarray, boxes: np.ndarray, wrists: Wrists, rate: float,
                   params: MotionParams) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """The rows of every segment by segment, then frame; their keys (ascending, distinct: two rows
    lie within d frames of one segment iff their keys differ by at most d <= the windows' half
    widths); the spread's half width in frames; signing_motion of those rows."""
    points, width = np.asarray(wrists.points), np.asarray(wrists.width)
    if points.shape != (len(segment), 2, 2) or width.shape != (len(segment),):
        raise ValueError(f'wrists have shapes {points.shape}, {width.shape}; expected ({len(segment)}, 2, 2), '
                         f'({len(segment)},)')
    half = round(params.motion_window_s * rate / 2)
    smooth = round(params.smooth_s * rate / 2)
    rows = np.flatnonzero(segment >= 0)
    if not len(rows):
        return rows, np.zeros(0, np.int64), half, np.zeros(0)
    rows = rows[np.lexsort((frame[rows], segment[rows]))]
    stride = int(frame[rows].max()) + 2 * max(half, smooth) + 2      # windows never reach another segment's keys
    key = segment[rows].astype(np.int64) * stride + frame[rows]
    if (np.diff(key) == 0).any():
        raise ValueError('a segment holds two rows of one frame')
    box = np.asarray(boxes, np.float64)[rows]
    scale = np.maximum(_window_median(key, width[rows].astype(np.float64), half),
                       params.width_floor * (box[:, 3] - box[:, 1]))     # NaN without a shoulder pair in the window
    xy = points[rows].astype(np.float64) / scale[:, None, None]
    xy[~(np.hypot(xy[..., 0], xy[..., 1]) <= params.max_reach)] = np.nan
    if smooth > 0:
        for point in range(2):
            for axis in range(2):
                xy[:, point, axis] = _window_median(key, xy[:, point, axis], smooth)
    lo = np.searchsorted(key, key - half, 'left')
    hi = np.searchsorted(key, key + half, 'right')
    need = max(2, math.ceil(params.min_seen * (2 * half + 1)))
    best = np.full(len(rows), np.nan)
    for point in range(2):
        p = xy[:, point]
        seen = np.isfinite(p).all(axis=1)
        p = np.where(seen[:, None], p, 0.0)
        count = np.concatenate([[0], np.cumsum(seen)])
        total = np.concatenate([np.zeros((1, 2)), np.cumsum(p, axis=0)])
        square = np.concatenate([np.zeros((1, 2)), np.cumsum(p * p, axis=0)])
        n = count[hi] - count[lo]
        ok = n >= need
        mean = (total[hi[ok]] - total[lo[ok]]) / n[ok, None]
        var = (square[hi[ok]] - square[lo[ok]]) / n[ok, None] - mean * mean
        spread = np.full(len(rows), np.nan)
        spread[ok] = np.sqrt(np.clip(var.sum(axis=1), 0.0, None))
        best = np.fmax(best, spread)
    return rows, key, half, best


def _window_median(key: np.ndarray, values: np.ndarray, half: int) -> np.ndarray:
    """(n,) float64 per row, the median of the finite `values` of the rows whose key lies within
    +- half of its own key, NaN when there is none. Keys are ascending and distinct, so those rows
    are among the half nearest rows on each side."""
    n = len(values)
    out = np.full(n, np.nan)
    shifts = np.arange(-half, half + 1)
    step = max(1, _MEDIAN_CELLS // len(shifts))
    for s in range(0, n, step):
        own = np.arange(s, min(n, s + step))
        idx = own[:, None] + shifts
        inside = (idx >= 0) & (idx < n)
        idx = np.clip(idx, 0, n - 1)
        cells = values[idx]
        ok = inside & (np.abs(key[idx] - key[own][:, None]) <= half) & np.isfinite(cells)
        cells = np.where(ok, cells, np.inf)
        cells.sort(axis=1)
        count = ok.sum(axis=1)
        has = np.flatnonzero(count)
        count = count[has]
        out[own[has]] = (cells[has, (count - 1) // 2] + cells[has, count // 2]) / 2
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
                  merge_gap: int, min_visit: int = 1, *, width: Optional[np.ndarray] = None, anchor: int = 0,
                  fit_iou: float = 0.0, fit_scale: float = 0.0, max_scale: float = 0.0, min_scale: float = 0.0,
                  half: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Spots and their segments (steps 2-3 of the module docstring).

    Args:
        boxes: (M,4) float64 posed boxes; frame: (M,) frame of each row; tracks: (M,) link_tracks
            ids; merge_gap: frames a spot may stay empty inside one segment; min_visit: tracks with
            fewer boxes belong to no spot. The spot identity of step 2 (SignerTrackParams with
            seconds in frames; the defaults are D14's: none): width: (M,) shoulder widths (None: no
            scale test); anchor: frames of an anchor; fit_iou, fit_scale, max_scale, min_scale as
            there; half: the boxes of a track within +- half frames give a box's shoulder width.
    Returns:
        segment: (M,) int64 segment of each row, -1 for a row outside every segment (a box of a
            short track, a box that does not overlap its track's spot, one that is too wide or too
            narrow, or a spot's second box in a frame). Segments hold at most one row per frame and
            are numbered in order of their first frame.
        spot: (S,) int64 spot of each segment; spots are numbered in founding order.
    """
    segment, spot, _, _ = _spot_segments(boxes, frame, tracks, link_iou, merge_gap, min_visit, width=width,
                                         anchor=anchor, fit_iou=fit_iou, fit_scale=fit_scale, max_scale=max_scale,
                                         min_scale=min_scale, half=half)
    return segment, spot


def _spot_segments(boxes: np.ndarray, frame: np.ndarray, tracks: np.ndarray, link_iou: float, merge_gap: int,
                   min_visit: int, *, width: Optional[np.ndarray] = None, anchor: int = 0, fit_iou: float = 0.0,
                   fit_scale: float = 0.0, max_scale: float = 0.0, min_scale: float = 0.0,
                   half: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """spot_segments, plus the (spots, 4) float64 box of every spot (the median box of the track
    that founded it), indexed by spot id, and the (M,) int64 spot of every row's track (-1 for a
    short track), whether or not the row itself overlaps that spot."""
    order = np.argsort(tracks, kind='stable')
    cuts = np.searchsorted(tracks[order], np.arange(tracks.max() + 2))
    median = np.array([np.median(boxes[order[cuts[j]:cuts[j + 1]]], axis=0) for j in range(len(cuts) - 1)])
    length = np.diff(cuts)
    scale = np.full(len(median), np.nan)   # median shoulder width of each track
    if width is not None:
        width = np.asarray(width, np.float64)
        for j in np.flatnonzero(length >= min_visit):
            seen = width[order[cuts[j]:cuts[j + 1]]]
            if np.isfinite(seen).any():
                scale[j] = np.nanmedian(seen)
    spot_of_track = np.full(len(median), -1, np.int64)
    anchor_track = np.zeros(len(median), bool)
    spot_box = np.zeros_like(median)   # rows past `spots` stay zero: spot_box[-1] is read for spot-less tracks
    spot_scale = np.full(len(median), np.nan)
    spots = 0
    for j in np.argsort(-length, kind='stable'):   # longest track first
        if length[j] < min_visit:
            break
        fit = _ious(np.broadcast_to(median[j], (spots, 4)), spot_box[:spots])
        k = int(np.argmax(fit)) if spots else -1
        if k >= 0 and fit[k] >= link_iou and _fits(median[j], scale[j], spot_box[k], spot_scale[k], fit_scale):
            anchors = median[anchor_track & (spot_of_track == k)]
            if length[j] >= anchor or _ious(np.broadcast_to(median[j], anchors.shape), anchors).max() >= fit_iou:
                spot_of_track[j], anchor_track[j] = k, length[j] >= anchor
                continue
        spot_box[spots], spot_scale[spots] = median[j], scale[j]   # a spot of its own
        spot_of_track[j], anchor_track[j] = spots, True
        spots += 1
    spot = spot_of_track[tracks]
    fit = _ious(spot_box[spot], boxes)
    inside = fit >= link_iou
    # An anchor that sat elsewhere than the spot's box (the interpreter shifted) is judged on its own box.
    elsewhere = anchor_track & (spot_of_track >= 0) & (_ious(median, spot_box[spot_of_track]) < fit_iou)
    inside |= elsewhere[tracks] & (_ious(median[tracks], boxes) >= link_iou)
    if (max_scale > 0 or min_scale > 0) and width is not None:   # anchor: its own width (a switch); else: the spot's
        ref = np.where(anchor_track & np.isfinite(scale), scale, spot_scale[spot_of_track])
        inside &= _narrow(np.flatnonzero(spot >= 0), frame, tracks, width, half, ref[tracks], min_scale,
                          max_scale if max_scale > 0 else np.inf)
    rows = np.flatnonzero(inside & (spot >= 0))
    if not len(rows):
        return np.full(len(boxes), -1, np.int64), np.zeros(0, np.int64), spot_box[:spots], spot
    rows = rows[np.lexsort((-fit[rows], frame[rows], spot[rows]))]   # by spot, frame, best fit first
    rows = rows[np.r_[True, (spot[rows][1:] != spot[rows][:-1]) | (frame[rows][1:] != frame[rows][:-1])]]
    starts = np.r_[True, (spot[rows][1:] != spot[rows][:-1]) | (np.diff(frame[rows]) - 1 > merge_gap)]
    rank = np.empty(int(starts.sum()), np.int64)
    rank[np.argsort(frame[rows][starts], kind='stable')] = np.arange(len(rank))
    segment = np.full(len(boxes), -1, np.int64)
    segment[rows] = rank[np.cumsum(starts) - 1]
    segment_spot = np.empty(len(rank), np.int64)
    segment_spot[rank] = spot[rows][starts]
    return segment, segment_spot, spot_box[:spots], spot


def _fits(box: np.ndarray, scale: float, spot_box: np.ndarray, spot_scale: float, fit_scale: float) -> bool:
    """Whether a track's median box and shoulder width are within a factor fit_scale of its spot's
    (box size: the square root of the area; an unmeasured width passes; fit_scale 0: no test)."""
    if fit_scale <= 0:
        return True
    if _area(spot_box) <= 0:
        return False
    ratios = (math.sqrt(_area(box) / _area(spot_box)), scale / spot_scale)
    return all(1 / fit_scale <= r <= fit_scale for r in ratios if not math.isnan(r))


def _narrow(rows: np.ndarray, frame: np.ndarray, tracks: np.ndarray, width: np.ndarray, half: int,
            ref: np.ndarray, low: float, high: float) -> np.ndarray:
    """(M,) bool: False for those of `rows` whose width, the median shoulder width of their track's
    boxes within +- half frames, is unmeasured or outside [low, high] times their reference `ref`
    (M,), where the reference is finite (a spot without a measured width tests nothing); True
    elsewhere."""
    out = np.ones(len(ref), bool)
    rows = rows[np.isfinite(ref[rows])]
    if not len(rows):
        return out
    rows = rows[np.lexsort((frame[rows], tracks[rows]))]
    key = tracks[rows].astype(np.int64) * (int(frame.max()) + 2 * half + 2) + frame[rows]
    wide = _window_median(key, width[rows], half)
    out[rows] = (wide >= low * ref[rows]) & (wide <= high * ref[rows])   # NaN (no shoulder pair in the window) fails
    return out


def _area(box: np.ndarray) -> float:
    return float((box[2] - box[0]) * (box[3] - box[1]))


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


def _main_spot(spot: np.ndarray, presence: np.ndarray, prior_key: np.ndarray, min_frames: float) -> int:
    """The main spot (step 4): among spots present in at least min_frames frames, the prior's
    favourite, then the most frames, then the lower id; -1 when no spot is present that often."""
    frames = np.bincount(spot, weights=presence)
    big = np.flatnonzero(frames >= min_frames)
    if not len(big):
        return -1
    return int(big[np.lexsort((-big, frames[big], prior_key[big]))[-1]])


def _signer_per_frame(frames: List[np.ndarray], actives: List[np.ndarray], prior_key: np.ndarray, window: int,
                      min_density: float, min_active: float, num_frames: int) -> np.ndarray:
    """(T,) int64 index of the signer segment (into `frames`) per frame, -1 where none (step 5).

    frames[j]: ascending frames where segment j has a box; segments ascend with their first frame;
    actives[j]: whether each of those boxes is active (e.g. signing_boxes; all True without an
    activity test). Score of a steady, active segment at t: its frames inside [t - window, t + window], then the
    prior key, then its presence, then the earlier segment; one int64 per (segment, frame) from dense ranks.
    """
    count = len(frames)
    prior_rank = np.unique(prior_key, return_inverse=True)[1].astype(np.int64)
    presence_rank = np.unique([len(f) for f in frames], return_inverse=True)[1].astype(np.int64)
    tie = (prior_rank * count + presence_rank) * count + (count - 1 - np.arange(count))
    best_score = np.full(num_frames, -1, np.int64)
    best = np.full(num_frames, -1, np.int64)
    for j, f in enumerate(frames):
        t = np.arange(f[0], f[-1] + 1)
        lo, hi = np.searchsorted(f, t - window, 'left'), np.searchsorted(f, t + window, 'right')
        cover = hi - lo
        moving = np.concatenate([[0], np.cumsum(actives[j])])
        span = np.minimum(f[-1], t + window) - np.maximum(f[0], t - window) + 1
        ok = (cover >= min_density * span) & (moving[hi] - moving[lo] >= min_active * cover)
        score = np.where(ok, cover * count ** 3 + tie[j], -1)
        better = score > best_score[t]
        best_score[t[better]] = score[better]
        best[t[better]] = j
    return best
