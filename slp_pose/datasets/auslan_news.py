"""Auslan-Daily TV news (News_V1 + News_V2): readable video ids, V1 splits and annotation names,
and this dataset's own signer rules.

Under <data root>/Auslan-Daily/ the videos are 'ABC News With Auslan - D M YYYY.mp4' in News_V1/
(2022-2023, sentence annotations in News_V1_Annotation.xlsx, whose Video_Name is 'D_M_YYYY';
'EA<n>' rows belong to the Web News sub-dataset and are not used) and News_V2/ (2023-2025, the
BANZ-FS extension; News_V2_Aligned_VTTS/<file stem>.vtt holds its sign-aligned subtitles). The
video id is 'abc_news_YYYY-MM-DD' (unique: the two subsets share no date). The workbook is read with
zipfile + ElementTree.

Signer rules (spec D14, D21; they serve this dataset only, so they live here and reach the
framework through AuslanNews.rules, D22). The interpreter stands in a panel on the right, beside
news footage or a studio newsreader who may be larger, and may hand over to an on-site interpreter
standing beside a speaker in the footage:
- `right_largest` (per frame; the GPU workers pose with it): the largest box whose centre lies in
  the right 40 % of the frame, else the largest box, so the panel interpreter is posed.
- `signer_track_right` (video level; the parent derives it on the CPU): the framework's
  signer_track steps 1-5 (signer.py; with D24's spot identity, the shoulder widths of the wrist
  measure) with
  (a) the right prior: the main spot and step-5 ties favour the spot with the larger median box
      centre x;
  (b) signing motion (signer.signing_boxes on the wrists, D21): a steady segment outside the main
      spot's segments (step 5) must sign in at least min_active of its boxes within the window (a
      steady newsreader, reporter or speaker does not);
  (c) step 6, fill_gaps: a gap is a stretch of at least fill_min_s inside a main segment's span
      without a box of the main spot's people (a box of a track whose spot is the main spot counts
      even where it does not overlap the spot: a person standing near the panel's place is no
      gap). The gap takes the other segment that is present in at least fill_cover of it and of
      whose boxes in it at least min_active sign (an on-site interpreter); among several, the one
      with the most signing boxes, then the most boxes, then the earlier segment: its box at t, or
      -1 where it has none, and its spot box as the region. A gap without one stays -1.
  (d) merge_gap_s 180 s (D25): the panel's segment bridges an absence of up to 3 minutes, so a
      full-screen insert of the bulletin (a press conference, a picture-in-picture interpreter
      while the panel is gone) is a gap of step 6 and takes one signer for its whole length: a
      picture-in-picture interpreter who interprets the bulletin is the signer while the inset is
      shown, and the frames between its showings stay -1 even when a signing interviewee fills
      the screen (D25). An absence over 3 minutes ends the segment (the interpreter leaving at the
      end of a broadcast; step 5 then keeps a centred V1 ending).
  Without keypoints (wrists=None) every box counts as signing in step 5, step 6 is skipped and the
  spot identity compares box sizes only; on a record the rule reads the wrists also with the motion
  test off (min_active 0), for the spot identity's shoulder widths. Its parameters (PARAMS, spelled
  out here) and their hashed form (RightTrackParams.fingerprint) are D21's with D24's spot identity
  and D25's merge gap.
  Known limit (D21, D25): at a handover in a split screen, the idle outgoing panel interpreter stays
  the signer for up to about 0.7 s after the incoming one starts signing (abc_news_2022-10-30
  around 1476 s); preferring the signing candidate at a handover would belong in this file's
  motion test.
"""
from __future__ import annotations

import dataclasses
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree

import numpy as np

from .. import signer
from ..select import FrameRule
from ..signer import MotionParams, PosedPeople, SignerChoice, SignerTrackParams, SpotLayout, VideoRule, Wrists
from . import register
from .base import Dataset, Video

V1_ANNOTATION = 'News_V1_Annotation.xlsx'   # inside Auslan-Daily/
_NAME = re.compile(r'ABC News With Auslan - (\d{1,2}) (\d{1,2}) (\d{4})')
_XLSX = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


# --------------------------------------------------------------------------- per-frame rule
RIGHT_MIN_X = 0.6   # right_largest: a box whose centre x is at or right of this fraction of W


def right_largest(boxes: np.ndarray, scores: np.ndarray, frame_size: Optional[Tuple[int, int]] = None) -> int:
    """Index of the largest box (as select.largest_bbox) among those whose centre x is >= RIGHT_MIN_X
    * W, or of the largest box when none is there. Needs frame_size."""
    if frame_size is None:
        raise ValueError('right_largest needs the frame size')
    b = np.asarray(boxes, np.float32)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    right = (b[:, 0] + b[:, 2]) / 2 >= RIGHT_MIN_X * frame_size[0]
    return int(np.argmax(np.where(right, area, -np.inf))) if right.any() else int(np.argmax(area))


# No params: the name alone is in the extraction hash, as before D22 (RIGHT_MIN_X is part of what
# the name means; a changed rule needs a new name).
RIGHT_LARGEST = FrameRule('right_largest', right_largest)


# --------------------------------------------------------------------------- video-level rule
def right_prior(boxes: np.ndarray) -> float:
    """The right prior's key of a spot (larger wins): the median box centre x of its boxes."""
    return float(np.median((boxes[:, 0] + boxes[:, 2]) / 2))


@dataclass(frozen=True)
class RightTrackParams:
    """Parameters of `signer_track_right` (module docstring): the spot rule's (track, with the right
    prior), the signing-motion measure's (motion) and this rule's own (D21). The motion test is on
    when min_active > 0; fill_gaps needs it."""

    track: SignerTrackParams = SignerTrackParams(link_iou=0.5, max_gap_s=1.0, min_track_s=20.0, window_s=60.0,
                                                 prior='right', merge_gap_s=180.0, min_visit_s=1.0, main_share=0.5,
                                                 min_density=0.8, max_visit_share=0.5, anchor_s=20.0, fit_iou=0.8,
                                                 fit_scale=1.25, max_scale=1.3, scale_window_s=1.0, min_scale=0.25,
                                                 point_conf=0.3)
    motion: MotionParams = MotionParams(motion_window_s=2.0, min_motion=0.12, point_conf=0.3, max_reach=3.0,
                                        min_seen=0.25, smooth_s=0.1, width_floor=0.1, min_sustain=0.75)
    min_active: float = 0.6    # a segment signs: this share of its boxes (step 5 window, step 6 gap) sign
    fill_gaps: bool = True     # step 6: main-spot gaps may take another signer (on-site interpreters)
    fill_min_s: float = 10.0   # step 6: gaps at least this long
    fill_cover: float = 0.5    # step 6: ... a segment present in at least this share of the gap

    def __post_init__(self) -> None:
        if self.track.prior != 'right':
            raise ValueError(f"signer_track_right needs the 'right' prior, got {self.track.prior!r}")
        if self.track.point_conf != self.motion.point_conf:   # the spot identity uses wrist_points' widths
            raise ValueError(f'track.point_conf {self.track.point_conf} != motion.point_conf {self.motion.point_conf}')
        if not (0.0 <= self.min_active <= 1.0 and isinstance(self.fill_gaps, bool) and self.fill_min_s >= 0
                and 0.0 < self.fill_cover <= 1.0 and not (self.fill_gaps and not self.uses_motion)):
            raise ValueError(f'bad signer_track_right parameters {self}')

    @property
    def uses_motion(self) -> bool:
        """Whether the rule tests signing motion (D21); without it, it is signer_track with the right prior."""
        return self.min_active > 0

    def fingerprint(self) -> Dict[str, object]:
        """The flat dict hashed into the derivation hash (VideoRule.params): the track's, the motion
        measure's and this rule's own parameters in one dict (D21's keys as signer.rule_params gave
        them before D22, plus D24's; the one point_conf is both the track's and the motion's)."""
        own = {name: getattr(self, name) for name in ('min_active', 'fill_gaps', 'fill_min_s', 'fill_cover')}
        return dict(dataclasses.asdict(self.track), **dataclasses.asdict(self.motion), **own)


PARAMS = RightTrackParams()


def signer_track_right(boxes: np.ndarray, offsets: np.ndarray, fps: Tuple[int, int], wrists: Optional[Wrists] = None,
                       params: RightTrackParams = PARAMS) -> SignerChoice:
    """The `signer_track_right` rule (module docstring) on a video's posed boxes (M,4) xyxy original
    pixels, frame-major, with (T+1,) CSR `offsets`, exact frame rate `fps` and the posed rows'
    signer.wrist_points (None: no motion test; their shoulder widths give the spot identity)."""
    layout = signer.spot_layout(boxes, offsets, fps, params.track, None if wrists is None else wrists.width)
    if not len(layout.spot):
        return layout.choice(np.full(layout.num_frames, -1, np.int64))
    motion = wrists is not None and params.uses_motion
    active = (signer.signing_boxes(layout.segment, layout.frame, layout.boxes, wrists, layout.rate, params.motion)
              if motion else None)
    chosen, main = signer.signer_segments(layout, params.track, signer.spot_keys(layout, right_prior), active,
                                          params.min_active if motion else 0.0)
    if motion and params.fill_gaps and main >= 0:
        fill_gaps(layout, chosen, main, active, params)
    return layout.choice(chosen)


def fill_gaps(layout: SpotLayout, chosen: np.ndarray, main: int, active: np.ndarray, params: RightTrackParams) -> None:
    """Step 6 (module docstring), in place on `chosen` ((T,) segment per frame, signer.signer_segments);
    main: the main spot; active: (M,) signer.signing_boxes."""
    seen = np.zeros(layout.num_frames, bool)   # a box of the main spot's people, on the spot or beside it
    seen[layout.frame[layout.track_spot == main]] = True
    actives = [active[rows] for rows in layout.rows]
    for start, stop, j in _gap_signers(list(layout.frames), actives, layout.spot, main, seen,
                                       max(1, round(params.fill_min_s * layout.rate)), params.fill_cover,
                                       params.min_active):
        chosen[start:stop] = j


def _gap_signers(frames: List[np.ndarray], actives: List[np.ndarray], spot: np.ndarray, main: int, seen: np.ndarray,
                 min_gap: int, cover: float, min_active: float) -> List[Tuple[int, int, int]]:
    """(start, stop, segment) per main-spot gap that takes another signer (step 6): a gap is a
    stretch [start, stop) of at least min_gap frames inside a main segment's span without a box of
    the main spot's people (`seen`: (T,) bool, a box of a track whose spot is the main spot, also
    one that does not overlap it); frames[j]: ascending frames where segment j has a box; actives[j]:
    whether each of those boxes signs; spot: (S,) spot of each segment."""
    first = np.array([f[0] for f in frames])
    last = np.array([f[-1] for f in frames])
    others = np.flatnonzero(spot != main)
    out = []
    for j in np.flatnonzero(spot == main):
        lo, hi = int(frames[j][0]), int(frames[j][-1]) + 1
        edges = np.flatnonzero(np.diff(np.concatenate([[0], (~seen[lo:hi]).astype(np.int8), [0]])))
        for start, stop in zip(lo + edges[::2], lo + edges[1::2]):
            if stop - start < min_gap:
                continue
            best: Optional[Tuple[Tuple[int, int, int], int]] = None
            for k in others[(first[others] < stop) & (last[others] >= start)]:
                a, b = np.searchsorted(frames[k], [start, stop])
                present = int(b - a)
                moving = int(np.count_nonzero(actives[k][a:b]))
                if present >= cover * (stop - start) and moving >= min_active * present:
                    key = (moving, present, -int(k))
                    if best is None or key > best[0]:
                        best = (key, int(k))
            if best is not None:
                out.append((int(start), int(stop), best[1]))
    return out


def _choose(people: PosedPeople, params: RightTrackParams = PARAMS) -> SignerChoice:
    """signer_track_right with `params` on a record's posed people: the wrists of their keypoints,
    read also without the motion test while the spot identity needs their shoulder widths."""
    wrists = (signer.wrist_points(people.kpts, people.frame_size, params.motion)
              if params.uses_motion or params.track.uses_scale else None)
    return signer_track_right(people.boxes, people.offsets, people.fps, wrists, params)


SIGNER_TRACK_RIGHT = VideoRule('signer_track_right', _choose, PARAMS.fingerprint())


# --------------------------------------------------------------------------- the dataset
@register
class AuslanNews(Dataset):
    """News_V1 + News_V2; split = the V1 workbook's split (None for V2); info: subset, annotation_name."""

    name = 'auslan_news'
    primary_rule = 'signer_track_right'
    extraction_rule = 'right_largest'
    rules = (RIGHT_LARGEST, SIGNER_TRACK_RIGHT)
    split_order = ('dev', 'test', 'train')

    @property
    def source(self) -> Path:
        return self.data_root / 'Auslan-Daily'

    def out_root(self) -> Path:
        return self.source / 'dwpose'

    def videos(self) -> List[Video]:
        splits = v1_splits(self.source / V1_ANNOTATION)
        out = []
        for path in sorted(self.source.glob('News_V[12]/*.mp4')):
            subset, name = path.parent.name, annotation_name(path)
            out.append(Video(video_id(path), path, splits.get(name) if subset == 'News_V1' else None,
                             dict(subset=subset, annotation_name=name)))
        return out


def _date(path: Path) -> Tuple[int, int, int]:
    match = _NAME.fullmatch(Path(path).stem)
    if match is None:
        raise ValueError(f"{path}: expected 'ABC News With Auslan - D M YYYY.mp4'")
    day, month, year = (int(g) for g in match.groups())
    return day, month, year


def video_id(path: Path) -> str:
    """'ABC News With Auslan - 16 10 2022.mp4' -> 'abc_news_2022-10-16'."""
    day, month, year = _date(path)
    return f'abc_news_{year:04d}-{month:02d}-{day:02d}'


def annotation_name(path: Path) -> str:
    """The video's name in its annotations: 'D_M_YYYY' (the Excel Video_Name) for News_V1, the
    file stem (the VTT name) for News_V2."""
    if Path(path).parent.name == 'News_V1':
        day, month, year = _date(path)
        return f'{day}_{month}_{year}'
    return Path(path).stem


def v1_splits(xlsx: Path) -> Dict[str, str]:
    """{Video_Name: split} of the TV-news rows of the News_V1 annotation workbook (first sheet;
    'EA<n>' Web News rows are skipped). Raises ValueError when a video has rows in more than one
    split."""
    with zipfile.ZipFile(xlsx) as zf:
        shared = [''.join(t.text or '' for t in si.iter(f'{_XLSX}t'))
                  for si in ElementTree.fromstring(zf.read('xl/sharedStrings.xml')).iter(f'{_XLSX}si')]
        sheet = ElementTree.fromstring(zf.read('xl/worksheets/sheet1.xml'))
    rows = [_cells(row, shared) for row in sheet.iter(f'{_XLSX}row')]
    columns = {name: col for col, name in rows[0].items()}
    out: Dict[str, str] = {}
    for row in rows[1:]:
        name, split = row.get(columns['Video_Name'], ''), row.get(columns['split'], '')
        if not name or name.startswith('EA'):
            continue
        if out.setdefault(name, split) != split:
            raise ValueError(f'{xlsx}: video {name!r} has rows in splits {out[name]!r} and {split!r}')
    return out


def _cells(row: ElementTree.Element, shared: List[str]) -> Dict[str, str]:
    """{column letters: text} of one sheet row (shared, inline and plain values)."""
    out = {}
    for cell in row.iter(f'{_XLSX}c'):
        column = re.match(r'[A-Z]+', cell.get('r', '')).group()
        value = cell.find(f'{_XLSX}v')
        if cell.get('t') == 's' and value is not None:
            out[column] = shared[int(value.text)]
        elif value is not None:
            out[column] = value.text or ''
        else:
            out[column] = ''.join(t.text or '' for t in cell.iter(f'{_XLSX}t'))
    return out
