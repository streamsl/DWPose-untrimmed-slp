"""Parity gate vs the official per-image reference (spec §4.7 item 3).

Reference ("official") for one frame, strict fp32 (env.strict_fp32), flip on, native frames:
  det = mmdet.apis.inference_detector(engines.build_detector(...), frame)  # one image per call
  pose set = person (label 0), score > 0.3, mmpose.evaluation.functional.nms(dets, 0.3)
  count set = person, score > 0.3, torchvision.ops.nms(boxes, scores, 0.45) (CPU)
  kpts = mmpose.apis.inference_topdown(engines.build_pose_model(...), frame, pose_set_boxes)
  (frames with an empty pose set are skipped, no full-image fallback).
Ours = worker.ChunkProcessor on video.chunk_from_frames(...) of the same decoded frames
(read_frames semantics, one forward decoding pass per video), with the requested backend.

Comparison, per frame:
- The official and our pose-set boxes are paired one-to-one (closest first, L-inf distance
  < MATCH_TOL_PX); likewise the count sets. A frame whose sets differ (size, or a box without a
  partner) is a mismatch, unless every unpaired box is borderline (|score - threshold| < borderline):
  such frames are excused and listed separately.
- The primary is picked by the dataset's per-frame extraction rule (the one the GPU workers pose
  with: largest_bbox for BOBSL, right_largest for Auslan news; a video-level primary rule is a CPU
  derive and not part of parity) on both sides; when both sides have one and they are different
  people, that is an identity mismatch (excused only when the frame's pose-set difference is
  borderline).
- A paired box that moved by >= max_box_diff_px while its detector score changed by at most
  nms_tie_ulps float32 ulps is a head-NMS near-tie: two priors of the same person with equal scores
  to float32 precision, and the head's NMS kept a different one on each side (numeric noise moves
  scores by far more ulps while moving boxes by < 0.01 px). Such people are excused, listed separately and left
  out of the box, score and keypoint metrics (measured: the official path itself does this when it
  batches its detections).
- Every other POSED person of ours whose box has an official partner is compared keypoint by
  keypoint in native pixels. A keypoint is confident when its official score > conf_thr. Moves are
  also measured in SimCC steps: one SimCC bin of the person's crop in original-frame pixels,
  step = scale_h / (input_h * simcc_split_ratio) with scale = prep.crop_params(official box) and
  input_h, simcc_split_ratio from the pose meta (= scale_w / (input_w * split_ratio) after the
  aspect-ratio fix). Numeric noise can flip a SimCC argmax to the neighbouring bin, which moves the
  keypoint by one step on one axis (sqrt(2) steps on both); a step is ~0.7 px for BOBSL-sized
  people but > 1 px for large people at 720p, so a pixel threshold is resolution-dependent. Two
  shares of confident keypoints are gated (all, lhand, rhand): those moving > step_tol steps
  (beyond a one-step flip; a rare-event limit) and those moving > flip_tol steps (left their SimCC
  bin: one-step flips included; a looser limit, so rare noise flips pass while a systematic
  one-bin shift, e.g. an off-by-one in our decode or flip-back, fails). Per part (body, feet, face,
  lhand, rhand, all) the report has both shares, the max in steps, and, for information only, the
  shares > 1 px and > 2 px, p95 / p99 (log-histogram upper bounds, bins 2.3 % wide) and the max in
  px. The gated keypoint score max diff covers the keypoints that moved <= step_tol steps: a
  keypoint that jumped to another SimCC peak changes its score by design and is already counted by
  the step shares (score_max_diff_raw keeps the max over all 133 keypoints for information).

Optional jitter baseline (off by default): the same comparison of numerically equivalent
reformulations of the official path against it, to show how much the reference itself moves:
'batched_det' = mmdet detection of up to 64 frames per test_step (strict fp32), 'tf32' = the
per-image official path with cuDNN TF32 on (torch's default numerics). Both use official
inference_topdown on their own boxes and pose the same K people as ours (select.select_frame).
The baselines' own rows are never gated, but 'batched_det' (GATE_JITTER, the official code at its
own fp32 precision) sets jitter-relative limits (gate_limits): each gated keypoint share and the
keypoint score max may go up to jitter_factor x the baseline's value of the same metric and part
when that is above the fixed limit; count / identity / box / detector score limits stay fixed.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .datasets import Dataset, Video, load_videos
from .settings import Settings
from .types import COUNT_SET, POSE_SET, ChunkDets, ChunkPoses, VideoInfo

if TYPE_CHECKING:
    import torch

    from .engines import PoseMeta
    from .select import FrameRule

PARTS: Dict[str, slice] = {'all': slice(0, 133), 'body': slice(0, 17), 'feet': slice(17, 23),
                           'face': slice(23, 91), 'lhand': slice(91, 112), 'rhand': slice(112, 133)}
GATED_PARTS = ('all', 'lhand', 'rhand')
MATCH_TOL_PX = 1.0     # the same detection on both sides differs by ~1e-3 px; two people by far more
MAX_LISTED = 100       # cases kept in each of report.borderline / report.mismatches (counters are exact)
# Distance histogram bins: [0, 1e-6) then log-spaced up to 1e4 px (100 per decade) -> O(1) memory.
_HIST_EDGES = np.concatenate([[0.0], np.logspace(-6, 4, 1001)])
# Official scores of confident keypoints moving > step_tol steps, in bins of 0.05 from 0.3 (last bin: >= 1).
_OUTLIER_SCORE_EDGES = np.round(np.arange(0.3, 1.0001, 0.05), 2)
# Confident keypoints by move in SimCC steps (informational), from 0.5 steps; the bins are centred on whole
# step counts (an n-bin argmax flip on one axis moves n steps), so one-step flips land in 0.5-1.5. The first
# edge is the default flip_tol: the histogram counts the keypoints of kp_frac_flipped.
_STEP_EDGES = np.array([0.5, 1.5, 2.5, 3.5, 5.5, 10.5])
_STEP_LABELS = [f'{lo:g}-{hi:g}' for lo, hi in zip(_STEP_EDGES[:-1], _STEP_EDGES[1:])] + [f'>={_STEP_EDGES[-1]:g}']
JITTER_VARIANTS = ('batched_det', 'tf32')
GATE_JITTER = 'batched_det'   # the baseline the jitter-relative limits scale (tf32 is lower precision, not noise)

Plan = List[Tuple[VideoInfo, List[Tuple[int, int]]]]
Rule = Callable[[np.ndarray, np.ndarray, Optional[Tuple[int, int]]], int]   # a select.FrameRule


@dataclass(frozen=True)
class ParityThresholds:
    """Acceptance thresholds of the lossless gate (spec §4.7 item 3)."""

    conf_thr: float = 0.3                  # a keypoint is 'confident' when the official score > this
    step_tol: float = 1.5                  # a keypoint moved when it moved > this many SimCC steps (> sqrt(2))
    max_kp_frac_gt_step: float = 5e-5      # <= 0.005 % confident keypoints move > step_tol steps (all, per hand)
    flip_tol: float = 0.5                  # a keypoint left its SimCC bin when it moved > this many steps
    max_kp_frac_flipped: float = 2e-3      # <= 0.2 % confident keypoints move > flip_tol steps (all, per hand):
    # ~6x the Auslan > 1 px share (mostly flips; lhand 3.3e-4), far below a one-bin bias of one hand joint (1/21)
    max_box_diff_px: float = 0.01          # max |box coordinate| difference on matched pose-set boxes
    max_kpt_score_diff: float = 0.01       # max |keypoint score| difference, keypoints within step_tol steps
    max_det_score_diff: float = 1e-3       # max |detector score| difference on matched pose-set boxes
    borderline: float = 1e-4               # |det score - threshold| below this excuses a count/identity mismatch
    nms_tie_ulps: int = 4                  # |det score| diff within this many float32 ulps + box moved -> near-tie
    jitter_factor: float = 1.5             # keypoint shares / score max may reach this x the GATE_JITTER baseline


@dataclass
class ParityReport:
    """Everything the gate looks at; JSON-serialisable via dataclasses.asdict."""

    backend: str
    videos: List[str]
    windows: List[Tuple[str, int, int]]            # (video_id, start_frame, num_frames)
    frames: int = 0
    frames_compared: int = 0                        # frames with a non-empty official pose set
    pose_count_mismatches: int = 0                  # frames whose pose sets differ (size or members), not borderline
    count_set_mismatches: int = 0                   # frames whose count sets differ (size or members), not borderline
    identity_mismatches: int = 0                    # primary (the extraction rule) is a different person
    borderline_excused: int = 0
    box_max_diff_px: float = 0.0                    # matched pose-set boxes, NMS near-ties excluded
    score_max_diff: float = 0.0                     # keypoints of the compared people within step_tol steps
    kp_confident: int = 0
    kp_frac_gt_step: Dict[str, float] = field(default_factory=dict)  # > step_tol SimCC steps (gated: all, hands)
    kp_frac_flipped: Dict[str, float] = field(default_factory=dict)  # > flip_tol SimCC steps (gated: all, hands)
    kp_frac_gt_1px: Dict[str, float] = field(default_factory=dict)   # 'all', 'lhand', 'rhand', 'body', 'face', 'feet'
    kp_max_diff_px: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    # Additions to the contract fields:
    primary_rule: str = ''                          # the dataset's final rule (informational)
    extraction_rule: str = ''                       # the per-frame rule compared (ours and official)
    complete: bool = False                          # False while run_parity is still going (partial JSON)
    persons_compared: int = 0                       # our posed people compared keypoint by keypoint
    persons_unmatched: int = 0                      # our posed people without an official partner box
    det_score_max_diff: float = 0.0                 # detector scores of the matched pose-set boxes
    kp_confident_parts: Dict[str, int] = field(default_factory=dict)
    kp_frac_gt_2px: Dict[str, float] = field(default_factory=dict)
    kp_max_diff_steps: Dict[str, float] = field(default_factory=dict)
    kp_steps_hist: Dict[str, int] = field(default_factory=dict)      # confident keypoints by move in steps (>= 0.5)
    simcc_step_px: Dict[str, float] = field(default_factory=dict)    # min / mean / max step of the compared people
    kp_p95_px: Dict[str, float] = field(default_factory=dict)
    kp_p99_px: Dict[str, float] = field(default_factory=dict)
    score_max_diff_raw: float = 0.0                 # all 133 keypoints of the compared people (not gated)
    nms_ties_excused: int = 0                       # paired people excused as head-NMS near-ties
    conf_thr: float = ParityThresholds.conf_thr     # thresholds the counts were computed with
    borderline_thr: float = ParityThresholds.borderline
    nms_tie_ulps: int = ParityThresholds.nms_tie_ulps
    step_tol: float = ParityThresholds.step_tol
    flip_tol: float = ParityThresholds.flip_tol
    borderline: List[Dict[str, object]] = field(default_factory=list)   # excused cases (first MAX_LISTED)
    nms_ties: List[Dict[str, object]] = field(default_factory=list)     # excused near-ties (first MAX_LISTED)
    mismatches: List[Dict[str, object]] = field(default_factory=list)   # unexcused cases (first MAX_LISTED)
    kp_outliers: List[Dict[str, object]] = field(default_factory=list)  # confident > step_tol steps (first MAX_LISTED)
    kp_outlier_joints: Dict[int, int] = field(default_factory=dict)      # keypoint index -> count (all of them)
    kp_outlier_scores: Dict[str, int] = field(default_factory=dict)      # official-score bin -> count (all of them)
    worst: Dict[str, Dict[str, object]] = field(default_factory=dict)    # where the box / score max diffs occur
    jitter: Dict[str, Dict[str, object]] = field(default_factory=dict)   # variant -> the same numbers (gate_limits)
    seconds: float = 0.0
    provenance: Dict[str, object] = field(default_factory=dict)


@dataclass
class ParityModels:
    """Official models and our engines on one CUDA device."""

    det_model: object       # engines.build_detector
    pose_model: object      # engines.build_pose_model (flip_test on)
    det_engine: object      # engines.load_engines for settings.backend
    pose_engine: object
    device: torch.device
    official_nms: str = 'mmcv'   # 'detpost.head_nms' where mmcv's CUDA nms has no code for the GPU


# --------------------------------------------------------------------------- one frame, both sides
def official_frame(det_model, pose_model, frame, settings: Settings = Settings(), pred=None) -> Dict[str, object]:
    """Official reference for one BGR frame: {'boxes': (n,4) float32, 'scores': (n,) float32 of
    the pose set in mmpose-nms order, 'count': int, 'kpts_px': (n,133,2) float32,
    'kpt_scores': (n,133) float32}, plus the count set as 'count_boxes' (c,4) / 'count_scores' (c,),
    and every person detection ('person_boxes', 'person_scores') with the pose / count sets as
    indices into them ('pose_index', 'count_index').

    Thresholds are the settings' pose/count rules (0.3 / 0.3 and 0.3 / 0.45 by default). `pred`
    (numpy pred_instances) replaces the per-image inference_detector call (jitter baseline).
    """
    import torch
    import torchvision
    from mmdet.apis import inference_detector
    from mmpose.apis import inference_topdown
    from mmpose.evaluation.functional import nms

    if pred is None:
        pred = inference_detector(det_model, frame).pred_instances.cpu().numpy()
    person = pred.labels == 0
    boxes, scores = pred.bboxes[person].astype(np.float32), pred.scores[person].astype(np.float32)
    high = np.flatnonzero(scores > settings.pose_score_thr)
    pose = high[nms(np.concatenate([boxes[high], scores[high, None]], axis=1), settings.pose_nms_thr)]
    high = np.flatnonzero(scores > settings.count_score_thr)
    count = high[torchvision.ops.nms(torch.from_numpy(boxes[high]), torch.from_numpy(scores[high]),
                                     settings.count_nms_thr).numpy()]
    if len(pose):
        results = inference_topdown(pose_model, frame, boxes[pose])
        kpts = np.stack([r.pred_instances.keypoints[0] for r in results]).astype(np.float32)
        kpt_scores = np.stack([r.pred_instances.keypoint_scores[0] for r in results]).astype(np.float32)
    else:
        kpts, kpt_scores = np.zeros((0, 133, 2), np.float32), np.zeros((0, 133), np.float32)
    return dict(boxes=boxes[pose], scores=scores[pose], count=len(count), kpts_px=kpts, kpt_scores=kpt_scores,
                count_boxes=boxes[count], count_scores=scores[count], person_boxes=boxes, person_scores=scores,
                pose_index=pose, count_index=count)


def as_ours(official: Dict[str, object], frame_size: Tuple[int, int], k: int,
            rule: Union[str, FrameRule]) -> Dict[str, object]:
    """An official_frame result in our_frame form: rows = its person detections, the posed rows and
    primary chosen by select.select_frame with the per-frame `rule` (as ours), keypoints in the saved
    encoding."""
    from . import select
    boxes, scores = official['person_boxes'], official['person_scores']
    flags = np.zeros(len(scores), np.uint8)
    flags[official['pose_index']] |= POSE_SET
    flags[official['count_index']] |= COUNT_SET
    posed, primary = select.select_frame(boxes, scores, (flags & POSE_SET) > 0, k, rule, tuple(frame_size))
    where = {int(row): j for j, row in enumerate(official['pose_index'])}
    rows = [where[int(row)] for row in posed]
    kpts = np.zeros((len(rows), 133, 3), np.float32)
    kpts[..., 0] = official['kpts_px'][rows, :, 0] / np.float32(frame_size[0])
    kpts[..., 1] = official['kpts_px'][rows, :, 1] / np.float32(frame_size[1])
    kpts[..., 2] = official['kpt_scores'][rows]
    return dict(boxes=boxes, scores=scores, flags=flags, posed=posed, kpts=kpts, primary=int(primary))


@contextlib.contextmanager
def _cudnn_tf32():
    """cuDNN TF32 on (torch's default) inside the block, strict fp32 again after it."""
    import torch
    from . import env
    torch.backends.cudnn.allow_tf32 = True
    try:
        yield
    finally:
        env.strict_fp32()


def _jitter_frames(variant: str, models: ParityModels, frames: np.ndarray, settings: Settings
                   ) -> List[Dict[str, object]]:
    """official_frame results of `frames` under a numerically equivalent reformulation (JITTER_VARIANTS)."""
    import torch
    from mmcv.transforms import Compose
    from mmdet.utils import get_test_pipeline_cfg
    if variant == 'tf32':
        with _cudnn_tf32():
            return [official_frame(models.det_model, models.pose_model, f, settings) for f in frames]
    if variant != 'batched_det':
        raise KeyError(f'unknown jitter variant {variant!r}; known: {JITTER_VARIANTS}')
    steps = get_test_pipeline_cfg(models.det_model.cfg.copy())
    steps[0].type = 'mmdet.LoadImageFromNDArray'   # as inference_detector does for arrays
    pipeline = Compose(steps)
    preds = []
    for s in range(0, len(frames), 64):
        data = [pipeline(dict(img=f, img_id=0)) for f in frames[s:s + 64]]
        with torch.no_grad():
            results = models.det_model.test_step(dict(inputs=[d['inputs'] for d in data],
                                                      data_samples=[d['data_samples'] for d in data]))
        preds += [r.pred_instances.cpu().numpy() for r in results]
    return [official_frame(models.det_model, models.pose_model, f, settings, pred=p) for f, p in zip(frames, preds)]


def our_frame(dets: ChunkDets, poses: ChunkPoses, t: int) -> Dict[str, object]:
    """Frame t of our chunk results as ParityAccumulator.add_frame takes them: the frame's det rows
    (boxes, scores, flags), `posed` frame-local det rows, their saved-encoding `kpts`, `primary`."""
    rows = dets.rows(t)
    k = slice(int(poses.offsets[t]), int(poses.offsets[t + 1]))
    return dict(boxes=dets.boxes[rows], scores=dets.scores[rows], flags=dets.flags[rows],
                posed=poses.det_index[k].astype(np.int64) - rows.start, kpts=poses.kpts[k],
                primary=int(poses.primary[t]))


def _within_ulps(a: float, b: float, ulps: int) -> bool:
    """|a - b| is at most `ulps` float32 units in the last place of the larger magnitude."""
    a32, b32 = np.float32(a), np.float32(b)
    return bool(abs(float(a32) - float(b32)) <= ulps * float(np.spacing(max(abs(a32), abs(b32)))))


def match_boxes(a: np.ndarray, b: np.ndarray, tol: float = MATCH_TOL_PX
                ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """One-to-one pairing of two xyxy box lists, closest first, L-inf distance < tol.

    Returns (pairs [(i, j)], unpaired indices of a, unpaired indices of b).
    """
    pairs: List[Tuple[int, int]] = []
    if len(a) and len(b):
        dist = np.abs(a[:, None, :].astype(np.float64) - b[None, :, :]).max(-1)
        used_a, used_b = set(), set()
        for flat in np.argsort(dist, axis=None, kind='stable'):
            i, j = divmod(int(flat), len(b))
            if dist[i, j] >= tol:
                break
            if i not in used_a and j not in used_b:
                pairs.append((i, j))
                used_a.add(i)
                used_b.add(j)
    paired_a, paired_b = {i for i, _ in pairs}, {j for _, j in pairs}
    return pairs, [i for i in range(len(a)) if i not in paired_a], [j for j in range(len(b)) if j not in paired_b]


def _rows(boxes: np.ndarray, scores: np.ndarray) -> List[List[float]]:
    return np.concatenate([boxes, scores[:, None]], axis=1).astype(np.float64).tolist()


def simcc_steps(boxes: np.ndarray, pose_meta: PoseMeta) -> np.ndarray:
    """(n,) float64: one SimCC bin of each xyxy box's pose crop in original-frame pixels,
    scale_h / (input_h * simcc_split_ratio) with scale = prep.crop_params(box) (padding 1.25 and the
    aspect-ratio fix, so it equals scale_w / (input_w * simcc_split_ratio)); a SimCC argmax that
    flips to the neighbouring bin moves the keypoint by one step."""
    from .prep import POSE_INPUT_SIZE, crop_params
    if tuple(pose_meta.input_size) != tuple(POSE_INPUT_SIZE):
        raise ValueError(f'pose meta input size {pose_meta.input_size} != prep crops {POSE_INPUT_SIZE}')
    if not len(boxes):
        return np.zeros(0, np.float64)
    _, scales = crop_params(boxes)
    return scales[:, 1].astype(np.float64) / (float(pose_meta.input_size[1]) * float(pose_meta.simcc_split_ratio))


def _quantile(hist: np.ndarray, q: float, maximum: float) -> float:
    """Upper bound of the q-quantile (nearest rank) from a _HIST_EDGES histogram, capped by the exact max."""
    total = int(hist.sum())
    if total == 0:
        return 0.0
    k = int(np.searchsorted(np.cumsum(hist), max(1, math.ceil(q * total))))
    return float(min(_HIST_EDGES[k + 1], maximum))


class ParityAccumulator:
    """Folds per-frame comparisons (official_frame vs our_frame) into the ParityReport numbers.

    `rule` is the per-frame extraction rule (a select.FrameRule: Dataset.frame_rule) applied to the official
    pose set; it is called as rule(boxes, scores, frame_size). `pose_meta` (engines.pose_meta) gives
    the pose input size and SimCC split ratio of the step (simcc_steps).
    """

    def __init__(self, settings: Settings, rule: Rule, pose_meta: PoseMeta,
                 thresholds: ParityThresholds = ParityThresholds()) -> None:
        self.settings, self.rule, self.pose_meta, self.thresholds = settings, rule, pose_meta, thresholds
        simcc_steps(np.zeros((0, 4), np.float32), pose_meta)   # fail early on a pose meta prep does not crop for
        self.counts = dict(frames=0, frames_compared=0, borderline_excused=0, persons_compared=0, persons_unmatched=0,
                           nms_ties_excused=0)
        self.mismatch_counts = {'pose set': 0, 'count set': 0, 'identity': 0}
        self.box_max = self.det_score_max = self.score_max = self.score_max_raw = 0.0
        self.confident = dict.fromkeys(PARTS, 0)
        self.gt_step = dict.fromkeys(PARTS, 0)
        self.flipped = dict.fromkeys(PARTS, 0)
        self.gt1 = dict.fromkeys(PARTS, 0)
        self.gt2 = dict.fromkeys(PARTS, 0)
        self.max_px = dict.fromkeys(PARTS, 0.0)
        self.max_steps = dict.fromkeys(PARTS, 0.0)
        self.hist = {part: np.zeros(len(_HIST_EDGES) - 1, np.int64) for part in PARTS}
        self.steps_hist = np.zeros(len(_STEP_EDGES), np.int64)
        self.step_min, self.step_max, self.step_sum = math.inf, 0.0, 0.0   # over the compared people
        self.borderline: List[Dict[str, object]] = []
        self.mismatches: List[Dict[str, object]] = []
        self.nms_ties: List[Dict[str, object]] = []
        self.kp_outliers: List[Dict[str, object]] = []
        self.outlier_joints = np.zeros(133, np.int64)
        self.outlier_scores = np.zeros(len(_OUTLIER_SCORE_EDGES), np.int64)
        self.worst: Dict[str, Dict[str, object]] = {}

    def add_frame(self, video_id: str, frame_index: int, frame_size: Tuple[int, int], official: Dict[str, object],
                  ours: Dict[str, object]) -> None:
        """Compare one frame; frame_size (W,H) converts our x/W, y/H back to native pixels."""
        s = self.settings
        where = dict(video=video_id, frame=int(frame_index))
        self.counts['frames'] += 1
        self.counts['frames_compared'] += int(len(official['scores']) > 0)
        pose_rows = np.flatnonzero(ours['flags'] & POSE_SET)
        pairs, pose_verdict = self._compare_set('pose set', where, s.pose_score_thr, official['boxes'],
                                                official['scores'], ours['boxes'][pose_rows], ours['scores'][pose_rows])
        count_rows = np.flatnonzero(ours['flags'] & COUNT_SET)
        self._compare_set('count set', where, s.count_score_thr, official['count_boxes'], official['count_scores'],
                          ours['boxes'][count_rows], ours['scores'][count_rows])
        partner, box_diff = {}, {}  # our frame-local det row -> official pose-set index, box L-inf diff
        tied = set()                # our rows excused as head-NMS near-ties
        for i, j in pairs:
            row = int(pose_rows[j])
            partner[row] = i
            box_diff[row] = float(np.abs(official['boxes'][i].astype(np.float64) - ours['boxes'][row]).max())
            det_diff = abs(float(official['scores'][i]) - float(ours['scores'][row]))
            case = dict(where, px=box_diff[row],
                        official=_rows(official['boxes'][i:i + 1], official['scores'][i:i + 1]),
                        ours=_rows(ours['boxes'][row:row + 1], ours['scores'][row:row + 1]))
            if box_diff[row] >= self.thresholds.max_box_diff_px and _within_ulps(
                    official['scores'][i], ours['scores'][row], self.thresholds.nms_tie_ulps):
                tied.add(row)
                self.counts['nms_ties_excused'] += 1
                if len(self.nms_ties) < MAX_LISTED:
                    self.nms_ties.append(dict(case, det_score_diff=det_diff))
                continue
            if box_diff[row] > self.box_max:
                self.box_max = box_diff[row]
                self.worst['box'] = case
            self.det_score_max = max(self.det_score_max, det_diff)
        self._compare_identity(where, frame_size, pose_verdict, official, ours, partner)
        self._compare_keypoints(where, frame_size, official, ours, partner, box_diff, tied,
                                simcc_steps(official['boxes'], self.pose_meta))

    def _record(self, kind: str, case: Dict[str, object], excused: bool) -> None:
        if excused:
            self.counts['borderline_excused'] += 1
            listing = self.borderline
        else:
            self.mismatch_counts[kind] += 1
            listing = self.mismatches
        if len(listing) < MAX_LISTED:
            listing.append(dict(case, kind=kind))

    def _compare_set(self, kind: str, where: Dict[str, object], thr: float, off_boxes: np.ndarray,
                     off_scores: np.ndarray, our_boxes: np.ndarray, our_scores: np.ndarray
                     ) -> Tuple[List[Tuple[int, int]], str]:
        """Pair the two sets; returns (pairs, 'same' | 'borderline' | 'mismatch')."""
        pairs, lone_off, lone_ours = match_boxes(off_boxes, our_boxes)
        if not lone_off and not lone_ours:
            return pairs, 'same'
        lone = np.concatenate([off_scores[lone_off], our_scores[lone_ours]]).astype(np.float64)
        excused = bool(np.all(np.abs(lone - thr) < self.thresholds.borderline))
        self._record(kind, dict(where, official=_rows(off_boxes[lone_off], off_scores[lone_off]),
                                ours=_rows(our_boxes[lone_ours], our_scores[lone_ours])), excused)
        return pairs, 'borderline' if excused else 'mismatch'

    def _compare_identity(self, where: Dict[str, object], frame_size: Tuple[int, int], pose_verdict: str,
                          official: Dict[str, object], ours: Dict[str, object], partner: Dict[int, int]) -> None:
        if not len(official['scores']) or ours['primary'] < 0:
            return  # a missing primary on one side is already a pose-set difference
        i = int(self.rule(official['boxes'], official['scores'], tuple(frame_size)))
        mine = int(ours['posed'][ours['primary']])
        if partner.get(mine) != i:
            case = dict(where, official=_rows(official['boxes'][i:i + 1], official['scores'][i:i + 1]),
                        ours=_rows(ours['boxes'][mine:mine + 1], ours['scores'][mine:mine + 1]))
            self._record('identity', case, pose_verdict == 'borderline')

    def _compare_keypoints(self, where: Dict[str, object], frame_size: Tuple[int, int], official: Dict[str, object],
                           ours: Dict[str, object], partner: Dict[int, int], box_diff: Dict[int, float],
                           tied: set, steps: np.ndarray) -> None:
        """`steps` = simcc_steps of the official pose-set boxes (the step of a compared person)."""
        width, height = frame_size
        tol = self.thresholds.step_tol
        for p, row in enumerate(ours['posed']):
            i = partner.get(int(row))
            if i is None:
                self.counts['persons_unmatched'] += 1
                continue
            if int(row) in tied:
                continue
            self.counts['persons_compared'] += 1
            mine = ours['kpts'][p].astype(np.float64)
            ref = official['kpts_px'][i].astype(np.float64)
            ref_scores = official['kpt_scores'][i].astype(np.float64)
            dist = np.hypot(mine[:, 0] * width - ref[:, 0], mine[:, 1] * height - ref[:, 1])
            step = float(steps[i])
            self.step_min, self.step_max = min(self.step_min, step), max(self.step_max, step)
            self.step_sum += step
            moves = dist / step if step > 0 else np.where(dist > 0, np.inf, 0.0)   # in SimCC steps
            score_diff = np.abs(mine[:, 2] - ref_scores)
            self.score_max_raw = max(self.score_max_raw, float(score_diff.max()))
            stayed = np.where(moves <= tol, score_diff, -1.0)   # jumped keypoints are gated by the step shares
            k = int(np.argmax(stayed))
            if stayed[k] > self.score_max:
                self.score_max = float(stayed[k])
                self.worst['score'] = dict(where, kpt=k, official=float(ref_scores[k]), ours=float(mine[k, 2]),
                                           px=float(dist[k]), step_px=step, steps=float(moves[k]),
                                           box_diff_px=box_diff[int(row)])
            confident = ref_scores > self.thresholds.conf_thr
            moved = np.flatnonzero(confident & (moves > tol))
            self.outlier_joints[moved] += 1
            score_bins = np.searchsorted(_OUTLIER_SCORE_EDGES, ref_scores[moved], side='right') - 1
            np.add.at(self.outlier_scores, score_bins, 1)
            for k in moved[:max(MAX_LISTED - len(self.kp_outliers), 0)]:
                self.kp_outliers.append(dict(where, kpt=int(k), score=float(ref_scores[k]), px=float(dist[k]),
                                             step_px=step, steps=float(moves[k]), box_diff_px=box_diff[int(row)]))
            far = moves[confident]
            far = far[far >= _STEP_EDGES[0]]
            self.steps_hist += np.bincount(np.searchsorted(_STEP_EDGES, far, side='right') - 1,
                                           minlength=len(_STEP_EDGES))
            for part, sl in PARTS.items():
                d = dist[sl][confident[sl]]
                if not len(d):
                    continue
                m = moves[sl][confident[sl]]
                self.confident[part] += len(d)
                self.gt_step[part] += int((m > tol).sum())
                self.flipped[part] += int((m > self.thresholds.flip_tol).sum())
                self.max_steps[part] = max(self.max_steps[part], float(m.max()))
                self.gt1[part] += int((d > 1.0).sum())
                self.gt2[part] += int((d > 2.0).sum())
                self.max_px[part] = max(self.max_px[part], float(d.max()))
                bins = np.clip(np.searchsorted(_HIST_EDGES, d, side='right') - 1, 0, len(_HIST_EDGES) - 2)
                self.hist[part] += np.bincount(bins, minlength=len(_HIST_EDGES) - 1)

    def fill(self, report: ParityReport) -> None:
        """Write the accumulated numbers into `report`."""
        for name, value in self.counts.items():
            setattr(report, name, value)
        report.pose_count_mismatches = self.mismatch_counts['pose set']
        report.count_set_mismatches = self.mismatch_counts['count set']
        report.identity_mismatches = self.mismatch_counts['identity']
        report.box_max_diff_px, report.det_score_max_diff = self.box_max, self.det_score_max
        report.score_max_diff, report.score_max_diff_raw = self.score_max, self.score_max_raw
        report.kp_confident = self.confident['all']
        report.kp_confident_parts = dict(self.confident)

        def share(counts: Dict[str, int]) -> Dict[str, float]:
            return {p: counts[p] / self.confident[p] if self.confident[p] else 0.0 for p in PARTS}

        report.kp_frac_gt_step, report.kp_frac_flipped = share(self.gt_step), share(self.flipped)
        report.kp_frac_gt_1px, report.kp_frac_gt_2px = share(self.gt1), share(self.gt2)
        report.kp_max_diff_steps = dict(self.max_steps)
        report.kp_steps_hist = dict(zip(_STEP_LABELS, self.steps_hist.tolist()))
        people = self.counts['persons_compared']
        report.simcc_step_px = (dict(min=self.step_min, mean=self.step_sum / people, max=self.step_max)
                                if people else {})
        report.kp_p95_px = {p: _quantile(self.hist[p], 0.95, self.max_px[p]) for p in PARTS}
        report.kp_p99_px = {p: _quantile(self.hist[p], 0.99, self.max_px[p]) for p in PARTS}
        report.kp_max_diff_px = dict(self.max_px)
        report.conf_thr, report.borderline_thr = self.thresholds.conf_thr, self.thresholds.borderline
        report.nms_tie_ulps, report.step_tol = self.thresholds.nms_tie_ulps, self.thresholds.step_tol
        report.flip_tol = self.thresholds.flip_tol
        report.borderline, report.mismatches = list(self.borderline), list(self.mismatches)
        report.nms_ties = list(self.nms_ties)
        report.kp_outliers = list(self.kp_outliers)
        report.worst = dict(self.worst)
        report.kp_outlier_joints = {int(k): int(self.outlier_joints[k]) for k in np.flatnonzero(self.outlier_joints)}
        labels = [f'{lo:.2f}-{hi:.2f}' for lo, hi in zip(_OUTLIER_SCORE_EDGES[:-1], _OUTLIER_SCORE_EDGES[1:])]
        report.kp_outlier_scores = dict(zip(labels + ['>=1.00'], self.outlier_scores.tolist()))


# --------------------------------------------------------------------------- sampling
def spread_windows(num_frames: int, windows: int, length: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    """`windows` sorted, non-overlapping (start, length) windows, one at a random offset inside each of
    `windows` equal segments of the video; [(0, num_frames)] when they do not fit."""
    if windows * length >= num_frames:
        return [(0, num_frames)]
    out = []
    for k in range(windows):
        lo, hi = k * num_frames // windows, (k + 1) * num_frames // windows
        out.append((lo + int(rng.integers(0, hi - lo - length + 1)), length))
    return out


def _video_order(videos: Sequence[Video], split_order: Sequence[str], rng: np.random.Generator) -> List[str]:
    """Every id, alternating between the splits (split_order, then the videos of no listed split),
    each split in a seeded random order."""
    groups = [[v.video_id for v in videos if v.split == split] for split in split_order]
    groups.append([v.video_id for v in videos if v.split not in split_order])
    queues = [[str(v) for v in rng.permutation(sorted(group))] for group in groups if group]
    order: List[str] = []
    while any(queues):
        for queue in queues:
            if queue:
                order.append(queue.pop(0))
    return list(dict.fromkeys(order))


def plan_windows(dataset: Dataset, num_videos: int = 20, windows: int = 3,
                 window_s: float = 30.0, full_videos: int = 2, seed: int = 0,
                 video_ids: Optional[Sequence[str]] = None) -> Plan:
    """The frames run_parity compares: [(video, [(start, count), ...])], windowed videos first.

    Videos are drawn round-robin across the dataset's splits, each split in a random order
    (numpy default_rng(seed)); `video_ids` replaces the drawn windowed videos. Each windowed video
    gets `windows` windows of `window_s` seconds spread through it (spread_windows); the next
    `full_videos` drawn videos are compared in full.
    """
    from .video import probe
    videos = load_videos(dataset)
    paths = {video.video_id: video.path for video in videos}
    rng = np.random.default_rng(seed)
    order = _video_order(videos, dataset.split_order, rng)
    if video_ids is None:
        windowed = order[:num_videos]
    else:
        unknown = [v for v in video_ids if v not in paths]
        if unknown:
            raise ValueError(f'{dataset.name}: unknown video ids {unknown}')
        windowed = list(dict.fromkeys(video_ids))
    full = [v for v in order if v not in set(windowed)][:full_videos]
    plan: Plan = []
    for vid in windowed:
        info = probe(paths[vid], vid)
        length = max(1, int(round(window_s * info.fps_num / info.fps_den)))
        plan.append((info, spread_windows(info.nb_frames, windows, length, rng)))
    for vid in full:
        info = probe(paths[vid], vid)
        plan.append((info, [(0, info.nb_frames)]))
    return plan


# --------------------------------------------------------------------------- the run
class _HeadNmsExt:
    """mmcv's compiled ops module, except that nms of CUDA boxes runs detpost.head_nms (D27)."""

    def __init__(self, ext) -> None:
        self._ext = ext

    def __getattr__(self, name: str):
        return getattr(self._ext, name)

    def nms(self, boxes, scores, iou_threshold: float, offset: int):
        if not boxes.is_cuda or offset != 0:
            return self._ext.nms(boxes, scores, iou_threshold=iou_threshold, offset=offset)
        from .detpost import head_nms
        return head_nms(boxes, scores, [len(boxes)], iou_threshold)[0]


def official_nms(device) -> str:
    """Make the official path's mmcv nms runnable on `device`: where mmcv's own CUDA kernel runs, leave
    it ('mmcv'); where it cannot (its prebuilt ops stop at sm_86, so an H100 has none), route mmcv's
    nms of CUDA boxes to detpost.head_nms, the same computation bit for bit ('detpost.head_nms')."""
    import importlib
    import torch
    from .detpost import mmcv_nms_runs
    module = importlib.import_module('mmcv.ops.nms')
    if isinstance(module.ext_module, _HeadNmsExt):
        return 'detpost.head_nms'
    if mmcv_nms_runs(torch.device(device)):
        return 'mmcv'
    module.ext_module = _HeadNmsExt(module.ext_module)
    return 'detpost.head_nms'


def load_models(settings: Settings, device=None) -> ParityModels:
    """Official models and our engines (settings.backend) on `device` (default: the current CUDA
    device, cuda:0 under CUDA_VISIBLE_DEVICES), with TF32 off."""
    import torch
    from . import engines, env
    env.strict_fp32()
    device = torch.device('cuda', torch.cuda.current_device()) if device is None else torch.device(device)
    det_engine, pose_engine = engines.load_engines(settings, device)
    return ParityModels(engines.build_detector(settings, device), engines.build_pose_model(settings, device),
                        det_engine, pose_engine, device, official_nms(device))


def _provenance(settings: Settings, device) -> Dict[str, object]:
    from . import engines, env
    return dict(git_sha=env.git_sha(settings.repo_root), models=engines.model_provenance(settings),
                libraries=env.library_versions(include_trt=settings.backend == 'trt'),
                gpu_name=env.gpu_name(device.index), settings=settings.as_json())


def run_parity(settings: Settings, dataset: Dataset, num_videos: int = 20, windows: int = 3,
               window_s: float = 30.0, full_videos: int = 2, seed: int = 0,
               video_ids: Optional[Sequence[str]] = None, out_dir: Optional[Path] = None,
               models: Optional[ParityModels] = None, jitter: Sequence[str] = (),
               jitter_videos: Optional[int] = None) -> ParityReport:
    """Compare ours vs official on `num_videos` random videos x `windows` random windows of
    `window_s` seconds (numpy default_rng(seed)) plus `full_videos` whole videos; writes
    <out_dir>/parity_<backend>.json (after every window, `complete` once finished) and a short
    parity_<backend>.md when out_dir is given.

    `models` defaults to load_models(settings). Memory stays bounded: frames are decoded in one
    forward pass per video and compared one chunk (video.chunk_frames) at a time. `jitter` names
    JITTER_VARIANTS to measure on the first `jitter_videos` videos of the plan (default all); they
    go to report.jitter, where GATE_JITTER sets the gate's jitter-relative limits (gate_limits).
    """
    import torch
    from . import engines, env
    from .render import FrameCursor
    from .video import chunk_frames, chunk_from_frames
    from .worker import ChunkProcessor

    unknown = sorted(set(jitter) - set(JITTER_VARIANTS))
    if unknown:
        raise KeyError(f'unknown jitter variants {unknown}; known: {JITTER_VARIANTS}')
    plan = plan_windows(dataset, num_videos, windows, window_s, full_videos, seed, video_ids)
    models = load_models(settings) if models is None else models
    env.strict_fp32()
    env.check_environment(need_trt=settings.backend == 'trt')
    if not models.pose_model.test_cfg.get('flip_test', False):
        raise ValueError('the official pose model must run with flip_test=True')
    if (models.det_engine.backend, models.pose_engine.backend) != (settings.backend, settings.backend):
        raise ValueError(f'engines are {models.det_engine.backend}/{models.pose_engine.backend}, '
                         f'settings.backend is {settings.backend}')
    rule = dataset.frame_rule(dataset.extraction_rule)   # always per-frame (a video-level primary is a CPU derive)
    pose_meta = engines.pose_meta(settings)
    thresholds = ParityThresholds()
    report = ParityReport(backend=settings.backend, videos=[v.video_id for v, _ in plan], windows=[],
                          primary_rule=dataset.primary_rule, extraction_rule=rule.name,
                          provenance=dict(_provenance(settings, models.device), official_nms=models.official_nms),
                          notes=['kp_p95_px / kp_p99_px are upper bounds from a log histogram (bins 2.3 % wide)',
                                 f'SimCC step = scale_h / ({pose_meta.input_size[1]} x '
                                 f'{pose_meta.simcc_split_ratio:g}) px with scale = prep.crop_params(official box); '
                                 f'the gate counts confident keypoints moving > {thresholds.step_tol:g} steps '
                                 f'(<= {thresholds.max_kp_frac_gt_step:g}) and > {thresholds.flip_tol:g} steps '
                                 f'(any SimCC flip, <= {thresholds.max_kp_frac_flipped:g}), or up to '
                                 f'{thresholds.jitter_factor:g} x the {GATE_JITTER} jitter baseline; the px shares '
                                 'are informational'])
    accumulator = ParityAccumulator(settings, rule, pose_meta, thresholds)
    jitter_accumulators = {name: ParityAccumulator(settings, rule, pose_meta, thresholds)
                           for name in jitter}
    jitter_seconds = dict.fromkeys(jitter, 0.0)
    processor = ChunkProcessor(settings, models.det_engine, models.pose_engine, engines.det_meta(settings),
                               pose_meta, rule)
    stream = torch.cuda.Stream(models.device)  # TensorRT must not run on the legacy default stream
    started, total = time.time(), sum(len(w) for _, w in plan)
    for index, (video, video_windows) in enumerate(plan):
        size, frame_size = chunk_frames(video.width, video.height, settings), (video.width, video.height)
        variants = jitter if jitter_videos is None or index < jitter_videos else ()
        with FrameCursor(video.path) as cursor:
            for start, count in video_windows:
                if not env.is_strict_fp32():
                    raise RuntimeError('TF32 was re-enabled during the parity run')
                done = 0
                while done < count:
                    want = min(size, count - done)
                    frames = cursor.read(start + done, want)
                    if len(frames):
                        with torch.cuda.stream(stream):
                            dets, poses = processor.process(chunk_from_frames(video, start + done, frames))
                        refs = [official_frame(models.det_model, models.pose_model, f, settings) for f in frames]
                        for t, ref in enumerate(refs):
                            accumulator.add_frame(video.video_id, start + done + t, frame_size, ref,
                                                  our_frame(dets, poses, t))
                        for name in variants:
                            began = time.time()
                            others = _jitter_frames(name, models, frames, settings)
                            jitter_seconds[name] += time.time() - began
                            for t, (ref, other) in enumerate(zip(refs, others)):
                                jitter_accumulators[name].add_frame(
                                    video.video_id, start + done + t, frame_size, ref,
                                    as_ours(other, frame_size, settings.max_posed, rule))
                    done += len(frames)
                    if len(frames) < want:
                        report.notes.append(f'{video.video_id}: decoding ended at frame {start + done}, '
                                            f'window [{start}, {start + count}) requested')
                        break
                report.windows.append((video.video_id, start, done))
                accumulator.fill(report)
                report.jitter = {name: _jitter_summary(acc, jitter_seconds[name])
                                 for name, acc in jitter_accumulators.items()}
                report.seconds = time.time() - started
                if out_dir is not None:
                    write_report(report, out_dir)
                print(f'parity {len(report.windows)}/{total} {video.video_id} [{start}, {start + done}): '
                      f'{report.frames} frames, {report.frames / max(report.seconds, 1e-9):.1f} fps, mismatches '
                      f'pose/count/identity {report.pose_count_mismatches}/{report.count_set_mismatches}/'
                      f"{report.identity_mismatches}, confident >{report.step_tol:g} steps "
                      f"{report.kp_frac_gt_step['all']:.2e}, >{report.flip_tol:g} steps "
                      f"{report.kp_frac_flipped['all']:.2e} (>1px {report.kp_frac_gt_1px['all']:.2e})", flush=True)
    report.complete = True
    report.seconds = time.time() - started
    if out_dir is not None:
        write_report(report, out_dir)
    return report


def _jitter_summary(accumulator: ParityAccumulator, seconds: float) -> Dict[str, object]:
    """The ParityReport numbers of one jitter variant (without run metadata)."""
    numbers = ParityReport(backend='', videos=[], windows=[], seconds=seconds)
    accumulator.fill(numbers)
    metadata = {'backend', 'videos', 'windows', 'notes', 'primary_rule', 'extraction_rule', 'complete', 'provenance',
                'jitter'}
    return {k: v for k, v in dataclasses.asdict(numbers).items() if k not in metadata}


# --------------------------------------------------------------------------- verdict and files
Limit = Dict[str, object]   # a gate_limits entry


def gate_limits(report: ParityReport, thresholds: ParityThresholds = ParityThresholds()) -> Dict[str, Limit]:
    """The jitter-relative gate metrics (D17), keyed 'kp_frac_gt_step.<part>' and
    'kp_frac_flipped.<part>' for the GATED_PARTS, then 'score_max_diff'. Each entry is {'value': the
    report's number, 'fixed': the ParityThresholds limit, 'jitter': the same number of the report's
    GATE_JITTER baseline (None without one), 'effective': max(fixed, jitter_factor x jitter), or
    fixed without a baseline, 'passed': value <= effective (the score: value < effective)}.
    """
    baseline = report.jitter.get(GATE_JITTER) or {}

    def limit(value: float, fixed: float, noise: object, strict: bool = False) -> Limit:
        jitter = float(noise) if isinstance(noise, (int, float)) and math.isfinite(noise) else None
        effective = fixed if jitter is None else max(fixed, thresholds.jitter_factor * jitter)
        return dict(value=value, fixed=fixed, jitter=jitter, effective=effective,
                    passed=bool(value < effective if strict else value <= effective))

    limits = {}
    for metric, fixed in (('kp_frac_gt_step', thresholds.max_kp_frac_gt_step),
                          ('kp_frac_flipped', thresholds.max_kp_frac_flipped)):
        ours, noise = getattr(report, metric), baseline.get(metric) or {}
        for part in GATED_PARTS:
            limits[f'{metric}.{part}'] = limit(ours.get(part, float('nan')), fixed, noise.get(part))
    limits['score_max_diff'] = limit(report.score_max_diff, thresholds.max_kpt_score_diff,
                                     baseline.get('score_max_diff'), strict=True)
    return limits


def _limit_text(limit: Limit, thresholds: ParityThresholds) -> str:
    """A gate_limits entry's effective limit and where it comes from, for a failure reason."""
    if limit['jitter'] is None:
        return (f"{limit['fixed']:g}, fixed: the report has no {GATE_JITTER} jitter baseline; re-run parity with "
                f"'--jitter {GATE_JITTER}' to gate against the reference's own noise")
    scaled = f"{thresholds.jitter_factor:g} x {GATE_JITTER} jitter {limit['jitter']:.3g}"
    if limit['effective'] > limit['fixed']:
        return f"{limit['effective']:.3g} = {scaled}"
    return f"{limit['fixed']:g}, fixed, >= {scaled}"


def gate(report: ParityReport, thresholds: ParityThresholds = ParityThresholds()) -> Tuple[bool, List[str]]:
    """(passed, reasons): passes iff 0 non-borderline count/identity mismatches, box and detector
    score max diffs under their thresholds (NMS near-ties excluded), and every gate_limits metric
    within its effective limit: the keypoint score max over keypoints that moved <= step_tol SimCC
    steps and, overall and for each hand, the share of confident keypoints moving > step_tol SimCC
    steps (max_kp_frac_gt_step) and the share moving > flip_tol steps (any SimCC argmax flip,
    max_kp_frac_flipped: rare noise flips pass, a systematic one-bin shift does not). With a
    GATE_JITTER baseline in the report each of those may reach jitter_factor x the baseline's value
    of the same metric and part (D17: at 720p the official code's flips vs itself exceed the fixed
    limits); without one the fixed limits apply. The > 1 px / > 2 px shares and px percentiles are
    informational (a SimCC step is ~1 px or more for large people at 720p, so a px threshold is
    resolution-dependent).

    Also fails a report that is incomplete, compared nothing (or no confident keypoints of a gated
    part), or whose numbers or GATE_JITTER baseline were computed with other conf_thr / borderline /
    nms_tie_ulps / step_tol / flip_tol values than `thresholds`.
    """
    reasons = []
    if not report.complete:
        reasons.append('the parity run did not finish (report incomplete)')
    if report.frames_compared == 0 or report.persons_compared == 0:
        reasons.append('nothing was compared (no frame with an official pose set and a matching posed person)')
    names = ('conf_thr', 'borderline_thr', 'nms_tie_ulps', 'step_tol', 'flip_tol')
    wanted = (thresholds.conf_thr, thresholds.borderline, thresholds.nms_tie_ulps, thresholds.step_tol,
              thresholds.flip_tol)
    computed = [('report', tuple(getattr(report, name) for name in names))]
    if GATE_JITTER in report.jitter:
        computed.append((f'{GATE_JITTER} jitter baseline', tuple(report.jitter[GATE_JITTER].get(name)
                                                                 for name in names)))
    for what, used in computed:
        if used != wanted:
            reasons.append(f'{what} computed with conf_thr, borderline, nms_tie_ulps, step_tol, flip_tol = {used}; '
                           f'gate uses {wanted}')
    for name in ('pose_count_mismatches', 'count_set_mismatches', 'identity_mismatches'):
        if getattr(report, name):
            reasons.append(f'{name} = {getattr(report, name)} (not borderline)')
    if not report.box_max_diff_px < thresholds.max_box_diff_px:
        reasons.append(f'box max diff {report.box_max_diff_px:.4g} px >= {thresholds.max_box_diff_px}')
    limits = gate_limits(report, thresholds)
    score = limits['score_max_diff']
    if not score['passed']:
        reasons.append(f'score_max_diff {report.score_max_diff:.4g} (keypoints within {thresholds.step_tol:g} SimCC '
                       f'steps) >= {_limit_text(score, thresholds)}')
    if not report.det_score_max_diff < thresholds.max_det_score_diff:
        reasons.append(f'det_score_max_diff {report.det_score_max_diff:.4g} >= {thresholds.max_det_score_diff}')
    for part in GATED_PARTS:
        if not report.kp_confident_parts.get(part):
            reasons.append(f'no confident {part} keypoints were compared')
            continue
        moved, flipped = limits[f'kp_frac_gt_step.{part}'], limits[f'kp_frac_flipped.{part}']
        if not moved['passed']:
            reasons.append(f"{moved['value']:.3g} of confident {part} keypoints move > {thresholds.step_tol:g} SimCC "
                           f"steps (max {_limit_text(moved, thresholds)})")
        if not flipped['passed']:
            reasons.append(f"{flipped['value']:.3g} of confident {part} keypoints move > {thresholds.flip_tol:g} "
                           f"SimCC steps, any SimCC flip (max {_limit_text(flipped, thresholds)})")
    return not reasons, reasons


def summary_markdown(report: ParityReport, thresholds: ParityThresholds = ParityThresholds()) -> str:
    """Short human-readable summary of a report (and its gate verdict once complete)."""
    passed, reasons = gate(report, thresholds)
    limits = gate_limits(report, thresholds)
    verdict = ('PASSED' if passed else 'FAILED') if report.complete else 'INCOMPLETE (partial report)'
    steps = report.simcc_step_px
    step_text = (f"{steps['min']:.3g} / {steps['mean']:.3g} / {steps['max']:.3g} px (min / mean / max)"
                 if steps else 'none')
    baseline_text = (f"the {GATE_JITTER} baseline covers {report.jitter[GATE_JITTER].get('frames')} of the "
                     f'{report.frames} frames' if GATE_JITTER in report.jitter else
                     f"the report has no {GATE_JITTER} baseline, so the fixed limits apply (re-run parity with "
                     f"'--jitter {GATE_JITTER}' to gate against the reference's own noise)")
    lines = [f'# Parity vs official ({report.backend}): {verdict}', '',
             f'{report.frames} frames in {len(report.windows)} windows of {len(report.videos)} videos, '
             f'{report.frames_compared} with an official pose set, {report.persons_compared} people compared, '
             f'{report.seconds / 60:.1f} min. Compared with the extraction rule {report.extraction_rule} '
             f'(final primary rule: {report.primary_rule}).', '',
             f'- pose-set / count-set / identity mismatches: {report.pose_count_mismatches} / '
             f'{report.count_set_mismatches} / {report.identity_mismatches}; borderline excused: '
             f'{report.borderline_excused}; NMS near-ties excused: {report.nms_ties_excused}; '
             f'posed people without an official box: {report.persons_unmatched}',
             f'- box max diff {report.box_max_diff_px:.3g} px (< {thresholds.max_box_diff_px}); keypoint score '
             f"max diff {report.score_max_diff:.3g} (< {limits['score_max_diff']['effective']:.3g}, keypoints that "
             f'moved <= {report.step_tol:g} SimCC steps); detector score max diff {report.det_score_max_diff:.3g} '
             f'(< {thresholds.max_det_score_diff}); keypoint score max incl. keypoints that moved further '
             f'{report.score_max_diff_raw:.3g} (not gated)',
             f'- SimCC step (official box scale_h / (pose input height x split ratio)) of the compared people: '
             f'{step_text}; gated for all / lhand / rhand: confident keypoints moving > {report.step_tol:g} '
             f'steps and > {report.flip_tol:g} steps (any SimCC flip), limits below; the px columns are '
             'informational', '',
             f'Jitter-relative limits: the fixed limit, or {thresholds.jitter_factor:g} x the {GATE_JITTER} '
             f'jitter baseline (the official code vs itself) of the same metric and part when that is higher; '
             f'{baseline_text}.', '',
             f'| gated metric | ours | fixed limit | {GATE_JITTER} jitter | effective limit | verdict |',
             '|---|---|---|---|---|---|']
    metric_names = dict(kp_frac_gt_step=f'> {report.step_tol:g} steps', kp_frac_flipped=f'> {report.flip_tol:g} '
                        'steps (any flip)', score_max_diff='keypoint score max diff (strict <)')
    for key, limit in limits.items():
        metric, _, part = key.partition('.')
        jitter = '-' if limit['jitter'] is None else f"{limit['jitter']:.3g}"
        lines.append(f"| {metric_names[metric]}{', ' + part if part else ''} | {limit['value']:.3g} | "
                     f"{limit['fixed']:g} | {jitter} | {limit['effective']:.3g} | "
                     f"{'ok' if limit['passed'] else 'FAIL'} |")
    lines += ['', f'| part | confident (score > {report.conf_thr}) | > {report.flip_tol:g} steps | '
              f'> {report.step_tol:g} steps | max steps | > 1 px | > 2 px | p95 px | p99 px | max px |',
              '|---|---|---|---|---|---|---|---|---|---|']
    nan = float('nan')
    for part in PARTS:
        if part in report.kp_confident_parts:
            lines.append(f'| {part} | {report.kp_confident_parts[part]} | '
                         f'{report.kp_frac_flipped.get(part, nan):.2e} | '
                         f'{report.kp_frac_gt_step.get(part, nan):.2e} | '
                         f'{report.kp_max_diff_steps.get(part, nan):.3g} | '
                         f'{report.kp_frac_gt_1px.get(part, nan):.2e} | '
                         f'{report.kp_frac_gt_2px.get(part, nan):.2e} | {report.kp_p95_px.get(part, nan):.3g} | '
                         f'{report.kp_p99_px.get(part, nan):.3g} | {report.kp_max_diff_px.get(part, nan):.3g} |')
    for name, case in report.worst.items():
        lines.append(f'- worst {name}: ' + ', '.join(f'{k} {v}' for k, v in case.items()))
    if any(report.kp_steps_hist.values()):
        lines += ['', 'Confident keypoints by move in SimCC steps (0.5-1.5: one-step argmax flips): '
                  + ', '.join(f'{b} {c}' for b, c in report.kp_steps_hist.items())]
    if report.kp_outlier_joints:
        joints = sorted(report.kp_outlier_joints.items(), key=lambda kv: -kv[1])
        lines += ['', f'Confident keypoints > {report.step_tol:g} SimCC steps by official score: '
                  + ', '.join(f'{b} {c}' for b, c in report.kp_outlier_scores.items() if c),
                  'by keypoint index: ' + ', '.join(f'{k}: {c}' for k, c in joints)]
    if report.jitter:
        lines += ['', '## Jitter baselines vs the official path', '',
                  f'Their rows are not gated; {GATE_JITTER} sets the jitter-relative limits above.', '',
                  '| compared | people | pose/count/identity mismatches | box max px | kpt / det score max | '
                  f'> {report.flip_tol:g} steps all / lhand / rhand | '
                  f'> {report.step_tol:g} steps all / lhand / rhand | > 1 px all / lhand / rhand | > 2 px all | '
                  'max px |', '|---|---|---|---|---|---|---|---|---|---|']
        rows = [(f'ours ({report.backend})', dataclasses.asdict(report))] + list(report.jitter.items())
        for name, r in rows:
            gts, gt1, gt2 = r.get('kp_frac_gt_step', {}), r['kp_frac_gt_1px'], r['kp_frac_gt_2px']
            flips = r.get('kp_frac_flipped', {})
            lines.append(f"| {name} | {r['persons_compared']} | {r['pose_count_mismatches']}/"
                         f"{r['count_set_mismatches']}/{r['identity_mismatches']} | {r['box_max_diff_px']:.3g} | "
                         f"{r['score_max_diff']:.3g} / {r['det_score_max_diff']:.3g} | {flips.get('all', nan):.2e} / "
                         f"{flips.get('lhand', nan):.2e} / {flips.get('rhand', nan):.2e} | {gts.get('all', nan):.2e} / "
                         f"{gts.get('lhand', nan):.2e} / {gts.get('rhand', nan):.2e} | {gt1.get('all', nan):.2e} / "
                         f"{gt1.get('lhand', nan):.2e} / {gt1.get('rhand', nan):.2e} | {gt2.get('all', nan):.2e} | "
                         f"{r['kp_max_diff_px'].get('all', nan):.3g} |")
    if report.complete and reasons:
        lines += ['', 'Gate failures:'] + [f'- {r}' for r in reasons]
    if report.notes:
        lines += ['', 'Notes:'] + [f'- {n}' for n in report.notes]
    return '\n'.join(lines) + '\n'


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def write_report(report: ParityReport, out_dir: Path, thresholds: ParityThresholds = ParityThresholds()) -> Path:
    """Write <out_dir>/parity_<backend>.json (the report, plus the gate verdict and gate_limits once
    complete) and parity_<backend>.md, each atomically; returns the JSON path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = dataclasses.asdict(report)
    if report.complete:
        passed, reasons = gate(report, thresholds)
        data['gate'] = dict(passed=passed, reasons=reasons, thresholds=dataclasses.asdict(thresholds),
                            limits=gate_limits(report, thresholds))
    path = out_dir / f'parity_{report.backend}.json'
    _atomic_write(path, json.dumps(data, indent=1) + '\n')
    _atomic_write(path.with_suffix('.md'), summary_markdown(report, thresholds))
    return path


def load_report(path: Path) -> ParityReport:
    """The ParityReport of a write_report JSON, without its stored verdict: gate(load_report(path))
    re-gates a finished run offline (no GPU) with the current gate and thresholds."""
    data = json.loads(Path(path).read_text())
    names = {f.name for f in dataclasses.fields(ParityReport)}
    return ParityReport(**{k: v for k, v in data.items() if k in names})
