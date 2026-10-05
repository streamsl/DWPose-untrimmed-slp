"""Detector post-processing (spec §4.2 detpost.py, D8, D10).

Reproduces `YOLOXHead.predict_by_feat(rescale=True)` restricted to the person class, batched on
the GPU, then splits the result into the three box sets of spec §4.1:
- candidates: person, score > cand_score_thr, after the head's class-aware NMS 0.65 (mmcv's CUDA nms,
  per frame, exactly as mmdet's batched_nms sees the person boxes; on a GPU that mmcv's prebuilt
  kernels do not cover, e.g. an H100, the same computation bit for bit in torch, `head_nms`, D27);
- pose set (POSE_SET): candidates with score > pose_score_thr after mmpose's numpy `nms`
  (legacy +1 IoU), the DWPose reference rule;
- count set (COUNT_SET): candidates with score > count_score_thr after torchvision `nms` on the CPU
  (standard IoU), the SignVerse rule. CPU so that `derive` can recompute it bit-for-bit.
Running the head NMS on person boxes above cand_score_thr only gives the same person survivors:
the NMS is class-aware (other classes never suppress a person), and in greedy NMS a box can only
suppress boxes ranked below it, so dropping lower-score boxes changes nothing above them.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .engines import DetMeta
from .settings import Settings
from .types import COUNT_SET, POSE_SET, ChunkDets


def to_det_input(letterbox: torch.Tensor) -> torch.Tensor:
    """(B,S,S,3) uint8 CUDA letterbox -> (B,3,S,S) float32, as mmdet's DetDataPreprocessor."""
    return letterbox.permute(0, 3, 1, 2).float().contiguous()


def decode_boxes(priors: torch.Tensor, bbox_preds: torch.Tensor) -> torch.Tensor:
    """YOLOXHead._bbox_decode with the identical op order: (N,4) priors, (B,N,4) preds -> (B,N,4) xyxy."""
    xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
    whs = bbox_preds[..., 2:].exp() * priors[:, 2:]
    tl_x = (xys[..., 0] - whs[..., 0] / 2)
    tl_y = (xys[..., 1] - whs[..., 1] / 2)
    br_x = (xys[..., 0] + whs[..., 0] / 2)
    br_y = (xys[..., 1] + whs[..., 1] / 2)
    return torch.stack([tl_x, tl_y, br_x, br_y], -1)


_FLT_TIE = 2.0 ** 128 - 2.0 ** 103   # halfway between FLT_MAX and 2^128: sums from here round to inf
_NMS_CELLS = 1 << 22                 # box pairs per overlap pass (~0.35 GB of temporaries at most)


def fma32(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """float32 fma(x, y, z) with one rounding, from float64: x*y is exact there, and where the float64
    sum lands exactly halfway between two float32 values (or at the overflow tie _FLT_TIE), its exact
    error (TwoSum) picks the side."""
    p, zz = x.double() * y.double(), z.double()
    s = p + zz
    t = s - zz
    err = (p - t) + (zz - (s - t))                  # p + zz == s + err exactly
    r = s.float()
    up = s > r.double()
    other = torch.nextafter(r, torch.where(up, r.new_tensor(float('inf')), r.new_tensor(float('-inf'))))
    halfway = torch.where(r.isinf(), s.abs() == _FLT_TIE, (r.double() + other.double()) * 0.5 == s)
    return torch.where(halfway & (err != 0) & ((err > 0) == up) & s.isfinite(), other, r)


def _suppressed(boxes: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor, iou_thr: float) -> np.ndarray:
    """(F, R, C) bool on the CPU: box cols[f, j] overlaps box rows[f, i] above iou_thr by mmcv's float32
    test, rows being the higher-ranked boxes: interS > thr * (fma(width_b, height_b, Sa) - interS)."""
    ax1, ay1, ax2, ay2 = boxes[rows].unbind(-1)
    bx1, by1, bx2, by2 = boxes[cols].unbind(-1)
    w = (torch.minimum(ax2[:, :, None], bx2[:, None]) - torch.maximum(ax1[:, :, None], bx1[:, None])).clamp(min=0)
    h = (torch.minimum(ay2[:, :, None], by2[:, None]) - torch.maximum(ay1[:, :, None], by1[:, None])).clamp(min=0)
    inter = w * h
    bw, bh = bx2 - bx1, by2 - by1
    both = fma32(bw[:, None].expand_as(inter), bh[:, None].expand_as(inter),
                 ((ax2 - ax1) * (ay2 - ay1))[:, :, None].expand_as(inter))
    return (inter > boxes.new_tensor(iou_thr) * (both - inter)).cpu().numpy()


def head_nms(boxes: torch.Tensor, scores: torch.Tensor, counts: Sequence[int],
             iou_thr: float) -> Tuple[torch.Tensor, List[int]]:
    """(rows kept, kept per frame) by mmcv.ops.nms(boxes_f, scores_f, iou_thr) of every frame f with
    >= 2 rows (a frame with fewer keeps them): rows grouped by frame (`counts` per frame, in order);
    the kept rows frame after frame, each frame's by descending score, as mmcv returns them.

    mmcv 2.1.0's CUDA nms recomputed in plain torch, so that no compiled mmcv code is needed for the GPU
    (its wheel has none for compute capability 9.0): torch's descending sort of the frame's scores (the
    same call, so equal scores keep the same order), the same float32 test of a higher-ranked box a
    against a lower-ranked box b, interS > thr * (Sa + Sb - interS), with Sa + Sb fused into
    fma(width_b, height_b, Sa) as nvcc compiled it (measured: the one contraction that reproduces mmcv
    on near-ties), and the same greedy pass down the ranking. Frames of similar size share an overlap
    pass of at most _NMS_CELLS box pairs, and a frame too big for one pass is done in row blocks, so a
    crowded frame needs no more GPU memory than that."""
    device = boxes.device
    bounds = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)])
    multi = [f for f, n in enumerate(counts) if n >= 2]
    ranked = [scores[bounds[f]:bounds[f + 1]].sort(0, descending=True)[1] for f in multi]
    orders = dict(zip(multi, np.split(torch.cat(ranked).cpu().numpy(), np.cumsum([counts[f] for f in multi])[:-1])
                      if multi else []))
    keep: Dict[int, np.ndarray] = {}
    groups: List[List[int]] = []
    for f in sorted(multi, key=lambda f: counts[f]):
        if groups and (len(groups[-1]) + 1) * counts[f] ** 2 <= _NMS_CELLS:
            groups[-1].append(f)
        else:
            groups.append([f])
    for group in groups:
        width = counts[group[-1]]
        index = np.zeros((len(group), width), np.int64)
        for i, f in enumerate(group):
            index[i, :counts[f]] = orders[f] + bounds[f]
        index_t = torch.from_numpy(index).to(device)
        step = max(1, _NMS_CELLS // (len(group) * width))
        over = np.concatenate([_suppressed(boxes, index_t[:, r:r + step], index_t, iou_thr)
                               for r in range(0, width, step)], axis=1)
        for i, f in enumerate(group):
            alive, kept, r = np.ones(counts[f], bool), [], 0
            while True:                                  # greedy: from each kept box to the next survivor
                kept.append(r)
                alive &= ~over[i, r, :counts[f]]
                later = np.flatnonzero(alive[r + 1:])
                if not len(later):
                    break
                r += 1 + later[0]
            keep[f] = orders[f][kept] + bounds[f]
    rows = [keep[f] if f in keep else np.arange(bounds[f], bounds[f + 1]) for f in range(len(counts))]
    return (torch.from_numpy(np.concatenate(rows) if rows else np.zeros(0, np.int64)).to(device),
            [len(r) for r in rows])


_MMCV_NMS_RUNS: Dict[int, bool] = {}


def mmcv_nms_runs(device: torch.device) -> bool:
    """Whether mmcv's compiled CUDA nms runs on `device` (one probe per device index): its prebuilt ops
    stop at sm_86 with no PTX, so on an H100 (compute capability 9.0) it raises 'no kernel image'."""
    if device.index not in _MMCV_NMS_RUNS:
        from mmcv.ops import nms
        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.]], device=device)
        try:
            nms(boxes, torch.tensor([0.9, 0.8], device=device), 0.5)
            _MMCV_NMS_RUNS[device.index] = True
        except RuntimeError:   # 'no kernel image is available' (not sticky) or a CPU-only mmcv build
            _MMCV_NMS_RUNS[device.index] = False
    return _MMCV_NMS_RUNS[device.index]


def mmcv_head_nms(boxes: torch.Tensor, scores: torch.Tensor, counts: Sequence[int],
                  iou_thr: float) -> Tuple[torch.Tensor, List[int]]:
    """head_nms with mmcv's own CUDA kernel, one call per frame (identical results; about 3x faster on
    crowded frames than head_nms on GPUs with slow float64, so it is used wherever it runs)."""
    from mmcv.ops import nms
    pieces, start = [], 0
    for n in counts:
        pieces.append(nms(boxes[start:start + n], scores[start:start + n], iou_thr)[1] + start if n >= 2
                      else torch.arange(start, start + n, device=boxes.device))
        start += n
    sel = torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.long, device=boxes.device)
    return sel, [len(p) for p in pieces]


def pose_set_mask(boxes: np.ndarray, scores: np.ndarray, score_thr: float, nms_thr: float) -> np.ndarray:
    """(n,) bool: rows kept by the DWPose rule (score > thr, then mmpose.evaluation nms)."""
    from mmpose.evaluation.functional import nms
    mask = np.zeros(len(scores), bool)
    cand = np.flatnonzero(scores > score_thr)
    if len(cand):
        dets = np.concatenate([boxes[cand], scores[cand, None]], axis=1)
        mask[cand[nms(dets, nms_thr)]] = True
    return mask


def count_set_mask(boxes: np.ndarray, scores: np.ndarray, score_thr: float, nms_thr: float) -> np.ndarray:
    """(n,) bool: rows kept by the count rule (score > thr, then torchvision nms on the CPU)."""
    import torchvision
    mask = np.zeros(len(scores), bool)
    cand = np.flatnonzero(scores > score_thr)
    if len(cand):
        keep = torchvision.ops.nms(torch.from_numpy(boxes[cand]), torch.from_numpy(scores[cand]), nms_thr)
        mask[cand[keep.numpy()]] = True
    return mask


def frame_flags(boxes: np.ndarray, scores: np.ndarray, settings: Settings) -> np.ndarray:
    """(n,) uint8 POSE_SET | COUNT_SET bits for the candidates of one frame."""
    pose = pose_set_mask(boxes, scores, settings.pose_score_thr, settings.pose_nms_thr)
    count = count_set_mask(boxes, scores, settings.count_score_thr, settings.count_nms_thr)
    return (pose * POSE_SET | count * COUNT_SET).astype(np.uint8)


class DetPostprocessor:
    """Raw YOLOX maps of B frames -> ChunkDets (candidates + POSE_SET/COUNT_SET flags)."""

    def __init__(self, meta: DetMeta, settings: Settings) -> None:
        from mmdet.models.task_modules.prior_generators import MlvlPointGenerator
        if settings.cand_score_thr < meta.score_thr:
            raise ValueError('cand_score_thr is below the head score threshold')
        self._settings = settings
        self._nms_iou = meta.nms_iou
        self._prior_generator = MlvlPointGenerator(meta.strides, offset=0)
        self._priors: Dict[Tuple, torch.Tensor] = {}

    def _priors_for(self, cls_maps: Sequence[torch.Tensor]) -> torch.Tensor:
        sizes = [m.shape[2:] for m in cls_maps]
        key = (tuple(tuple(s) for s in sizes), str(cls_maps[0].device))
        if key not in self._priors:
            self._priors[key] = torch.cat(self._prior_generator.grid_priors(
                sizes, dtype=cls_maps[0].dtype, device=cls_maps[0].device, with_stride=True))
        return self._priors[key]

    def __call__(self, maps: Sequence[torch.Tensor], scale_factor: Tuple[float, float]) -> ChunkDets:
        """maps: the 9 engine outputs for B frames; scale_factor: mmdet (w_scale, h_scale)."""
        cls_maps, box_maps, obj_maps = maps[0:3], maps[3:6], maps[6:9]
        b, c = cls_maps[0].shape[:2]
        cls = torch.cat([m.permute(0, 2, 3, 1).reshape(b, -1, c) for m in cls_maps], 1).sigmoid()
        box = torch.cat([m.permute(0, 2, 3, 1).reshape(b, -1, 4) for m in box_maps], 1)
        obj = torch.cat([m.permute(0, 2, 3, 1).reshape(b, -1) for m in obj_maps], 1).sigmoid()
        boxes = decode_boxes(self._priors_for(cls_maps), box)
        max_scores, labels = torch.max(cls, -1)
        scores = max_scores * obj
        frame, prior = ((labels == 0) & (scores > self._settings.cand_score_thr)).nonzero(as_tuple=True)
        cand_boxes = boxes[frame, prior] / boxes.new_tensor(scale_factor).repeat((1, 2))
        cand_scores = scores[frame, prior]

        # Head NMS per frame; nonzero() keeps frames contiguous and in order.
        nms = mmcv_head_nms if cand_boxes.is_cuda and mmcv_nms_runs(cand_boxes.device) else head_nms
        sel, kept = nms(cand_boxes, cand_scores, torch.bincount(frame, minlength=b).tolist(), self._nms_iou)
        out_boxes = cand_boxes[sel].cpu().numpy()
        out_scores = cand_scores[sel].cpu().numpy()
        offsets = np.zeros(b + 1, np.int64)
        np.cumsum(kept, out=offsets[1:])
        flags = np.zeros(len(out_scores), np.uint8)
        for t in range(b):
            rows = slice(offsets[t], offsets[t + 1])
            if rows.stop > rows.start:
                flags[rows] = frame_flags(out_boxes[rows], out_scores[rows], self._settings)
        return ChunkDets(offsets, out_boxes, out_scores, flags)


def detect(engine, post: DetPostprocessor, letterbox: torch.Tensor, scale_factor: Tuple[float, float],
           batch: int) -> ChunkDets:
    """Run the detector over a chunk's pinned (F,640,640,3) uint8 letterbox, `batch` frames per call."""
    parts = []
    for s in range(0, letterbox.shape[0], batch):
        x = letterbox[s:s + batch].to(engine.device, non_blocking=True)
        parts.append(post(engine(to_det_input(x)), scale_factor))
    return ChunkDets.concat(parts) if parts else ChunkDets.empty(0)
