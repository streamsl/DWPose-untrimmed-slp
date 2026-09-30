"""Detector post-processing (spec §4.2 detpost.py, D8, D10).

Reproduces `YOLOXHead.predict_by_feat(rescale=True)` restricted to the person class, batched on
the GPU, then splits the result into the three box sets of spec §4.1:
- candidates: person, score > cand_score_thr, after the head's class-aware NMS 0.65 (mmcv nms on
  the GPU, per frame, exactly as mmdet's batched_nms sees the person boxes);
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
        from mmcv.ops import nms
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
        pieces: List[torch.Tensor] = []
        kept: List[int] = []
        start = 0
        everything = torch.arange(len(cand_scores), device=cand_scores.device)
        for n in torch.bincount(frame, minlength=b).tolist():
            if n >= 2:
                _, keep = nms(cand_boxes[start:start + n], cand_scores[start:start + n], self._nms_iou)
                pieces.append(keep + start)
            else:
                pieces.append(everything[start:start + n])
            kept.append(len(pieces[-1]))
            start += n
        sel = torch.cat(pieces)
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
