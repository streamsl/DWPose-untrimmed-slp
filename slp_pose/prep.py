"""Exact CPU image preparation (spec D10): detector letterbox and pose crops.

Both call the same mmcv / mmpose functions as the official pipelines, so the bytes are identical
by construction (the test suite checks them against the pipelines):
- letterbox = mmdet `Resize(scale=(640, 640), keep_ratio=True)` (mmcv.imrescale, bilinear) +
  `Pad(pad_to_square=True, pad_val=114)` on the bottom/right;
- crop = mmpose `GetBBoxCenterScale(padding=1.25)` + `TopdownAffine(input_size=(288, 384))`
  (bbox_xyxy2cs, _fix_aspect_ratio, get_warp_matrix, cv2.warpAffine INTER_LINEAR, black border).
"""
from __future__ import annotations

from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mmcv
import numpy as np
from mmpose.datasets.transforms import TopdownAffine
from mmpose.structures.bbox import bbox_xyxy2cs, get_warp_matrix

DET_SIZE = 640
DET_PAD_VALUE = 114
POSE_INPUT_SIZE = (288, 384)  # (w, h)
POSE_BBOX_PADDING = 1.25


@dataclass(frozen=True)
class LetterboxGeometry:
    """Where an original frame lands inside the square detector input."""

    width: int                          # original frame size
    height: int
    new_w: int                          # resized image inside the letterbox (top-left aligned)
    new_h: int
    size: int                           # letterbox side (640)
    scale_factor: Tuple[float, float]   # (new_w / width, new_h / height), mmdet `scale_factor`


def letterbox_geometry(width: int, height: int, size: int = DET_SIZE) -> LetterboxGeometry:
    """Geometry of mmdet's keep-ratio Resize to (size, size) followed by the square Pad."""
    new_w, new_h = mmcv.rescale_size((width, height), (size, size))
    if max(new_w, new_h) != size:
        raise ValueError(f'unexpected rescale of {width}x{height} to {new_w}x{new_h}')
    return LetterboxGeometry(width, height, new_w, new_h, size, (new_w / width, new_h / height))


def letterbox(frame: np.ndarray, geom: LetterboxGeometry, out: np.ndarray) -> None:
    """Write the detector input of one BGR frame into `out` ((size,size,3) uint8)."""
    resized = mmcv.imrescale(frame, (geom.size, geom.size), interpolation='bilinear')
    if resized.shape[:2] != (geom.new_h, geom.new_w):
        raise ValueError(f'frame {frame.shape} does not match geometry {geom}')
    out[:geom.new_h, :geom.new_w] = resized
    out[geom.new_h:] = DET_PAD_VALUE
    out[:geom.new_h, geom.new_w:] = DET_PAD_VALUE


def crop_params(boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(n,4) float32 xyxy boxes -> centers (n,2), scales (n,2) float32 (mmpose input_center/scale)."""
    boxes = np.ascontiguousarray(boxes, dtype=np.float32).reshape(-1, 4)
    centers, scales = bbox_xyxy2cs(boxes, padding=POSE_BBOX_PADDING)
    w, h = POSE_INPUT_SIZE
    scales = TopdownAffine._fix_aspect_ratio(scales, aspect_ratio=w / h)
    return centers.reshape(-1, 2), scales.reshape(-1, 2)


def crop(frame: np.ndarray, center: np.ndarray, scale: np.ndarray, out: np.ndarray) -> None:
    """Write the (384,288,3) uint8 BGR pose crop of `frame` for one center/scale into `out`."""
    w, h = POSE_INPUT_SIZE
    mat = get_warp_matrix(center, scale, 0., output_size=(w, h))
    out[...] = cv2.warpAffine(frame, mat, (int(w), int(h)), flags=cv2.INTER_LINEAR)


def crop_many(frames: np.ndarray, frame_idx: np.ndarray, centers: np.ndarray, scales: np.ndarray,
              out: np.ndarray, pool: Optional[Executor] = None) -> None:
    """Crop M boxes: out[i] = crop(frames[frame_idx[i]], centers[i], scales[i]).

    `out` is (M,384,288,3) uint8 (typically a numpy view of a pinned tensor). With a thread pool
    the crops run in parallel; set cv2.setNumThreads(1) in the process to avoid oversubscription.
    """
    def one(i: int) -> None:
        crop(frames[frame_idx[i]], centers[i], scales[i], out[i])

    if pool is None:
        for i in range(len(frame_idx)):
            one(i)
    else:
        list(pool.map(one, range(len(frame_idx))))
