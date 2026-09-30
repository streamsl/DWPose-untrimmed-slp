"""Pose pre/post-processing (spec §4.2 posepost.py): what mmpose does around the network.

Per engine call: upload uint8 BGR crops, BGR->RGB + ImageNet mean/std on the GPU (mmengine
ImgDataPreprocessor order), append the mirrored batch (`inputs.flip(-1)`), run the engine once
on [crops; mirrors], merge the RAW logits as RTMCCHead.predict does (reverse x, remap
flip_indices, average), decode with get_simcc_maximum semantics (first argmax, score =
min(max_x, max_y), locations -1 where score <= 0, then / split ratio), and back-project with the
exact numpy expression of TopdownPoseEstimator.add_pred_to_datasample (float64, stored float32).
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch

from .engines import PoseMeta
from .settings import Settings
from .types import NUM_KEYPOINTS


def to_pose_input(crops: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """(B,H,W,3) uint8 BGR CUDA crops -> (B,3,H,W) float32 normalised RGB.

    mean/std are float32 (1,3,1,1) tensors on the crops' device.
    """
    x = crops.permute(0, 3, 1, 2)[:, [2, 1, 0]].float()
    return ((x - mean) / std).contiguous()


def flip_merge(sx: torch.Tensor, sy: torch.Tensor, n: int, flip_indices: Sequence[int]
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Logits of [orig(n); mirrored(n)] -> merged (n,K,Wx), (n,K,Wy), as RTMCCHead flip_test."""
    fx = sx[n:][:, list(flip_indices)].flip(-1)
    fy = sy[n:][:, list(flip_indices)]
    return (sx[:n] + fx) * 0.5, (sy[:n] + fy) * 0.5


def simcc_decode(sx: torch.Tensor, sy: torch.Tensor, split_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """SimCC argmax decode on the GPU -> locations (n,K,2) float32 in crop pixels, scores (n,K)."""
    max_x, ix = sx.max(-1)
    max_y, iy = sy.max(-1)
    scores = torch.minimum(max_x, max_y)
    locs = torch.stack([ix, iy], -1).float()
    locs[scores <= 0] = -1
    return locs / split_ratio, scores


def back_project(locs: np.ndarray, centers: np.ndarray, scales: np.ndarray,
                 input_size: Tuple[int, int]) -> np.ndarray:
    """Crop pixels -> original-frame pixels, the numpy expression mmpose uses (float64 math).

    locs (n,K,2) float32, centers/scales (n,2) float32 -> (n,K,2) float32.
    """
    # Same dtypes and op order as add_pred_to_datasample; broadcasting over instances does not
    # change element-wise IEEE results.
    scales, centers = scales[:, None], centers[:, None]
    return (locs / input_size * scales + centers - 0.5 * scales).astype(np.float32)


class PoseEstimator:
    """Crops -> keypoints in the saved encoding (x/W, y/H, raw score), flip test on."""

    def __init__(self, engine, meta: PoseMeta, settings: Settings) -> None:
        self._engine = engine
        self._meta = meta
        self._batch = settings.pose_batch
        device = engine.device
        self._mean = torch.tensor(meta.mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(meta.std, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    def __call__(self, crops: torch.Tensor, centers: np.ndarray, scales: np.ndarray,
                 frame_size: Tuple[int, int]) -> np.ndarray:
        """crops (M,384,288,3) uint8 CPU (pinned for async upload), centers/scales (M,2) float32
        from prep.crop_params, frame_size (W,H) -> (M,133,3) float32."""
        m = crops.shape[0]
        if m == 0:
            return np.zeros((0, NUM_KEYPOINTS, 3), np.float32)
        locs, scores = [], []
        for s in range(0, m, self._batch):
            x = to_pose_input(crops[s:s + self._batch].to(self._engine.device, non_blocking=True),
                              self._mean, self._std)
            n = x.shape[0]
            sx, sy = self._engine(torch.cat([x, x.flip(-1)]))
            sx, sy = flip_merge(sx, sy, n, self._meta.flip_indices)
            l, v = simcc_decode(sx, sy, self._meta.simcc_split_ratio)
            locs.append(l.cpu().numpy())
            scores.append(v.cpu().numpy())
        pixels = back_project(np.concatenate(locs), centers, scales, self._meta.input_size)
        width, height = frame_size
        out = np.empty((m, NUM_KEYPOINTS, 3), np.float32)
        out[..., 0] = pixels[..., 0] / np.float32(width)
        out[..., 1] = pixels[..., 1] / np.float32(height)
        out[..., 2] = np.concatenate(scores)
        return out
