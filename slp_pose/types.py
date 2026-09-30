"""Shared data types passed between the decode, GPU, selection and record stages.

Conventions used everywhere:
- Frame indices are 0-based positions in the decoded video (every frame, native fps).
- Boxes are xyxy float32 in ORIGINAL-frame pixels.
- Keypoints use the saved encoding (spec §4.1): (x/W, y/H, raw SimCC score) float32 in
  COCO-WholeBody-133 order.
- Ragged per-frame lists are CSR: `offsets` has length F+1, rows of frame t are
  `offsets[t]:offsets[t+1]`, offsets[0] == 0.

This module imports numpy only (torch appears in annotations), so CPU tools can use it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    import torch

SCHEMA_VERSION = 1
NUM_KEYPOINTS = 133

# det_flags bits (spec §4.1)
POSE_SET = 1   # survives the pose-set rule (score > 0.3, mmpose legacy NMS 0.3)
COUNT_SET = 2  # survives the count-set rule (score > 0.3, torchvision NMS 0.45)
POSED = 4      # was posed (at most K per frame, always a subset of POSE_SET)


@dataclass(frozen=True)
class VideoInfo:
    """What ffprobe says about one video (spec §4.1: T = nb_frames)."""

    path: Path
    video_id: str
    width: int
    height: int
    fps_num: int     # exact rational frame rate num/den (e.g. 25/1, 30000/1001)
    fps_den: int
    nb_frames: int
    codec: str
    size_bytes: int

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den

    @property
    def duration_s(self) -> float:
        """T * den / num, the value written to video_meta.csv."""
        return self.nb_frames * self.fps_den / self.fps_num


@dataclass
class Chunk:
    """A run of consecutive decoded frames of one video, in pooled (reused) buffers.

    `frames` and `letterbox` stay valid until `release()`; the reader reuses the buffers after
    that. A chunk with `error` set is terminal for its video (`last` is True, no frames) and the
    video must be failed; chunks of that video yielded before it must be discarded.
    `num_frames == 0` also happens for a clean end exactly on a chunk boundary when frame-count
    mismatches are allowed, so consumers must accept empty chunks.
    """

    video: VideoInfo
    start: int                   # video frame index of frames[0]
    frames: np.ndarray           # (F,H,W,3) uint8 BGR originals
    letterbox: 'torch.Tensor'    # (F,640,640,3) uint8, pinned CPU; mmdet Resize+Pad bytes
    last: bool = False           # final chunk of the video
    error: Optional[str] = None
    _release: Optional[Callable[[], None]] = field(default=None, repr=False)

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])

    def release(self) -> None:
        """Return the buffers to the reader's pool (idempotent)."""
        if self._release is not None:
            release, self._release = self._release, None
            release()


def _check_offsets(offsets: np.ndarray, n: int, what: str) -> None:
    if offsets.dtype != np.int64 or offsets.ndim != 1 or len(offsets) < 1:
        raise ValueError(f'{what}: offsets must be 1-D int64 with length F+1')
    if offsets[0] != 0 or offsets[-1] != n or np.any(np.diff(offsets) < 0):
        raise ValueError(f'{what}: offsets must start at 0, be non-decreasing and end at {n}')


@dataclass
class ChunkDets:
    """Per-frame person candidates of a chunk (CSR), spec §4.1 `det_*`.

    Rows of one frame are in descending score order (the head NMS output order). Candidates are
    person detections with score > cand_score_thr after the head's class-aware NMS 0.65.
    """

    offsets: np.ndarray  # (F+1,) int64
    boxes: np.ndarray    # (N,4) float32 xyxy, original pixels
    scores: np.ndarray   # (N,) float32 detector score (sigmoid(cls) * sigmoid(obj))
    flags: np.ndarray    # (N,) uint8, POSE_SET | COUNT_SET | POSED bits

    def __post_init__(self) -> None:
        n = len(self.scores)
        _check_offsets(self.offsets, n, 'ChunkDets')
        if self.boxes.shape != (n, 4) or self.boxes.dtype != np.float32:
            raise ValueError('ChunkDets: boxes must be (N,4) float32')
        if self.scores.dtype != np.float32 or self.flags.shape != (n,) or self.flags.dtype != np.uint8:
            raise ValueError('ChunkDets: scores (N,) float32 and flags (N,) uint8 required')

    @property
    def num_frames(self) -> int:
        return len(self.offsets) - 1

    def rows(self, t: int) -> slice:
        return slice(int(self.offsets[t]), int(self.offsets[t + 1]))

    def frame_of_row(self) -> np.ndarray:
        """(N,) int64 frame index of every row."""
        return np.repeat(np.arange(self.num_frames, dtype=np.int64), np.diff(self.offsets))

    def count_per_frame(self, bit: int) -> np.ndarray:
        """(F,) int64 number of rows with `bit` set, per frame."""
        has = (self.flags & bit) != 0
        return np.bincount(self.frame_of_row()[has], minlength=self.num_frames).astype(np.int64)

    def num_persons(self) -> np.ndarray:
        """(F,) uint8 size of the count set per frame (spec §4.1 `num_persons`)."""
        counts = self.count_per_frame(COUNT_SET)
        if counts.size and counts.max() > 255:
            raise ValueError('more than 255 people in the count set of one frame')
        return counts.astype(np.uint8)

    @staticmethod
    def empty(num_frames: int) -> 'ChunkDets':
        return ChunkDets(np.zeros(num_frames + 1, np.int64), np.zeros((0, 4), np.float32),
                         np.zeros(0, np.float32), np.zeros(0, np.uint8))

    @staticmethod
    def concat(parts: Sequence['ChunkDets']) -> 'ChunkDets':
        """Concatenate consecutive frame ranges."""
        if not parts:
            return ChunkDets.empty(0)
        offsets = [np.zeros(1, np.int64)]
        base = 0
        for p in parts:
            offsets.append(p.offsets[1:] + base)
            base += len(p.scores)
        return ChunkDets(np.concatenate(offsets), np.concatenate([p.boxes for p in parts]),
                         np.concatenate([p.scores for p in parts]),
                         np.concatenate([p.flags for p in parts]))


@dataclass
class ChunkPoses:
    """Keypoints of the posed people of a chunk (CSR), spec §4.1 `kpt_*`.

    Invariant: the kpt rows of frame t are exactly the POSED det rows of frame t, in ascending
    det-row order (i.e. descending detector score). `primary[t]` is the position of the primary
    signer inside that list, or -1 when the frame has no posed person.
    """

    offsets: np.ndarray    # (F+1,) int64
    det_index: np.ndarray  # (M,) int32, chunk-local row index into the matching ChunkDets
    kpts: np.ndarray       # (M,133,3) float32 saved encoding
    primary: np.ndarray    # (F,) int8

    def __post_init__(self) -> None:
        m = len(self.det_index)
        _check_offsets(self.offsets, m, 'ChunkPoses')
        if self.det_index.dtype != np.int32 or self.kpts.shape != (m, NUM_KEYPOINTS, 3) \
                or self.kpts.dtype != np.float32:
            raise ValueError('ChunkPoses: det_index (M,) int32 and kpts (M,133,3) float32 required')
        if self.primary.shape != (self.num_frames,) or self.primary.dtype != np.int8:
            raise ValueError('ChunkPoses: primary must be (F,) int8')
        counts = np.diff(self.offsets)
        if np.any(self.primary >= counts) or np.any((self.primary < 0) != (counts == 0)):
            raise ValueError('ChunkPoses: primary must index the frame list, -1 only for empty frames')

    @property
    def num_frames(self) -> int:
        return len(self.offsets) - 1

    def primary_poses(self) -> np.ndarray:
        """(F,133,3) float32 rows of `poses/<vid>.npy`: the primary's keypoints, zeros if none."""
        out = np.zeros((self.num_frames, NUM_KEYPOINTS, 3), np.float32)
        has = self.primary >= 0
        out[has] = self.kpts[self.offsets[:-1][has] + self.primary[has]]
        return out
