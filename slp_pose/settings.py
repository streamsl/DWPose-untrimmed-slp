"""Run configuration: thresholds, batching, precision and model paths (datasets: slp_pose.datasets).

Everything that changes GPU output is listed by `Settings.extraction_fields` (plus the dataset's
extraction rule when it changes which people are posed, record.extraction_hash); everything a CPU
`derive` may change is listed by `Settings.derivation_fields` (spec §4.4).

Settings objects are pickled to the GPU workers, and a respawned worker unpickles Settings made by
the parent's (possibly older) code: fields are only ever added, each with a plain default (a
missing field then reads the class default), and never renamed or removed.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from . import paths
from .paths import REPO_ROOT

DET_CONFIG = 'yolox_l_8xb8-300e_coco.py'           # in paths.CONFIG_DIR
POSE_CONFIG = 'dwpose-l_384x288.py'
DET_CHECKPOINT = 'yolox/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'   # in the models dir
POSE_CHECKPOINT = 'dwpose/dw-ll_ucoco_384.pth'


@dataclass(frozen=True)
class PrecisionProfile:
    """One numeric tier for both the TRT build and the torch backend."""

    name: str
    tf32: bool      # allow TF32 tensor-core math (TRT BuilderFlag.TF32, torch allow_tf32)
    lossless: bool  # measured and approved as lossless vs strict fp32 (spec D5/D7)


# Only the lossless tier exists for now; lossy tiers are added here once measured and approved.
PRECISION_PROFILES: Dict[str, PrecisionProfile] = {
    'fp32': PrecisionProfile('fp32', tf32=False, lossless=True),
}
BACKENDS = ('trt', 'torch')


@dataclass(frozen=True)
class Settings:
    """Frozen run configuration. Nothing is hardcoded: configs ship with the package, checkpoints,
    ONNX and engines live in `models_dir` (paths.models_dir() unless given)."""

    # Detection thresholds (spec D8). The head's own score_thr 0.01 / NMS 0.65 come from the
    # detector config (engines.DetMeta), not from here.
    cand_score_thr: float = 0.1    # stored person candidates: score > this
    pose_score_thr: float = 0.3    # pose set: score > this, then mmpose legacy NMS
    pose_nms_thr: float = 0.3
    count_score_thr: float = 0.3   # count set: score > this, then torchvision NMS
    count_nms_thr: float = 0.45
    # Pose
    max_posed: int = 3             # K: people posed per frame (spec D4)
    # Batching (spec D9): chosen at the speed plateau, not to fill memory.
    det_batch: int = 64            # frames per detector call
    pose_batch: int = 128          # crops per pose call; the call also carries their 128 mirrors
    chunk_bytes: int = 512 << 20   # decode chunk budget: originals + 640x640 letterboxes
    max_chunk_frames: int = 256
    queue_depth: int = 2           # decoded chunks waiting ahead of the consumer
    decode_threads: int = 4        # FFmpeg threads inside cv2.VideoCapture
    crop_threads: int = 4
    # Numerics
    precision: str = 'fp32'
    backend: str = 'trt'
    # TensorRT optimisation profiles (min, opt, max batch); opt = the steady-state batch.
    det_profile: Tuple[int, int, int] = (1, 64, 64)
    pose_profile: Tuple[int, int, int] = (1, 256, 256)
    repo_root: Path = REPO_ROOT          # the source checkout: git sha in the provenance
    # None (only in Settings pickled by older code): <repo_root>/models, as that code used.
    models_root: Optional[Path] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'models_root', paths.models_dir(self.models_root))
        if self.precision not in PRECISION_PROFILES:
            raise ValueError(f'unknown precision {self.precision!r}; known: {sorted(PRECISION_PROFILES)}')
        if self.backend not in BACKENDS:
            raise ValueError(f'unknown backend {self.backend!r}; known: {BACKENDS}')
        if not 1 <= self.max_posed <= 127:
            raise ValueError('max_posed must be in 1..127 (kpt_primary is int8)')
        if not self.cand_score_thr <= min(self.pose_score_thr, self.count_score_thr):
            raise ValueError('pose/count score thresholds must not be below the candidate threshold')
        if not (self.det_profile[0] <= self.det_batch <= self.det_profile[2]):
            raise ValueError('det_batch must lie inside det_profile')
        if not (self.pose_profile[0] <= 2 * self.pose_batch <= self.pose_profile[2]):
            raise ValueError('2 * pose_batch (crops + mirrors) must lie inside pose_profile')

    def replace(self, **changes) -> 'Settings':
        """Return a copy with `changes` applied (validated again)."""
        return dataclasses.replace(self, **changes)

    @property
    def profile(self) -> PrecisionProfile:
        return PRECISION_PROFILES[self.precision]

    def as_json(self) -> Dict[str, object]:
        """dataclasses.asdict with paths as str (provenance and reports)."""
        return {k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(self).items()}

    # ------------------------------------------------------------------ paths
    @property
    def models_dir(self) -> Path:
        return self.repo_root / 'models' if self.models_root is None else Path(self.models_root)

    @property
    def det_config(self) -> Path:
        return paths.CONFIG_DIR / DET_CONFIG

    @property
    def det_checkpoint(self) -> Path:
        return self.models_dir / DET_CHECKPOINT

    @property
    def pose_config(self) -> Path:
        return paths.CONFIG_DIR / POSE_CONFIG

    @property
    def pose_checkpoint(self) -> Path:
        return self.models_dir / POSE_CHECKPOINT

    @property
    def onnx_dir(self) -> Path:
        return self.models_dir / 'onnx'

    @property
    def engine_dir(self) -> Path:
        return self.models_dir / 'engines'

    # ----------------------------------------------------------------- hashes
    def extraction_fields(self) -> Dict[str, object]:
        """Settings that change GPU output (hashed with engine/checkpoint sha256 + schema version)."""
        return dict(cand_score_thr=self.cand_score_thr, pose_score_thr=self.pose_score_thr,
                    pose_nms_thr=self.pose_nms_thr, max_posed=self.max_posed,
                    precision=self.precision, backend=self.backend)

    def derivation_fields(self, primary_rule: str) -> Dict[str, object]:
        """Settings a CPU `derive` can change: the count rule and the primary rule."""
        return dict(count_score_thr=self.count_score_thr, count_nms_thr=self.count_nms_thr,
                    primary_rule=primary_rule)
