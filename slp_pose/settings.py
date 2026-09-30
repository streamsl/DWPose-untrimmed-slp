"""Run configuration: thresholds, batching, precision, model paths and the dataset registry.

Everything that changes GPU output is listed by `Settings.extraction_fields`; everything a CPU
`derive` may change is listed by `Settings.derivation_fields` (spec §4.4).
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]


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
    """Frozen run configuration. Paths are derived from `repo_root`; nothing is hardcoded."""

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
    repo_root: Path = REPO_ROOT

    def __post_init__(self) -> None:
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

    # ------------------------------------------------------------------ paths
    @property
    def models_dir(self) -> Path:
        return self.repo_root / 'models'

    @property
    def det_config(self) -> Path:
        return self.models_dir / 'yolox' / 'yolox_l_8xb8-300e_coco.py'

    @property
    def det_checkpoint(self) -> Path:
        return self.models_dir / 'yolox' / 'yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'

    @property
    def pose_config(self) -> Path:
        return self.models_dir / 'dwpose' / 'dwpose-l_384x288.py'

    @property
    def pose_checkpoint(self) -> Path:
        return self.models_dir / 'dwpose' / 'dw-ll_ucoco_384.pth'

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


@dataclass(frozen=True)
class DatasetSpec:
    """One benchmark: where its videos are, where outputs go, how its signer is chosen.

    Paths are relative to the repo root. The video id is the filename stem.
    """

    name: str
    video_glob: str
    out_root: str
    primary_rule: str
    split_file: Optional[str] = None
    split_order: Tuple[str, ...] = ()

    @staticmethod
    def video_id(path: Path) -> str:
        return Path(path).stem

    def video_paths(self, repo_root: Path = REPO_ROOT) -> List[Path]:
        """All videos of the dataset, sorted by path."""
        return sorted(Path(repo_root).glob(self.video_glob))

    def output_root(self, repo_root: Path = REPO_ROOT) -> Path:
        return Path(repo_root) / self.out_root

    def splits(self, repo_root: Path = REPO_ROOT) -> Dict[str, List[str]]:
        """{split name: [video ids]} in `split_order`; empty when the dataset has no split file."""
        if self.split_file is None:
            return {}
        with open(Path(repo_root) / self.split_file) as f:
            raw = json.load(f)
        unknown = set(raw) - set(self.split_order)
        if unknown:
            raise ValueError(f'{self.name}: splits {sorted(unknown)} missing from split_order')
        return {name: [str(v) for v in raw[name]] for name in self.split_order if name in raw}


DATASETS: Dict[str, DatasetSpec] = {
    'bobsl': DatasetSpec(
        name='bobsl',
        video_glob='data/BOBSL/original_data/videos/mp4/*.mp4',
        out_root='data/BOBSL/dwpose',
        primary_rule='largest_bbox',
        split_file='data/BOBSL/original_data/metadata/subset2episode.json',
        split_order=('val', 'test', 'train', 'challenge_test'),
    ),
}
# Development / pilot outputs for BOBSL (never the production root).
BOBSL_DEV_ROOT = 'data/BOBSL/dwpose_dev'
