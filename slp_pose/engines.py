"""Models, ONNX export, TensorRT build and the inference engines (spec §4.2 engines.py, D7).

This is the only module that builds mmdet / mmpose models. Both backends share one interface:
- detector engine: x (B,3,640,640) float32 CUDA, BGR 0..255 letterbox (no normalisation)
  -> list of 9 float32 maps in DET_OUTPUTS order: cls (B,80,h,w), box (B,4,h,w), obj (B,1,h,w)
  for strides 8, 16, 32 (h = w = 80, 40, 20);
- pose engine: x (B,3,384,288) float32 CUDA, normalised RGB -> (simcc_x (B,133,576),
  simcc_y (B,133,768)) raw float32 logits, no flip handling.
TensorRT engines are fp32 with TF32 cleared, dynamic batch in the Settings profiles, stored in
models/engines/ with a `<file>.sha256` (sha256sum format) and a `<file>.json` build record.

TensorRT links its own CUDA runtime statically, so a TRT process should see exactly one GPU
(CUDA_VISIBLE_DEVICES) or at least make `device` the current CUDA device before loading.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn

from .env import is_strict_fp32, library_versions, strict_fp32
from .settings import Settings

DET_OUTPUTS = tuple(f'{kind}_s{stride}' for kind in ('cls', 'box', 'obj') for stride in (8, 16, 32))
POSE_OUTPUTS = ('simcc_x', 'simcc_y')


@dataclass(frozen=True)
class ModelSpec:
    key: str                        # 'det' or 'pose'
    stem: str                       # file stem for .onnx / .engine
    input_shape: Tuple[int, int, int]
    outputs: Tuple[str, ...]


MODEL_SPECS = {
    'det': ModelSpec('det', 'yolox_l_640', (3, 640, 640), DET_OUTPUTS),
    'pose': ModelSpec('pose', 'dwpose_l_384x288', (3, 384, 288), POSE_OUTPUTS),
}


# --------------------------------------------------------------------------- metadata
@dataclass(frozen=True)
class DetMeta:
    """Post-processing constants of the detector head, read from its config."""

    score_thr: float             # head pre-NMS score threshold (0.01)
    nms_iou: float               # head class-aware NMS IoU threshold (0.65), mmcv nms
    strides: Tuple[int, ...]     # (8, 16, 32)
    num_classes: int             # 80, class 0 = person


@dataclass(frozen=True)
class PoseMeta:
    """Constants of the pose model: normalisation, flip pairs, SimCC decode, dataset metainfo."""

    flip_indices: Tuple[int, ...]            # 133 entries, from the checkpoint's dataset_meta
    mean: Tuple[float, float, float]         # RGB, applied after BGR->RGB
    std: Tuple[float, float, float]
    input_size: Tuple[int, int]              # (w, h) = (288, 384)
    simcc_split_ratio: float                 # 2.0
    dataset_meta: dict                       # full mmpose dataset_meta (skeleton, names, colours)


def det_meta(settings: Settings) -> DetMeta:
    """Read the YOLOX head constants from the detector config."""
    from mmengine import Config
    cfg = Config.fromfile(str(settings.det_config))
    test_cfg, head = cfg.model.test_cfg, cfg.model.bbox_head
    if test_cfg.nms.type != 'nms':
        raise ValueError(f'unexpected head NMS {test_cfg.nms}')
    return DetMeta(score_thr=float(test_cfg.score_thr), nms_iou=float(test_cfg.nms.iou_threshold),
                   strides=tuple(int(s) for s in head.strides), num_classes=int(head.num_classes))


def pose_meta(settings: Settings) -> PoseMeta:
    """Read the pose constants from the config and the checkpoint's dataset_meta.

    mmpose.apis.init_model takes dataset_meta from the checkpoint first, so the config's relative
    `metainfo.from_file` is never resolved; this reads the same source.
    """
    from mmengine import Config
    cfg = Config.fromfile(str(settings.pose_config))
    ckpt = torch.load(str(settings.pose_checkpoint), map_location='cpu', mmap=True)
    dataset_meta = ckpt['meta']['dataset_meta']
    pre, codec = cfg.model.data_preprocessor, cfg.codec
    if not (pre.bgr_to_rgb and not codec.use_dark and cfg.model.test_cfg.flip_test):
        raise ValueError('pose config differs from the DWPose-L setup this code reproduces')
    return PoseMeta(flip_indices=tuple(int(i) for i in dataset_meta['flip_indices']),
                    mean=tuple(float(v) for v in pre.mean), std=tuple(float(v) for v in pre.std),
                    input_size=tuple(int(v) for v in codec.input_size),
                    simcc_split_ratio=float(codec.simcc_split_ratio), dataset_meta=dataset_meta)


# --------------------------------------------------------------------------- torch models
def build_detector(settings: Settings, device: torch.device) -> nn.Module:
    """The official mmdet YOLOX-L (init_detector + mmpose's adapt_mmdet_pipeline), eval mode."""
    from mmdet.apis import init_detector
    from mmpose.utils import adapt_mmdet_pipeline
    det = init_detector(str(settings.det_config), str(settings.det_checkpoint), device=str(device))
    det.cfg = adapt_mmdet_pipeline(det.cfg)
    return det.eval()


def build_pose_model(settings: Settings, device: torch.device) -> nn.Module:
    """The official mmpose DWPose-L (init_model, flip_test=True from the config), eval mode."""
    from mmpose.apis import init_model
    return init_model(str(settings.pose_config), str(settings.pose_checkpoint), device=str(device)).eval()


class DetRaw(nn.Module):
    """backbone + neck + head -> the 9 raw maps in DET_OUTPUTS order."""

    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        self.detector = detector

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        cls, box, obj = self.detector.bbox_head(self.detector.extract_feat(x))
        return tuple(cls) + tuple(box) + tuple(obj)


class PoseRaw(nn.Module):
    """backbone + RTMCC head -> (simcc_x, simcc_y) logits for un-flipped inputs."""

    def __init__(self, pose_model: nn.Module) -> None:
        super().__init__()
        self.pose_model = pose_model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pose_model.head.forward(self.pose_model.extract_feat(x))


def _swish_to_silu(model: nn.Module) -> None:
    from mmcv.cnn.bricks.swish import Swish
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, Swish):
                setattr(module, name, nn.SiLU())


def _cuda_device(device) -> torch.device:
    """`device` with an explicit CUDA index ('cuda' -> the current device)."""
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError(f'engines run on CUDA devices, got {device}')
    return device if device.index is not None else torch.device('cuda', torch.cuda.current_device())


def _check_input(x: torch.Tensor, shape: Tuple[int, int, int], device: torch.device, max_batch: int) -> None:
    if x.dtype != torch.float32 or tuple(x.shape[1:]) != shape or x.device != device:
        raise ValueError(f'expected float32 (B,{shape}) on {device}, got {x.dtype} {tuple(x.shape)} on {x.device}')
    if not 1 <= x.shape[0] <= max_batch:
        raise ValueError(f'batch {x.shape[0]} outside 1..{max_batch}')


class TorchDetEngine:
    """Strict-fp32 PyTorch detector (Swish replaced by nn.SiLU), the reference backend."""

    backend = 'torch'

    def __init__(self, settings: Settings, device: torch.device) -> None:
        strict_fp32()
        self.device = _cuda_device(device)
        self.max_batch = settings.det_profile[2]
        detector = build_detector(settings, self.device)
        _swish_to_silu(detector)
        self._net = DetRaw(detector).eval()

    def __call__(self, x: torch.Tensor) -> List[torch.Tensor]:
        _check_input(x, MODEL_SPECS['det'].input_shape, self.device, self.max_batch)
        if not is_strict_fp32():
            raise RuntimeError('TF32 was re-enabled after the engine was built')
        with torch.no_grad():
            return list(self._net(x))


class TorchPoseEngine:
    """Strict-fp32 PyTorch DWPose-L, the reference backend."""

    backend = 'torch'

    def __init__(self, settings: Settings, device: torch.device) -> None:
        strict_fp32()
        self.device = _cuda_device(device)
        self.max_batch = settings.pose_profile[2]
        self._net = PoseRaw(build_pose_model(settings, self.device)).eval()

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _check_input(x, MODEL_SPECS['pose'].input_shape, self.device, self.max_batch)
        if not is_strict_fp32():
            raise RuntimeError('TF32 was re-enabled after the engine was built')
        with torch.no_grad():
            sx, sy = self._net(x)
        return sx, sy


# --------------------------------------------------------------------------- TensorRT
@functools.lru_cache(maxsize=None)
def _trt_logger():
    """One logger per process: TensorRT keeps the first one it sees and warns about any other."""
    import tensorrt as trt
    return trt.Logger(trt.Logger.WARNING)


class TrtEngine:
    """A deserialised TensorRT engine run with execute_async_v3 on torch tensors.

    The engine file must match its .sha256 sidecar. Outputs are allocated per call on the input's
    device and enqueued on the current torch stream (no synchronisation). Run under a non-default
    stream (`with torch.cuda.stream(torch.cuda.Stream()):`): on the legacy default stream TensorRT
    adds a cudaStreamSynchronize per call.
    """

    backend = 'trt'

    def __init__(self, path: Path, spec: ModelSpec, device: torch.device) -> None:
        import tensorrt as trt
        self.path = Path(path)
        self.sha256 = verified_sha256(self.path)
        self.device = _cuda_device(device)
        self._spec = spec
        torch.cuda.set_device(self.device)
        self._runtime = trt.Runtime(_trt_logger())
        self._engine = self._runtime.deserialize_cuda_engine(self.path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f'TensorRT could not deserialise {self.path} (TRT version / GPU mismatch?)')
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError(f'no execution context for {self.path.name} (out of GPU memory?)')
        names = [self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors)]
        inputs = [n for n in names if self._engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        missing = set(spec.outputs) - set(names)
        if inputs != ['input'] or missing:
            raise RuntimeError(f'{self.path}: unexpected I/O tensors {names}')
        self.max_batch = int(self._engine.get_tensor_profile_shape('input', 0)[2][0])

    def run(self, x: torch.Tensor) -> List[torch.Tensor]:
        _check_input(x, self._spec.input_shape, self.device, self.max_batch)
        if torch.cuda.current_device() != self.device.index:
            raise RuntimeError(f'current CUDA device {torch.cuda.current_device()} is not {self.device}')
        x = x.contiguous()
        ctx = self._context
        ctx.set_input_shape('input', tuple(x.shape))
        ctx.set_tensor_address('input', x.data_ptr())
        outs = []
        for name in self._spec.outputs:
            out = torch.empty(tuple(ctx.get_tensor_shape(name)), dtype=torch.float32, device=x.device)
            ctx.set_tensor_address(name, out.data_ptr())
            outs.append(out)
        if not ctx.execute_async_v3(torch.cuda.current_stream(x.device).cuda_stream):
            raise RuntimeError(f'TensorRT execution failed for {self.path.name}')
        return outs


class TrtDetEngine(TrtEngine):
    def __init__(self, path: Path, device: torch.device) -> None:
        super().__init__(path, MODEL_SPECS['det'], device)

    def __call__(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.run(x)


class TrtPoseEngine(TrtEngine):
    def __init__(self, path: Path, device: torch.device) -> None:
        super().__init__(path, MODEL_SPECS['pose'], device)

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sx, sy = self.run(x)
        return sx, sy


def load_engines(settings: Settings, device: torch.device):
    """(detector engine, pose engine) for settings.backend on `device`."""
    if settings.backend == 'torch':
        return TorchDetEngine(settings, device), TorchPoseEngine(settings, device)
    return (TrtDetEngine(engine_path(settings, 'det'), device),
            TrtPoseEngine(engine_path(settings, 'pose'), device))


# --------------------------------------------------------------------------- files and hashes
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + '.sha256')


def write_sha256_sidecar(path: Path) -> str:
    """Write `<path>.sha256` in sha256sum format and return the digest."""
    digest = sha256_file(path)
    _atomic_write_text(_sidecar(path), f'{digest}  {path.name}\n')
    return digest


def verified_sha256(path: Path) -> str:
    """sha256 of `path`, checked against its sidecar (raises if missing or different)."""
    side = _sidecar(path)
    if not side.exists():
        raise FileNotFoundError(f'{side} missing; rebuild with `python -m slp_pose build-engines`')
    want = side.read_text().split()[0]
    got = sha256_file(path)
    if got != want:
        raise RuntimeError(f'{path} sha256 {got} does not match its sidecar {want}')
    return got


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def onnx_path(settings: Settings, key: str) -> Path:
    return settings.onnx_dir / f'{MODEL_SPECS[key].stem}.onnx'


def engine_path(settings: Settings, key: str) -> Path:
    profile = settings.det_profile if key == 'det' else settings.pose_profile
    return settings.engine_dir / f'{MODEL_SPECS[key].stem}_{settings.precision}_b{profile[2]}.engine'


def model_provenance(settings: Settings) -> Dict[str, object]:
    """sha256 of configs, checkpoints and (for the TRT backend) engines + their build records."""
    out: Dict[str, object] = dict(
        det_config_sha256=sha256_file(settings.det_config), det_checkpoint_sha256=sha256_file(settings.det_checkpoint),
        pose_config_sha256=sha256_file(settings.pose_config),
        pose_checkpoint_sha256=sha256_file(settings.pose_checkpoint))
    if settings.backend == 'trt':
        for key in MODEL_SPECS:
            path = engine_path(settings, key)
            out[f'{key}_engine'] = path.name
            out[f'{key}_engine_sha256'] = verified_sha256(path)
            out[f'{key}_engine_build'] = json.loads(path.with_name(path.name + '.json').read_text())
    return out


# --------------------------------------------------------------------------- export and build
def export_onnx(settings: Settings, key: str, device: torch.device, force: bool = False) -> Path:
    """Export the strict-fp32 model to ONNX (opset 17, dynamic batch), simplify with onnxsim."""
    import onnx
    import onnxsim
    spec, out = MODEL_SPECS[key], onnx_path(settings, key)
    if out.exists() and not force:
        verified_sha256(out)
        return out
    strict_fp32()
    device = torch.device(device)
    net = DetRaw(build_detector(settings, device)) if key == 'det' else PoseRaw(build_pose_model(settings, device))
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_name(out.stem + '.raw.onnx')
    dynamic = {name: {0: 'batch'} for name in ('input',) + spec.outputs}
    with torch.no_grad():
        torch.onnx.export(net.eval(), torch.randn((2,) + spec.input_shape, device=device), str(raw),
                          opset_version=17, input_names=['input'], output_names=list(spec.outputs),
                          dynamic_axes=dynamic, do_constant_folding=True)
    simplified, ok = onnxsim.simplify(onnx.load(str(raw)))
    if not ok:
        raise RuntimeError(f'onnxsim could not validate the simplified {key} model')
    tmp = out.with_name(out.name + '.tmp')
    onnx.save(simplified, str(tmp))
    os.replace(tmp, out)
    raw.unlink()
    write_sha256_sidecar(out)
    return out


def build_trt_engine(settings: Settings, key: str, device: torch.device, force: bool = False) -> Path:
    """Build the TensorRT engine for `key` from its ONNX file (fp32, TF32 cleared unless allowed).

    Refuses to overwrite an existing engine unless `force` (engines are never rebuilt mid-corpus).
    """
    import tensorrt as trt
    spec, out = MODEL_SPECS[key], engine_path(settings, key)
    if out.exists() and not force:
        verified_sha256(out)
        return out
    source = onnx_path(settings, key)
    onnx_sha = verified_sha256(source)
    device = torch.device(device)
    torch.cuda.set_device(device)
    builder = trt.Builder(_trt_logger())
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, _trt_logger())
    if not parser.parse(source.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError('ONNX parse failed: ' + '; '.join(errors))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    if not settings.profile.tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
    lo, opt, hi = settings.det_profile if key == 'det' else settings.pose_profile
    profile = builder.create_optimization_profile()
    profile.set_shape('input', (lo,) + spec.input_shape, (opt,) + spec.input_shape, (hi,) + spec.input_shape)
    config.add_optimization_profile(profile)
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f'TensorRT build failed for {key}')
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + '.tmp')
    with open(tmp, 'wb') as f:
        f.write(serialized)
    os.replace(tmp, out)
    record = dict(model=spec.stem, onnx=source.name, onnx_sha256=onnx_sha, precision=settings.precision,
                  tf32=settings.profile.tf32, profile=[lo, opt, hi], tensorrt=trt.__version__,
                  gpu=torch.cuda.get_device_name(device),
                  compute_capability='%d.%d' % torch.cuda.get_device_capability(device),
                  build_seconds=round(time.time() - t0, 1), libraries=library_versions())
    _atomic_write_text(out.with_name(out.name + '.json'), json.dumps(record, indent=1, sort_keys=True) + '\n')
    write_sha256_sidecar(out)
    return out


def build_engines(settings: Settings, device: torch.device, force: bool = False) -> Dict[str, Path]:
    """Export both ONNX models (if missing) and build both TRT engines; returns {key: engine path}.

    Existing, sidecar-verified files are kept unless `force`. The CLI's `build-engines` calls this.
    """
    paths = {}
    for key in MODEL_SPECS:
        export_onnx(settings, key, device, force=force)
        paths[key] = build_trt_engine(settings, key, device, force=force)
    return paths


def engine_summary(paths: Dict[str, Path]) -> List[dict]:
    """Human-readable {path, sha256, build record} per engine, for logs and reports."""
    return [dict(path=str(p), sha256=verified_sha256(p), **json.loads(p.with_name(p.name + '.json').read_text()))
            for p in paths.values()]
