"""Start-up checks and process setup (spec §4.2 env.py).

`configure_process()` must run before CUDA is initialised; `python -m slp_pose` does it first.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from .settings import REPO_ROOT

PINNED_VERSIONS = {'torch': '2.1.2', 'mmcv': '2.1.0', 'mmdet': '3.3.0', 'mmpose': '1.3.2',
                   'tensorrt': '10.13.3.9'}


class EnvironmentProblem(RuntimeError):
    """The interpreter, libraries or process settings differ from what the outputs assume."""


def configure_process() -> None:
    """Pin GPU numbering to PCI bus order (GPU0 = display GPU) before CUDA starts."""
    order = os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    if order != 'PCI_BUS_ID':
        raise EnvironmentProblem(f'CUDA_DEVICE_ORDER={order!r}; expected PCI_BUS_ID')


def strict_fp32() -> None:
    """Disable TF32 in cuDNN convolutions and cuBLAS matmuls; no cuDNN autotuning."""
    import torch
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def is_strict_fp32() -> bool:
    import torch
    return not (torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32)


def library_versions(include_trt: bool = True) -> Dict[str, str]:
    """Versions of the libraries that determine numerical output."""
    import cv2
    import mmcv
    import mmdet
    import mmengine
    import mmpose
    import numpy
    import torch
    out = dict(torch=torch.__version__, cuda=str(torch.version.cuda), cudnn=str(torch.backends.cudnn.version()),
               mmcv=mmcv.__version__, mmdet=mmdet.__version__, mmengine=mmengine.__version__,
               mmpose=mmpose.__version__, numpy=numpy.__version__, opencv=cv2.__version__)
    if include_trt:
        import tensorrt
        out['tensorrt'] = tensorrt.__version__
    return out


def environment_problems(need_trt: bool, repo_root: Path = REPO_ROOT) -> List[str]:
    """Everything wrong with the environment, as human-readable lines (empty = fine)."""
    import mmpose
    problems = []
    expected = (Path(repo_root) / 'mmpose' / 'mmpose').resolve()
    got = Path(mmpose.__file__ or '').resolve().parent
    if got != expected:
        problems.append(f'mmpose imported from {got}, expected {expected} (check .venv easy-install.pth)')
    versions = library_versions(include_trt=need_trt)
    for name, want in PINNED_VERSIONS.items():
        if name == 'tensorrt' and not need_trt:
            continue
        have = versions[name].split('+')[0]
        if have != want:
            problems.append(f'{name} {versions[name]} != pinned {want}')
    if os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
        problems.append('CUDA_DEVICE_ORDER is not PCI_BUS_ID')
    if not is_strict_fp32():
        problems.append('TF32 is enabled in torch')
    return problems


def check_environment(need_trt: bool, repo_root: Path = REPO_ROOT) -> None:
    """Raise EnvironmentProblem listing every mismatch."""
    problems = environment_problems(need_trt, repo_root)
    if problems:
        raise EnvironmentProblem('; '.join(problems))


def gpu_name(device_index: int = 0) -> str:
    import torch
    return torch.cuda.get_device_name(device_index)


def git_sha(repo_root: Path = REPO_ROOT) -> Optional[str]:
    """HEAD commit of the repo (read-only `git rev-parse`), or None outside a git checkout."""
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(repo_root), capture_output=True,
                             text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None
