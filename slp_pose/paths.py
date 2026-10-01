"""Where inputs live: the data root, the models directory and the shipped model configs.

Resolution order, first match wins:
- data root: an explicit value (CLI --data-root), $SLP_POSE_DATA, ./data (relative to the
  current directory);
- models directory (checkpoints, ONNX, TensorRT engines): an explicit value (CLI --models-dir),
  $SLP_POSE_MODELS, <checkout>/models when running from a source checkout (fetch-models creates
  it), else the per-user cache ($XDG_CACHE_HOME or ~/.cache)/slp_pose/models.
The detector and pose configs ship inside the package (CONFIG_DIR), for a checkout and an installed
copy alike; their sha256 is part of the extraction hash, so they are never edited.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, 'os.PathLike[str]']

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent          # the source checkout (site-packages for an installed copy)
CONFIG_DIR = PACKAGE_DIR / 'model_configs'
DATA_ENV = 'SLP_POSE_DATA'
MODELS_ENV = 'SLP_POSE_MODELS'


def is_source_checkout() -> bool:
    """True when slp_pose runs from its source tree (pyproject.toml next to the package)."""
    return (REPO_ROOT / 'pyproject.toml').is_file()


def data_root(explicit: Optional[PathLike] = None) -> Path:
    """The absolute data root (module docstring)."""
    return Path(explicit or os.environ.get(DATA_ENV) or 'data').expanduser().resolve()


def models_dir(explicit: Optional[PathLike] = None) -> Path:
    """The absolute models directory (module docstring); it need not exist yet."""
    value = explicit or os.environ.get(MODELS_ENV)
    if value:
        return Path(value).expanduser().resolve()
    if is_source_checkout():
        return REPO_ROOT / 'models'
    cache = os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache'
    return Path(cache).expanduser().resolve() / 'slp_pose' / 'models'
