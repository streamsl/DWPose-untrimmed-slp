"""`fetch-models`: put the two checkpoints into the models directory and verify their sha256.

YOLOX-L 640 is downloaded from the mmdetection model zoo and DWPose-L 384x288 (dw-ll_ucoco_384.pth)
from its Hugging Face mirror; a file or directory given with --from is copied instead (offline
installs: the DWPose README also links Google Drive and Baidu). A checkpoint already in place is
verified and kept; a file with the wrong sha256 is an error and is never overwritten. Files are
written to `<name>.part`, verified, then renamed. The configs ship with the package; ONNX files and
TensorRT engines are built from the checkpoints by `build-engines`.
"""
from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, List, Optional, Sequence, Tuple

from .settings import Settings

log = logging.getLogger(__name__)
_BLOCK = 16 << 20


@dataclass(frozen=True)
class Checkpoint:
    """One checkpoint: the Settings property naming its path, its sha256 and where it comes from."""

    attr: str              # Settings property: det_checkpoint / pose_checkpoint
    sha256: str
    url: Optional[str]     # direct download, None = only from --from
    origin: str            # where to get it by hand


CHECKPOINTS = (
    Checkpoint('det_checkpoint', 'd3bd2b23e4cd178bfcc756df67e0d0949f3d77e0a73482f6da694c580ed54da1',
               'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/'
               'yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
               'the mmdetection YOLOX model zoo'),
    Checkpoint('pose_checkpoint', '0d9408b13cd863c4e95a149dd31232f88f2a12aa6cf8964ed74d7d97748c7a07',
               'https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth',
               'the DWPose README (https://github.com/IDEA-Research/DWPose, Google Drive / Baidu links)'),
)


class FetchError(RuntimeError):
    """A checkpoint is missing, unreachable or has the wrong sha256."""


def fetch_models(settings: Settings, sources: Sequence[Path] = (), download: bool = True) -> List[Tuple[Path, str]]:
    """Make every checkpoint of `settings` present and verified; returns [(path, how)] with how in
    'present', 'copied', 'downloaded'.

    `sources`: files or directories to copy from; a checkpoint is found as a file with its name, or
    as <dir>/<name> or <dir>/<its subdirectory>/<name> (the models-dir layout). Copies are preferred
    over downloads; `download=False` never touches the network. Raises FetchError naming every
    checkpoint it could not provide.
    """
    done, missing = [], []
    for ckpt in CHECKPOINTS:
        dest = Path(getattr(settings, ckpt.attr))
        try:
            done.append((dest, _provide(ckpt, dest, sources, download)))
        except FetchError as exc:
            missing.append(str(exc))
    if missing:
        raise FetchError('; '.join(missing))
    return done


def _provide(ckpt: Checkpoint, dest: Path, sources: Sequence[Path], download: bool) -> str:
    if dest.exists():
        got = sha256_file(dest)
        if got != ckpt.sha256:
            raise FetchError(f'{dest} exists with sha256 {got}, expected {ckpt.sha256}; move it away and re-run')
        return 'present'
    source = _find(dest, sources)
    if source is not None:
        log.info('copy %s -> %s', source, dest)
        with open(source, 'rb') as f:
            _install(f, dest, ckpt.sha256, str(source))
        return 'copied'
    if ckpt.url is None or not download:
        raise FetchError(f'{dest.name} not found in --from {[str(s) for s in sources]}; get it from {ckpt.origin} '
                         f'and pass --from FILE_OR_DIR')
    log.info('download %s -> %s', ckpt.url, dest)
    try:
        with urllib.request.urlopen(ckpt.url, timeout=60) as response:
            _install(response, dest, ckpt.sha256, ckpt.url)
    except OSError as exc:   # urllib.error.URLError is an OSError
        raise FetchError(f'{ckpt.url}: {exc}; download it from {ckpt.origin} and pass --from FILE') from exc
    return 'downloaded'


def _find(dest: Path, sources: Sequence[Path]) -> Optional[Path]:
    for source in map(Path, sources):
        candidates = [source] if source.is_file() else [source / dest.name, source / dest.parent.name / dest.name]
        for path in candidates:
            if path.is_file() and path.name == dest.name:
                return path
    return None


def _install(stream: BinaryIO, dest: Path, sha256: str, origin: str) -> None:
    """Copy `stream` to dest.part while hashing it; rename to `dest` only if the sha256 matches."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.part')
    digest = hashlib.sha256()
    try:
        with open(part, 'wb') as out:
            for block in iter(lambda: stream.read(_BLOCK), b''):
                digest.update(block)
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        if digest.hexdigest() != sha256:
            raise FetchError(f'{origin} has sha256 {digest.hexdigest()}, expected {sha256}')
        os.replace(part, dest)
    finally:
        if part.exists():
            part.unlink()


def sha256_file(path: Path) -> str:
    """sha256 hex digest of a file, read in blocks."""
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(_BLOCK), b''):
            digest.update(block)
    return digest.hexdigest()
