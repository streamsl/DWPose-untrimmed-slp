"""Smoke check of the slp_pose environment (scripts/setup_env.sh runs it last).

    python scripts/check_env.py                    # everything below; exit code 1 on any problem
    python scripts/check_env.py --arch 9.0         # also require CUDA code for compute capability 9.0
    python scripts/check_env.py --covers 9.0 mmcv  # only: exit 0 iff mmcv's CUDA ops run on 9.0

Checks: Python 3.11 or 3.8; installed versions equal that Python's pins (requirements.txt for 3.11,
requirements-py38.txt for 3.8); the `slp_pose.env` start-up checks
(pins, ./mmpose location when the checkout has one, TF32 off); onnx/onnxsim/onnxruntime import;
ffmpeg with libx264 and ffprobe; the NVIDIA driver (>= 535 for TensorRT 10); CUDA code of torch,
torchvision and mmcv for every GPU's compute capability (nvidia-smi) and `--arch`; and, when a GPU is
visible, mmcv and torchvision NMS on it and a TensorRT builder.

A CUDA library runs on compute capability X.Y when its fatbinaries hold SASS for X.Z with Z <= Y,
or PTX for an architecture <= X.Y (which the driver compiles when the library loads).
"""
from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import mmap
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[1]
FATBIN_MAGIC = struct.pack('<I', 0xBA55ED50)
MIN_DRIVER = 535   # TensorRT 10 prerequisites: r535 (torch 2.1.2+cu121 alone needs >= 525.60.13)
PTX, SASS = 1, 2   # fatbinary entry kinds
# library -> (top-level package, file inside it that holds the CUDA code)
CUDA_LIBRARIES = {'torch': ('torch', 'lib/libtorch_cuda.so'),
                  'torchvision': ('torchvision', '_C.so'),
                  'mmcv': ('mmcv', '_ext' + importlib.machinery.EXTENSION_SUFFIXES[0])}
# the pinned environment of each supported Python
REQUIREMENTS = {(3, 11): REPO_ROOT / 'requirements.txt', (3, 8): REPO_ROOT / 'requirements-py38.txt'}


def fatbin_entries(data) -> Iterator[Tuple[int, int]]:
    """(kind, sm) of every entry of every CUDA fatbinary in `data`; sm 86 = compute capability 8.6.

    Layout written by nvcc: a 16-byte header (u32 magic, u16 version 1, u16 header size 16,
    u64 size of the entries), then entries with u16 kind at 0, u32 header size at 4, u64 payload
    size at 8 and u32 sm at 28.
    """
    pos = data.find(FATBIN_MAGIC)
    while pos != -1:
        version, header_size, size = struct.unpack_from('<HHQ', data, pos + 4)
        if version != 1 or header_size != 16:
            pos = data.find(FATBIN_MAGIC, pos + 4)
            continue
        entry, end = pos + 16, pos + 16 + size
        while entry < end:
            kind, _, entry_header, payload = struct.unpack_from('<HHIQ', data, entry)
            yield kind, struct.unpack_from('<I', data, entry + 28)[0]
            if entry_header == 0:
                break
            entry += entry_header + payload
        pos = data.find(FATBIN_MAGIC, end)


def library_entries(path: Path) -> List[Tuple[int, int]]:
    """Fatbinary entries of the shared library at `path`."""
    with open(path, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        return list(fatbin_entries(data))


def runs_on(entries: Sequence[Tuple[int, int]], capability: str) -> bool:
    """True when the entries hold code that compute capability `capability` (e.g. '9.0') can run."""
    major, minor = (int(p) for p in capability.split('.'))
    return any((kind == SASS and sm // 10 == major and sm % 10 <= minor) or
               (kind == PTX and sm <= 10 * major + minor) for kind, sm in entries)


def describe(entries: Sequence[Tuple[int, int]]) -> str:
    """e.g. 'SASS sm_80 sm_86, PTX none'."""
    def archs(kind: int) -> str:
        return ' '.join(f'sm_{sm}' for sm in sorted({s for k, s in entries if k == kind})) or 'none'
    return f'SASS {archs(SASS)}, PTX {archs(PTX)}'


def library_path(name: str) -> Path:
    """Path of library `name`'s CUDA code, found without importing the package."""
    package, relative = CUDA_LIBRARIES[name]
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError(f'{package} is not installed')
    return Path(list(spec.submodule_search_locations)[0]) / relative


def nvidia_smi(field: str) -> List[str]:
    """`field` of every GPU nvidia-smi lists ([] without a driver)."""
    if shutil.which('nvidia-smi') is None:
        return []
    out = subprocess.run(['nvidia-smi', f'--query-gpu={field}', '--format=csv,noheader'],
                         capture_output=True, text=True)
    return [line.strip() for line in out.stdout.splitlines() if line.strip()] if out.returncode == 0 else []


def gpu_capabilities() -> List[str]:
    """Distinct compute capabilities of the GPUs nvidia-smi lists."""
    return sorted(set(nvidia_smi('compute_cap')))


def driver_problems(driver_versions: Sequence[str]) -> List[str]:
    """Problems with the NVIDIA driver versions nvidia-smi reports (e.g. '535.183.01')."""
    return [f'NVIDIA driver {v} < {MIN_DRIVER} (TensorRT 10 needs r{MIN_DRIVER} or newer)'
            for v in sorted(set(driver_versions)) if int(v.split('.')[0]) < MIN_DRIVER]


def pinned_versions(requirements: Path) -> Dict[str, str]:
    """{normalized name: version} from the `name==version` and `name @ <wheel URL>` lines."""
    pins = {}
    for line in requirements.read_text().splitlines():
        match = re.match(r'([A-Za-z0-9_.-]+)\s*(?:==\s*([^\s;\\]+)|@\s*(\S+))', line.split('#', 1)[0].strip())
        if match:
            name, version, url = match.groups()
            pins[_normalize(name)] = version or unquote(url.rsplit('/', 1)[-1]).split('-')[1]
    return pins


def _normalize(name: str) -> str:
    return re.sub(r'[-_.]+', '-', name).lower()


def _installed(name: str) -> Optional[str]:
    """Installed version of distribution `name`, None when it is not installed."""
    from importlib import metadata
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def version_problems(requirements: Path) -> List[str]:
    """Installed distributions that are missing or differ from the pins of `requirements`."""
    problems = []
    for name, want in pinned_versions(requirements).items():
        have = _installed(name)
        if have is None and name == 'tensorrt-cu12' and _installed('tensorrt-cu12-bindings') == want:
            # The tensorrt-cu12 sdist installs only tensorrt/__init__.py, two lines re-exporting
            # tensorrt_bindings; the BOBSL .venv has them as a local module instead. The slp_pose
            # start-up checks import it and compare its version.
            print(f'  tensorrt-cu12 not installed: `import tensorrt` must re-export tensorrt_bindings {want}')
        elif have is None:
            problems.append(f'{name} is not installed ({requirements.name} pins {want})')
        elif have != want:
            problems.append(f'{name} {have} != {want} pinned in {requirements.name}')
    return problems


def slp_pose_problems() -> List[str]:
    sys.path.insert(0, str(REPO_ROOT))
    from slp_pose import env
    env.configure_process()
    env.strict_fp32()
    print('  ' + ', '.join(f'{k} {v}' for k, v in env.library_versions(include_trt=True).items()))
    import onnx
    import onnxruntime
    import onnxsim
    print(f'  onnx {onnx.__version__}, onnxsim {onnxsim.__version__}, onnxruntime {onnxruntime.__version__}')
    return env.environment_problems(need_trt=True)


def tool_problems() -> List[str]:
    problems = [f'{tool} not found on PATH' for tool in ('ffmpeg', 'ffprobe') if shutil.which(tool) is None]
    if not problems:
        encoders = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True).stdout
        if 'libx264' not in encoders:
            problems.append('ffmpeg has no libx264 encoder (render / render-done need it)')
    return problems


def cuda_code_problems(capabilities: Sequence[str]) -> List[str]:
    problems = []
    for name in CUDA_LIBRARIES:
        entries = library_entries(library_path(name))
        print(f'  {name}: {describe(entries)}')
        problems += [f'{name} has no CUDA code for compute capability {c} (scripts/setup_env.sh rebuilds mmcv)'
                     for c in capabilities if not runs_on(entries, c)]
    return problems


def gpu_problems() -> List[str]:
    """Run the CUDA kernels the extractor uses outside its engines, and create a TensorRT builder."""
    import torch
    if not torch.cuda.is_available():
        if gpu_capabilities() and os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            return ['nvidia-smi lists GPUs but torch sees none (driver too old for CUDA 12.1, or CUDA_VISIBLE_DEVICES)']
        print('  no GPU visible: skipped')
        return []
    import tensorrt
    import torchvision
    from mmcv.ops import nms
    problems = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        label = f'GPU{i} {torch.cuda.get_device_name(i)} (compute capability {major}.{minor})'
        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.], [50., 50., 60., 60.]], device=f'cuda:{i}')
        scores = torch.tensor([0.9, 0.8, 0.7], device=f'cuda:{i}')
        try:
            kept = sorted(nms(boxes, scores, 0.5)[1].tolist()), sorted(torchvision.ops.nms(boxes, scores, 0.5).tolist())
        except RuntimeError as e:
            problems.append(f'{label}: NMS failed: {e}')
            continue
        if kept != ([0, 2], [0, 2]):
            problems.append(f'{label}: NMS kept {kept}, expected [0, 2] from mmcv and torchvision')
        print(f'  {label}: mmcv and torchvision NMS ran')
    tensorrt.Builder(tensorrt.Logger(tensorrt.Logger.ERROR)).create_builder_config()
    print(f'  TensorRT {tensorrt.__version__} builder created')
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--arch', action='append', default=[], metavar='X.Y[;X.Y]',
                        help='also require CUDA code for these compute capabilities (repeatable)')
    parser.add_argument('--covers', nargs=2, metavar=('X.Y[;X.Y]', 'LIBRARY'),
                        help=f'only check that LIBRARY ({", ".join(CUDA_LIBRARIES)}) runs on these capabilities')
    args = parser.parse_args(argv)
    if args.covers:
        capabilities, name = args.covers[0].split(';'), args.covers[1]
        entries = library_entries(library_path(name))
        missing = [c for c in capabilities if not runs_on(entries, c)]
        print(f'{name}: {describe(entries)}' + (f'; no code for {", ".join(missing)}' if missing else ''))
        return 1 if missing else 0

    requirements = REQUIREMENTS.get(sys.version_info[:2])
    if requirements is None:
        problems = [f'Python {sys.version.split()[0]}, expected 3.11 or 3.8']
    else:
        print(f'installed versions vs {requirements.name}')
        problems = version_problems(requirements)
    print('slp_pose start-up checks')
    problems += slp_pose_problems()
    print('ffmpeg / ffprobe')
    problems += tool_problems()
    drivers = nvidia_smi('driver_version')
    print(f'NVIDIA driver {", ".join(sorted(set(drivers))) or "(none: no nvidia-smi)"}')
    problems += driver_problems(drivers)
    capabilities = sorted(set(gpu_capabilities()) | {c for a in args.arch for c in a.split(';')})
    print(f'CUDA code for compute capability {", ".join(capabilities) or "(none: no GPU found, no --arch)"}')
    problems += cuda_code_problems(capabilities)
    print('GPU kernels')
    problems += gpu_problems()
    for p in problems:
        print(f'PROBLEM: {p}')
    print('environment OK' if not problems else f'{len(problems)} problem(s)')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
