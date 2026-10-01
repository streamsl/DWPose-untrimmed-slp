#!/usr/bin/env bash
# Build the pinned Python environment of slp_pose in a fresh clone (Linux x86_64).
#
#   scripts/setup_env.sh [--venv DIR] [--python 3.11|3.8]      (defaults: <repo>/.venv, 3.11)
#
#   1. checks Linux x86_64, glibc >= 2.28, ffmpeg with libx264, ffprobe
#   2. creates the venv from that Python: $PYTHON, else pythonX.Y on PATH, else uv's (installed with
#      `uv python install --no-bin X.Y` when uv is on PATH and has none; $UV_PYTHON_INSTALL_DIR is respected)
#   3. installs requirements.txt (3.11, newest stack) or requirements-py38.txt (3.8, the BOBSL machine):
#      exact versions, every file hash-checked, 6.9 GB of downloads (5.7 GB for 3.8). mmpose is the
#      PyPI wheel, except in a checkout with its own ./mmpose source tree (gitignored, so never in a
#      fresh clone; the BOBSL machine has one): that tree is linked into the venv instead, as
#      slp_pose.env requires then.
#   4. deletes TensorRT's Windows-only builder resources (1.8 GB in 10.13, 2.9 GB in 10.16; they serve
#      engines built for Windows only); `import tensorrt` comes from the tensorrt-cu12 sdist, no shim
#   5. rebuilds mmcv's CUDA ops when the prebuilt wheel has no code for a GPU (it stops at sm_86, so an
#      H100 = compute capability 9.0 needs it; needs a CUDA 12.x nvcc and a host compiler it supports)
#   6. installs this checkout editable with its [trt] extra (`slp-pose` command)
#   7. pip check + scripts/check_env.py (imports, versions, ffmpeg, CUDA code for every GPU, GPU kernels)
# Each step skips what is already done, so after a failure fix the cause and re-run. It refuses to
# touch a venv it did not create.
#
# Environment:
#   PYTHON          interpreter for the new venv (must be the chosen Python version)
#   MMCV_CUDA_ARCH  compute capabilities for step 5, e.g. 9.0 or "8.6;9.0" [the GPUs', from nvidia-smi]
#   CUDA_HOME       CUDA 12.x toolkit for step 5 [/usr/local/cuda]
#   MAX_JOBS        parallel compile jobs for step 5 [nproc]
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=$REPO/.venv
PY_VERSION=3.11
MMCV_SDIST='mmcv @ https://files.pythonhosted.org/packages/a7/19/f97f0fd4d432113a232e75f965335be6b7bb6557ae9faf8051e9a012cadd/mmcv-2.1.0.tar.gz#sha256=d387bcab66b467479b6660310e23746cfc79c6e57acf04094680adb499a5cd3f'
NINJA='ninja==1.13.2 --hash=sha256:65a24341b5ac09fcadcc37082660be40a94174e51a937fabf6e2cae26225fa2c'
MARKER=.created-by-slp-pose-setup

die() { echo "setup_env: ERROR: $*" >&2; exit 1; }
step() { echo "== $* (at $(( $(date +%s) - START )) s)"; }
is_python() { "$1" -c "import sys; sys.exit(sys.version_info[:2] != tuple(map(int, '$PY_VERSION'.split('.'))))" 2> /dev/null; }

find_python() {
    if [ -n "${PYTHON:-}" ]; then
        is_python "$PYTHON" || die "PYTHON=$PYTHON is not a working Python $PY_VERSION"
        echo "$PYTHON"; return
    fi
    local py
    py=$(command -v "python$PY_VERSION" || true)
    if [ -n "$py" ] && is_python "$py"; then echo "$py"; return; fi
    command -v uv > /dev/null || die "no Python $PY_VERSION found. Install one and re-run, e.g.
    curl -LsSf https://astral.sh/uv/install.sh | sh     (then this script installs Python $PY_VERSION with uv)
or  pyenv install $PY_VERSION && PYTHON=\$(pyenv root)/versions/<that version>/bin/python scripts/setup_env.sh"
    py=$(uv python find "$PY_VERSION" 2> /dev/null || true)
    if [ -z "$py" ] || ! is_python "$py"; then
        echo "installing Python $PY_VERSION with uv" >&2
        # --no-bin: no pythonX.Y link in ~/.local/bin (the venv is all that uses this interpreter).
        uv python install -q --no-bin "$PY_VERSION" >&2
        py=$(uv python find "$PY_VERSION")
    fi
    is_python "$py" || die "uv's $py is not a working Python $PY_VERSION"
    echo "$py"
}

# The lines of packages $2 ("name name ...") in requirements file $1, with their --hash continuation lines.
pins_of() {
    awk -v names=" $2 " '/^[A-Za-z0-9]/ { split($1, p, "=="); keep = index(names, " " tolower(p[1]) " ") > 0 }
                         keep && !/^#/' "$1"
}

while [ $# -gt 0 ]; do
    case $1 in
        --venv) [ $# -ge 2 ] || die '--venv needs a directory'; VENV=$(realpath -m "$2"); shift 2 ;;
        --python) [ $# -ge 2 ] || die '--python needs 3.11 or 3.8'; PY_VERSION=$2; shift 2 ;;
        -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument $1 (see --help)" ;;
    esac
done
case $PY_VERSION in
    3.11) REQUIREMENTS=$REPO/requirements.txt ;;
    3.8) REQUIREMENTS=$REPO/requirements-py38.txt ;;
    *) die "--python $PY_VERSION: only 3.11 (requirements.txt) and 3.8 (requirements-py38.txt) are pinned" ;;
esac
START=$(date +%s)

step '1/7 system'
[ "$(uname -s)-$(uname -m)" = Linux-x86_64 ] || die 'needs Linux x86_64'
# (Command outputs are captured before grep: with pipefail, a reader that stops early would fail the
# pipeline through SIGPIPE.)
GLIBC=$(getconf GNU_LIBC_VERSION | awk '{print $2}')
[ "$(printf '%s\n' 2.28 "$GLIBC" | sort -V | sed -n 1p)" = 2.28 ] ||
    die "glibc $GLIBC < 2.28 (the TensorRT and OpenCV wheels are manylinux_2_28)"
for tool in ffmpeg ffprobe; do
    command -v "$tool" > /dev/null || die "$tool not found (Debian/Ubuntu: sudo apt-get install -y ffmpeg)"
done
ENCODERS=$(ffmpeg -hide_banner -encoders 2> /dev/null)
grep -q libx264 <<< "$ENCODERS" || die 'ffmpeg has no libx264 encoder'
FFMPEG=$(ffmpeg -version)
echo "glibc $GLIBC, ${FFMPEG%%Copyright*}"

step "2/7 venv $VENV (Python $PY_VERSION)"
if [ -e "$VENV" ]; then
    [ -f "$VENV/$MARKER" ] || die "$VENV exists but was not created by this script; delete it or pass --venv NEW_DIR"
else
    PY=$(find_python)
    # The base interpreter, so that the new venv does not depend on another venv that provided it.
    PY=$("$PY" -c "import os, sys; print(os.path.join(sys.base_prefix, 'bin', 'python$PY_VERSION'))")
    is_python "$PY" || die "$PY is not a working Python $PY_VERSION"
    echo "creating it with $PY ($("$PY" -V 2>&1))"
    "$PY" -m venv "$VENV"
    touch "$VENV/$MARKER"
fi
VPY=$VENV/bin/python
is_python "$VPY" || die "$VPY is not Python $PY_VERSION (pass the matching --python)"
SITE=$("$VPY" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
# Wheels of several GB: keep pip's temporary files on the venv's disk, not in a RAM /tmp.
export TMPDIR=$VENV/.tmp PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
mkdir -p "$TMPDIR"
trap 'rm -rf "$VENV/.tmp"' EXIT
pip_q() { "$VPY" -m pip -q "$@"; }

step "3/7 $(basename "$REQUIREMENTS")"
if [ -d "$REPO/mmpose/mmpose" ]; then
    # Like `pip install -e ./mmpose` without writing into ./mmpose: the path makes its package and its
    # mmpose.egg-info (the installed version for pip) visible, so the pinned mmpose counts as installed.
    [ -d "$REPO/mmpose/mmpose.egg-info" ] || die "$REPO/mmpose has no mmpose.egg-info, so pip would also install
the mmpose wheel; move ./mmpose out of the checkout to use the wheel"
    echo "$REPO/mmpose" > "$SITE/slp_pose_mmpose_source.pth"
    echo "mmpose: the source tree $REPO/mmpose"
fi
pins_of "$REQUIREMENTS" 'pip setuptools wheel' > "$TMPDIR/build-tools.txt"
pip_q install --require-hashes --no-deps -r "$TMPDIR/build-tools.txt"
pip_q install --require-hashes --no-deps --no-build-isolation -r "$REQUIREMENTS"

step '4/7 TensorRT'
# 10.13: libnvinfer_builder_resource_win.so.*; 10.16: libnvinfer_builder_resource_win_{ptx,sm75,...}.so.*
rm -f "$SITE"/tensorrt_libs/libnvinfer_builder_resource_win*.so.*
"$VPY" -I -c 'import tensorrt; print("tensorrt", tensorrt.__version__)'

step '5/7 mmcv CUDA ops'
ARCH=${MMCV_CUDA_ARCH:-}
if [ -z "$ARCH" ] && command -v nvidia-smi > /dev/null; then
    ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | sort -u | paste -sd';' - || true)
fi
if [ -z "$ARCH" ]; then
    echo 'no GPU found and MMCV_CUDA_ARCH unset: skipped (re-run this script on the GPU machine)'
elif "$VPY" "$REPO/scripts/check_env.py" --covers "$ARCH" mmcv; then
    echo "mmcv's CUDA ops run on compute capability $ARCH"
else
    CUDA=${CUDA_HOME:-/usr/local/cuda}
    [ -x "$CUDA/bin/nvcc" ] || die "mmcv must be rebuilt for compute capability $ARCH, but $CUDA/bin/nvcc is missing:
install a CUDA 12.x toolkit (see README) and set CUDA_HOME"
    NVCC=$("$CUDA/bin/nvcc" --version)
    grep -q 'release 12\.' <<< "$NVCC" ||
        die "$CUDA/bin/nvcc is not CUDA 12.x (torch 2.1.2 is built with CUDA 12.1; set CUDA_HOME)"
    echo "building mmcv 2.1.0 for $ARCH with $(grep release <<< "$NVCC") (10-30 min)"
    echo "$NINJA" > "$TMPDIR/ninja.txt"
    pip_q install --require-hashes --no-deps -r "$TMPDIR/ninja.txt"
    # --no-cache-dir: pip's wheel cache does not know TORCH_CUDA_ARCH_LIST.
    PATH=$VENV/bin:$CUDA/bin:$PATH CUDA_HOME=$CUDA MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=$ARCH \
        MAX_JOBS=${MAX_JOBS:-$(nproc)} \
        "$VPY" -m pip install -q --no-deps --no-build-isolation --no-cache-dir --force-reinstall "$MMCV_SDIST"
    pip_q uninstall -y ninja
    "$VPY" "$REPO/scripts/check_env.py" --covers "$ARCH" mmcv || die "the rebuilt mmcv has no code for $ARCH"
fi

step '6/7 slp-pose[trt] (editable)'
if ! "$VPY" -I - "$REPO" << 'EOF'
import json, sys
from importlib import metadata
from pathlib import Path
try:
    url = json.loads(metadata.distribution('slp-pose').read_text('direct_url.json') or '{}')
except metadata.PackageNotFoundError:
    sys.exit(1)
sys.exit(not (url.get('dir_info', {}).get('editable') and url.get('url') == Path(sys.argv[1]).as_uri()))
EOF
then
    if "$VPY" -I -c 'import setuptools, sys; sys.exit(int(setuptools.__version__.split(".")[0]) < 64)'; then
        # --no-index: every requirement of slp-pose[trt] must already be installed from the lock file.
        pip_q install --no-index --no-build-isolation -e "$REPO[trt]"
    else
        # An editable pyproject build needs setuptools >= 64, newer than requirements-py38.txt's (the BOBSL
        # venv's): pip builds it in isolation with a current setuptools; pip check covers the requirements.
        pip_q install --no-deps -e "$REPO"
    fi
fi
"$VENV/bin/slp-pose" --help > /dev/null

step '7/7 checks'
"$VPY" -m pip check
(cd "$REPO" && "$VPY" scripts/check_env.py ${MMCV_CUDA_ARCH:+--arch "$MMCV_CUDA_ARCH"})
echo "setup_env: done in $(( $(date +%s) - START )) s"
