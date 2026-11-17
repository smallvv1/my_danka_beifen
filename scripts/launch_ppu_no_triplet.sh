#!/usr/bin/env bash
set -euo pipefail

# Local-only launcher for the single visible PPU.  It deliberately selects the
# SAIL PyTorch tree and refuses to fall back to the system CPU-only torch.
PPU_TORCH_ROOT="${PPU_TORCH_ROOT:-/tmp/pytorch-sail-converted}"
PPU_SDK_ROOT="${PPU_SDK_ROOT:-/usr/local/PPU_SDK}"
PPU_COMPAT_LIB="${PPU_COMPAT_LIB:-/tmp/ppu_compat/libacsparse_compat.so}"
PPU_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

test -f "${PPU_SDK_ROOT}/envsetup.sh"
test -f "${PPU_TORCH_ROOT}/torch/__init__.py"
test -f "${PPU_TORCH_ROOT}/torch/lib/libtorch_python.so"
test -f "${PPU_COMPAT_LIB}"

# envsetup.sh intentionally narrows PATH to the PPU toolchain; retain CMake
# and normal host utilities after sourcing it.
source "${PPU_SDK_ROOT}/envsetup.sh" ppu
export PATH="/usr/local/bin:${PATH}"
export PYTORCH_SAIL_ARCH="${PYTORCH_SAIL_ARCH:-ppu_10}"
export PPU_TRAINING=1
export PYTHONPATH="/tmp/ppu_compat:${PPU_REPO_ROOT}:${PPU_TORCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_PRELOAD="${PPU_COMPAT_LIB}${LD_PRELOAD:+:${LD_PRELOAD}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec /usr/bin/python "$@"
