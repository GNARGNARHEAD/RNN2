# Source this file before configuring or running the native C++ probe.
# The local Python environment supplies the SDK and headers; it does not run
# inference, differentiation, optimization, or any other model computation.
RNN2_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RNN2_PYTHON="$RNN2_ROOT/build/venv/bin/python"
if [[ ! -x "$RNN2_PYTHON" || ! -x "$RNN2_ROOT/build/venv/bin/rocm-sdk" ]]; then
  echo 'Missing local backend. See experiments/2026-11-04-gpu-probe.md.' >&2
  return 1
fi
RNN2_TORCH_ROOT="$("$RNN2_PYTHON" -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/torch")')" || return
RNN2_ROCM_ROOT="$("$RNN2_ROOT/build/venv/bin/rocm-sdk" path --root)" || return
[[ -n "$RNN2_ROCM_ROOT" && -d "$RNN2_ROCM_ROOT/lib" ]] || return 1
export ROCM_PATH="$RNN2_ROCM_ROOT"
export PYTORCH_ROCM_ARCH=gfx1035
export CMAKE_PREFIX_PATH="$RNN2_TORCH_ROOT/share/cmake;$ROCM_PATH/lib/cmake"
export LD_LIBRARY_PATH="$RNN2_TORCH_ROOT/lib:$ROCM_PATH/lib:$ROCM_PATH/lib/host-math/lib:$ROCM_PATH/lib/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XDG_CACHE_HOME="$RNN2_ROOT/build/cache"
export TMPDIR="$RNN2_ROOT/build/tmp"
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
