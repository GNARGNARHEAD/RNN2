#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/backend_env.sh"
cmake -S "$RNN2_ROOT" -B "$RNN2_ROOT/build/native" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release '-DCMAKE_CXX_FLAGS_RELEASE=-O1 -DNDEBUG' \
  -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH"
# A single compiler process keeps setup memory use bounded on this laptop.
cmake --build "$RNN2_ROOT/build/native" --parallel 1
