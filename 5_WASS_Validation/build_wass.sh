#!/usr/bin/env bash
# Build the WASS C++ pipeline (wass/ submodule) into wass_bin/, using the dependencies from shell.nix.
# Run from anywhere, outside nix-shell. Re-run after updating the wass submodule or after nix garbage collection.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

if [ ! -f wass/src/CMakeLists.txt ] || [ -z "$(ls -A wass/ext/incfg 2>/dev/null)" ]; then
  echo "wass submodule missing; fetching it..."
  git submodule update --init --recursive wass
fi

# Keep the shell's dependencies out of `nix-collect-garbage`: the executables link against them.
nix-build shell.nix -A inputDerivation -o .nix-gcroot-wass >/dev/null

# Note: don't use wass/build.sh, it deletes wass/ext (which holds the incfg submodule).
# WASS's CMake downloads and compiles its SBA dependency into wass/ext/sba-1.6 on first build (git-ignored there).
nix-shell shell.nix --run '
  set -euo pipefail
  rm -rf wass_bin/build
  cmake -S wass/src -B wass_bin/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD/wass_bin"
  cmake --build wass_bin/build -j"$(nproc)"
  cmake --install wass_bin/build
'

echo
echo "Built: $(ls wass_bin/bin | tr '\n' ' ')"
echo "Run 'nix-shell' in $here to use them."
