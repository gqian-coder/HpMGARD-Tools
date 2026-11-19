#!/bin/bash
# Build the tools against an installed CompressHpMGARDOperator (CPU build).
#   HPMGARD_PREFIX=/path/to/CompressHpMGARDOperator/install-cpu bash build_scripts_cpu.sh [build_dir]
set -e
# The Cray wrapper auto-links libsci (static TLS: breaks dlopen of the plugin) and darshan (MPI-only);
# neither is needed, so unload them for the whole configure + build shell.
if type module >/dev/null 2>&1; then module unload cray-libsci darshan-runtime 2>/dev/null || true; fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${HPMGARD_PREFIX:-${HERE}/../CompressHpMGARDOperator/install-cpu}"
BUILD="${1:-${HERE}/build_cpu}"
MGARD_DIR="${MGARD_DIR:-/lustre/orion/cfd164/proj-shared/gongq/Software/MGARD-serial/install-serial}"
cmake -S "${HERE}" -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER:-cc}" -DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER:-CC}" \
  -Dhpmgard_DIR="${PREFIX}/lib64/cmake/hpmgard" -Dmgard_DIR="${MGARD_DIR}/lib64/cmake/mgard" \
  -DCMAKE_PREFIX_PATH="${PREFIX};${MGARD_DIR}" \
  -DHPMT_BUILD_ADIOS2_TOOLS="${HPMT_BUILD_ADIOS2_TOOLS:-ON}" -DADIOS2_DIR="${ADIOS2_DIR:-/lustre/orion/cfd164/proj-shared/gongq/Software/ADIOS2/install-adios-serial}/lib64/cmake/adios2" \
  -DHPMGARD_PLUGIN_DIR="${PREFIX}/lib64" \
  -DCMAKE_BUILD_RPATH="${MGARD_DIR}/lib64;${MGARD_DIR}/lib;${ADIOS2_DIR:-/lustre/orion/cfd164/proj-shared/gongq/Software/ADIOS2/install-adios-serial}/lib64"
cmake --build "${BUILD}" -j16
echo "built in ${BUILD}; run tests with: ctest --test-dir ${BUILD} --output-on-failure"
