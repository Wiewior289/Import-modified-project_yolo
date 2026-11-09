#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NCNN_SRC="${1:-}"
if [[ -z "${NCNN_SRC}" ]]; then
    echo "Usage: $0 /path/to/ncnn-source [path/to/android-ndk]" >&2
    exit 2
fi
NCNN_SRC="$(cd "${NCNN_SRC}" && pwd)"

NDK_ROOT="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-${2:-}}}"
if [[ -z "${NDK_ROOT}" || ! -f "${NDK_ROOT}/build/cmake/android.toolchain.cmake" ]]; then
    echo "ERROR: set ANDROID_NDK_HOME or pass the Android NDK directory as argument 2." >&2
    exit 2
fi
NDK_ROOT="$(cd "${NDK_ROOT}" && pwd)"

ABI="${ANDROID_ABI:-arm64-v8a}"
API="${ANDROID_API:-29}"
BUILD_DIR="${NCNN_BUILD_DIR:-${REPO_ROOT}/.build/ncnn-${ABI}-ahb-patched}"
INSTALL_PREFIX="${NCNN_INSTALL_PREFIX:-${REPO_ROOT}/app/src/main/cpp/ncnn-android-vulkan/${ABI}}"

if [[ ! -f "${NCNN_SRC}/CMakeLists.txt" ]]; then
    echo "ERROR: ${NCNN_SRC} does not look like an NCNN source checkout." >&2
    exit 2
fi

"${SCRIPT_DIR}/apply_ncnn_ahb_normalized_sampling.sh" "${NCNN_SRC}"

cmake -S "${NCNN_SRC}" -B "${BUILD_DIR}" \
    -DCMAKE_TOOLCHAIN_FILE="${NDK_ROOT}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="${ABI}" \
    -DANDROID_PLATFORM="android-${API}" \
    -DANDROID_STL=c++_shared \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DNCNN_VULKAN=ON \
    -DNCNN_BENCHMARK=OFF \
    -DNCNN_BUILD_BENCHMARK=OFF \
    -DNCNN_BUILD_TESTS=OFF \
    -DNCNN_BUILD_TOOLS=OFF \
    -DNCNN_BUILD_EXAMPLES=OFF \
    -DNCNN_INSTALL_SDK=ON

cmake --build "${BUILD_DIR}" --parallel
cmake --install "${BUILD_DIR}"

if [[ ! -f "${INSTALL_PREFIX}/lib/cmake/ncnn/ncnnConfig.cmake" ]]; then
    echo "ERROR: NCNN CMake package was not installed where app/CMakeLists expects it:" >&2
    echo "       ${INSTALL_PREFIX}/lib/cmake/ncnn/ncnnConfig.cmake" >&2
    exit 4
fi

echo "OK: patched NCNN installed."
echo "ncnn_DIR=${INSTALL_PREFIX}/lib/cmake/ncnn"
echo "Next: rm -rf app/.cxx app/build && ./gradlew clean assembleDebug --no-configuration-cache"
