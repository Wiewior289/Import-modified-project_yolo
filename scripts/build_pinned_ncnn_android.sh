#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXPECTED_SHA="$(tr -d '[:space:]' < "${REPO_ROOT}/ncnn.commit")"
NCNN_URL="https://github.com/Tencent/ncnn.git"

if [[ ! "${EXPECTED_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: ncnn.commit must contain one 40-character lowercase Git SHA." >&2
    exit 2
fi

NDK_ROOT="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-${1:-}}}"
if [[ -z "${NDK_ROOT}" || ! -f "${NDK_ROOT}/build/cmake/android.toolchain.cmake" ]]; then
    echo "ERROR: set ANDROID_NDK_HOME / ANDROID_NDK_ROOT or pass the Android NDK path as argument 1." >&2
    exit 2
fi
NDK_ROOT="$(cd "${NDK_ROOT}" && pwd)"

ABI="${ANDROID_ABI:-arm64-v8a}"
API="${ANDROID_API:-29}"
SOURCE_DIR="${REPO_ROOT}/.deps/ncnn-${EXPECTED_SHA}"
BUILD_DIR="${REPO_ROOT}/.build/ncnn-${ABI}-${EXPECTED_SHA}"
INSTALL_PREFIX="${REPO_ROOT}/app/src/main/cpp/ncnn-android-vulkan/${ABI}"

case "${ABI}" in
    arm64-v8a) ;;
    *) echo "ERROR: this app currently packages arm64-v8a only; got ABI=${ABI}." >&2; exit 2 ;;
esac

if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
    mkdir -p "$(dirname "${SOURCE_DIR}")"
    git clone --filter=blob:none --no-checkout "${NCNN_URL}" "${SOURCE_DIR}"
fi

REMOTE_URL="$(git -C "${SOURCE_DIR}" remote get-url origin)"
case "${REMOTE_URL}" in
    *github.com/Tencent/ncnn*|*github.com:tencent/ncnn*) ;;
    *)
        echo "ERROR: managed source directory has unexpected origin: ${REMOTE_URL}" >&2
        exit 3
        ;;
esac

# Reset only the dedicated .deps checkout used by this script.
git -C "${SOURCE_DIR}" fetch --no-tags origin "${EXPECTED_SHA}"
git -C "${SOURCE_DIR}" checkout --detach "${EXPECTED_SHA}"
git -C "${SOURCE_DIR}" reset --hard "${EXPECTED_SHA}"
git -C "${SOURCE_DIR}" clean -fdx

ACTUAL_SHA="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]]; then
    echo "ERROR: NCNN checkout mismatch. expected=${EXPECTED_SHA} actual=${ACTUAL_SHA}" >&2
    exit 4
fi
if [[ -n "$(git -C "${SOURCE_DIR}" status --porcelain)" ]]; then
    echo "ERROR: NCNN checkout is not clean after reset." >&2
    git -C "${SOURCE_DIR}" status --short >&2
    exit 4
fi

# Apply the repository-owned MatMul Vulkan extension after every pristine checkout.
# The upstream NCNN SHA remains pinned; the local patchset is identified separately.
PATCH_DIR="${REPO_ROOT}/patches/ncnn"
PATCH_HEADER="${PATCH_DIR}/matmul_vulkan.h"
PATCH_SOURCE="${PATCH_DIR}/matmul_vulkan.cpp"
PATCH_SHADER="${PATCH_DIR}/matmul.comp"
PATCH_BENCHMARK="${PATCH_DIR}/benchmark_timestamp_safety.patch"
PATCH_BENCHMARK_RUNTIME="${PATCH_DIR}/benchmark_runtime_gating.patch"
PATCH_SYNC="${PATCH_DIR}/runtime_sync_timing.patch"
PATCH_ASYNC="${PATCH_DIR}/async_vk_submit.patch"

for patch_file in "${PATCH_HEADER}" "${PATCH_SOURCE}" "${PATCH_SHADER}" "${PATCH_BENCHMARK}" "${PATCH_BENCHMARK_RUNTIME}" "${PATCH_SYNC}" "${PATCH_ASYNC}"; do
    if [[ ! -s "${patch_file}" ]]; then
        echo "ERROR: required MatMul Vulkan patch file is missing or empty: ${patch_file}" >&2
        exit 4
    fi
done

install -D -m 0644 "${PATCH_HEADER}" "${SOURCE_DIR}/src/layer/vulkan/matmul_vulkan.h"
install -D -m 0644 "${PATCH_SOURCE}" "${SOURCE_DIR}/src/layer/vulkan/matmul_vulkan.cpp"
install -D -m 0644 "${PATCH_SHADER}" "${SOURCE_DIR}/src/layer/vulkan/shader/matmul.comp"

# Apply NCNN runtime instrumentation only to the pristine pinned source. The patch
# defers query reads until the final fence, preserves query IDs across partial
# submits, validates query handles/ranges, and emits MatMul CPU/Vulkan route logs.
git -C "${SOURCE_DIR}" apply --check "${PATCH_BENCHMARK}"
git -C "${SOURCE_DIR}" apply "${PATCH_BENCHMARK}"
git -C "${SOURCE_DIR}" apply --check "${PATCH_BENCHMARK_RUNTIME}"
git -C "${SOURCE_DIR}" apply "${PATCH_BENCHMARK_RUNTIME}"
git -C "${SOURCE_DIR}" apply --check "${PATCH_SYNC}"
git -C "${SOURCE_DIR}" apply "${PATCH_SYNC}"
git -C "${SOURCE_DIR}" apply --unidiff-zero --check "${PATCH_ASYNC}"
git -C "${SOURCE_DIR}" apply --unidiff-zero "${PATCH_ASYNC}"

PATCHSET_SHA="$(cat "${PATCH_HEADER}" "${PATCH_SOURCE}" "${PATCH_SHADER}" "${PATCH_BENCHMARK}" "${PATCH_BENCHMARK_RUNTIME}" "${PATCH_SYNC}" "${PATCH_ASYNC}" | sha256sum | awk '{print $1}')"
if [[ ! "${PATCHSET_SHA}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ERROR: failed to calculate NCNN patchset SHA." >&2
    exit 4
fi

echo "NCNN_SOURCE=${SOURCE_DIR}"
echo "NCNN_SOURCE_SHA=${ACTUAL_SHA}"
echo "NCNN_SOURCE_STATUS=PINNED_BASE_WITH_LOCAL_PATCHSET"
echo "NCNN_PATCHES=MATMUL_VULKAN_FP32_ACCUMULATE+BENCHMARK_TIMESTAMP_SAFETY+RUNTIME_BENCHMARK_GATING+RUNTIME_SYNC_TIMING+ASYNC_VK_SUBMIT"
echo "NCNN_PATCHSET_SHA=${PATCHSET_SHA}"

# Remove previous generated outputs so no old library can survive the reset.
rm -rf "${BUILD_DIR}" "${INSTALL_PREFIX}"
mkdir -p "${BUILD_DIR}" "$(dirname "${INSTALL_PREFIX}")"

cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_TOOLCHAIN_FILE="${NDK_ROOT}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="${ABI}" \
    -DANDROID_PLATFORM="android-${API}" \
    -DANDROID_STL=c++_shared \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DNCNN_VULKAN=ON \
    -DNCNN_BATCH=ON \
    -DNCNN_BENCHMARK=ON \
    -DNCNN_BUILD_BENCHMARK=OFF \
    -DNCNN_BUILD_TESTS=OFF \
    -DNCNN_BUILD_TOOLS=OFF \
    -DNCNN_BUILD_EXAMPLES=OFF \
    -DNCNN_INSTALL_SDK=ON

cmake --build "${BUILD_DIR}" --parallel
cmake --install "${BUILD_DIR}"

NCNN_CONFIG_DIR="${INSTALL_PREFIX}/lib/cmake/ncnn"
if [[ ! -f "${NCNN_CONFIG_DIR}/ncnnConfig.cmake" ]]; then
    echo "ERROR: NCNN CMake package was not installed at ${NCNN_CONFIG_DIR}." >&2
    exit 5
fi

printf '%s\n' "${EXPECTED_SHA}" > "${NCNN_CONFIG_DIR}/ncnn-source-sha.txt"
INSTALLED_SHA="$(tr -d '[:space:]' < "${NCNN_CONFIG_DIR}/ncnn-source-sha.txt")"
if [[ "${INSTALLED_SHA}" != "${EXPECTED_SHA}" ]]; then
    echo "ERROR: installed NCNN marker mismatch." >&2
    exit 6
fi

printf '%s\n' "${PATCHSET_SHA}" > "${NCNN_CONFIG_DIR}/ncnn-patchset-sha.txt"
INSTALLED_PATCHSET_SHA="$(tr -d '[:space:]' < "${NCNN_CONFIG_DIR}/ncnn-patchset-sha.txt")"
if [[ "${INSTALLED_PATCHSET_SHA}" != "${PATCHSET_SHA}" ]]; then
    echo "ERROR: installed NCNN patchset marker mismatch." >&2
    exit 6
fi

# Discard Gradle/CMake caches that could still point at an older NCNN package.
rm -rf "${REPO_ROOT}/app/.cxx" "${REPO_ROOT}/app/build"

echo "OK: pinned NCNN base + MatMul Vulkan patchset installed."
echo "NCNN_SOURCE_SHA=${INSTALLED_SHA}"
echo "NCNN_PATCHSET_SHA=${INSTALLED_PATCHSET_SHA}"
echo "ncnn_DIR=${NCNN_CONFIG_DIR}"
echo "Next:"
echo "  rm -rf app/.cxx app/build"
echo "  ./gradlew clean assembleDebug --no-configuration-cache"
