cmake_minimum_required(VERSION 3.22.1)
project(yolo11ncnn)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for Samsung Exynos 9810 (ARM64-v8a Bifrost)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fvisibility=hidden -fvisibility-inlines-hidden -fopenmp")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fopenmp")

# 16KB Page Size Support for Android 15+ compatibility
add_link_options("-Wl,-z,max-page-size=16384")

# Locate ncnn Vulkan package
# The ncnn-android-vulkan directory should be placed in:
# 1) app/src/main/cpp/ncnn-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn
# OR 2) app/src/main/jni/ncnn/${ANDROID_ABI}/lib/cmake/ncnn
# OR 3) Passed via -Dncnn_DIR=/path/to/cmake/ncnn
if(NOT DEFINED ncnn_DIR)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ncnn-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn")
        set(ncnn_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ncnn-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../jni/ncnn/${ANDROID_ABI}/lib/cmake/ncnn")
        set(ncnn_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../jni/ncnn/${ANDROID_ABI}/lib/cmake/ncnn")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ncnn/${ANDROID_ABI}/lib/cmake/ncnn")
        set(ncnn_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ncnn/${ANDROID_ABI}/lib/cmake/ncnn")
    else()
        # Default fallback path
        set(ncnn_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ncnn-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn")
    endif()
endif()

message(STATUS "Looking for ncnn package at: ${ncnn_DIR}")
find_package(ncnn REQUIRED)

# Refuse stale or unknown NCNN binaries. The pinned builder writes this marker
# beside ncnnConfig.cmake only after building the exact clean upstream commit.
get_filename_component(YOLO_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../" ABSOLUTE)
set(NCNN_PIN_FILE "${YOLO_REPO_ROOT}/ncnn.commit")
if(NOT EXISTS "${NCNN_PIN_FILE}")
    message(FATAL_ERROR "Missing ${NCNN_PIN_FILE}; expected pinned Tencent/ncnn source SHA.")
endif()
file(READ "${NCNN_PIN_FILE}" NCNN_EXPECTED_SHA)
string(STRIP "${NCNN_EXPECTED_SHA}" NCNN_EXPECTED_SHA)
string(LENGTH "${NCNN_EXPECTED_SHA}" NCNN_EXPECTED_SHA_LENGTH)
if(NOT NCNN_EXPECTED_SHA_LENGTH EQUAL 40 OR NOT NCNN_EXPECTED_SHA MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "Invalid NCNN SHA in ${NCNN_PIN_FILE}: '${NCNN_EXPECTED_SHA}'")
endif()

set(NCNN_BUILD_MARKER "${ncnn_DIR}/ncnn-source-sha.txt")
if(NOT EXISTS "${NCNN_BUILD_MARKER}")
    message(FATAL_ERROR "NCNN package has no source-SHA marker: ${NCNN_BUILD_MARKER}. Run scripts/build_pinned_ncnn_android.sh.")
endif()
file(READ "${NCNN_BUILD_MARKER}" NCNN_BUILT_SHA)
string(STRIP "${NCNN_BUILT_SHA}" NCNN_BUILT_SHA)
if(NOT NCNN_BUILT_SHA STREQUAL NCNN_EXPECTED_SHA)
    message(FATAL_ERROR "NCNN package SHA mismatch. Required=${NCNN_EXPECTED_SHA}; package=${NCNN_BUILT_SHA}. Rebuild pinned clean NCNN.")
endif()
message(STATUS "Verified pinned Tencent/ncnn base SHA: ${NCNN_BUILT_SHA}")

# Verify the installed NCNN library was built with exactly this branch's local patchset.
set(NCNN_PATCHSET_DIR "${YOLO_REPO_ROOT}/patches/ncnn")
set(NCNN_PATCH_HEADER "${NCNN_PATCHSET_DIR}/matmul_vulkan.h")
set(NCNN_PATCH_SOURCE "${NCNN_PATCHSET_DIR}/matmul_vulkan.cpp")
set(NCNN_PATCH_SHADER "${NCNN_PATCHSET_DIR}/matmul.comp")
set(NCNN_PATCH_BENCHMARK "${NCNN_PATCHSET_DIR}/benchmark_timestamp_safety.patch")
set(NCNN_PATCH_BENCHMARK_RUNTIME "${NCNN_PATCHSET_DIR}/benchmark_runtime_gating.patch")
set(NCNN_PATCH_SYNC "${NCNN_PATCHSET_DIR}/runtime_sync_timing.patch")
set(NCNN_PATCH_ASYNC "${NCNN_PATCHSET_DIR}/async_vk_submit.patch")
foreach(PATCH_FILE IN ITEMS "${NCNN_PATCH_HEADER}" "${NCNN_PATCH_SOURCE}" "${NCNN_PATCH_SHADER}" "${NCNN_PATCH_BENCHMARK}" "${NCNN_PATCH_BENCHMARK_RUNTIME}" "${NCNN_PATCH_SYNC}" "${NCNN_PATCH_ASYNC}")
    if(NOT EXISTS "${PATCH_FILE}")
        message(FATAL_ERROR "Missing required NCNN patch file: ${PATCH_FILE}")
    endif()
endforeach()
file(READ "${NCNN_PATCH_HEADER}" NCNN_PATCH_HEADER_TEXT)
file(READ "${NCNN_PATCH_SOURCE}" NCNN_PATCH_SOURCE_TEXT)
file(READ "${NCNN_PATCH_SHADER}" NCNN_PATCH_SHADER_TEXT)
file(READ "${NCNN_PATCH_BENCHMARK}" NCNN_PATCH_BENCHMARK_TEXT)
file(READ "${NCNN_PATCH_BENCHMARK_RUNTIME}" NCNN_PATCH_BENCHMARK_RUNTIME_TEXT)
file(READ "${NCNN_PATCH_SYNC}" NCNN_PATCH_SYNC_TEXT)
file(READ "${NCNN_PATCH_ASYNC}" NCNN_PATCH_ASYNC_TEXT)
string(SHA256 NCNN_EXPECTED_PATCHSET_SHA "${NCNN_PATCH_HEADER_TEXT}${NCNN_PATCH_SOURCE_TEXT}${NCNN_PATCH_SHADER_TEXT}${NCNN_PATCH_BENCHMARK_TEXT}${NCNN_PATCH_BENCHMARK_RUNTIME_TEXT}${NCNN_PATCH_SYNC_TEXT}${NCNN_PATCH_ASYNC_TEXT}")

set(NCNN_PATCHSET_MARKER "${ncnn_DIR}/ncnn-patchset-sha.txt")
if(NOT EXISTS "${NCNN_PATCHSET_MARKER}")
    message(FATAL_ERROR "NCNN package has no patchset-SHA marker: ${NCNN_PATCHSET_MARKER}. Run scripts/build_pinned_ncnn_android.sh.")
endif()
file(READ "${NCNN_PATCHSET_MARKER}" NCNN_BUILT_PATCHSET_SHA)
string(STRIP "${NCNN_BUILT_PATCHSET_SHA}" NCNN_BUILT_PATCHSET_SHA)
if(NOT NCNN_BUILT_PATCHSET_SHA STREQUAL NCNN_EXPECTED_PATCHSET_SHA)
    message(FATAL_ERROR "NCNN patchset mismatch. Required=${NCNN_EXPECTED_PATCHSET_SHA}; package=${NCNN_BUILT_PATCHSET_SHA}. Rebuild patched NCNN.")
endif()
message(STATUS "Verified NCNN patchset SHA: ${NCNN_BUILT_PATCHSET_SHA}")

add_library(yolo11ncnn SHARED
    yolo11.cpp
    yolo11ncnn_jni.cpp
)

target_include_directories(yolo11ncnn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(yolo11ncnn PRIVATE
    YOLO_NCNN_SOURCE_SHA=\"${NCNN_EXPECTED_SHA}\"
    YOLO_NCNN_PATCHSET_SHA=\"${NCNN_BUILT_PATCHSET_SHA}\"
)

target_link_libraries(yolo11ncnn PRIVATE
    ncnn
    vulkan
    android
    log
    jnigraphics
)
