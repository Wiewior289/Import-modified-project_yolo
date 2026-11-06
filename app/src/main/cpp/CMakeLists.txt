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

add_library(yolo11ncnn SHARED
    yolo11.cpp
    yolo11ncnn_jni.cpp
)

target_include_directories(yolo11ncnn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(yolo11ncnn PRIVATE
    ncnn
    vulkan
    android
    log
    jnigraphics
)
