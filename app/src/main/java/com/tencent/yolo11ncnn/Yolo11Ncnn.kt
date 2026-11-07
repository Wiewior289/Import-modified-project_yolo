package com.tencent.yolo11ncnn

import android.content.res.AssetManager
import android.hardware.HardwareBuffer

/**
 * ============================================================================
 * Yolo11Ncnn - Kotlin JNI Wrapper for selectable CPU / Vulkan / AHB paths
 * Target: Samsung Galaxy S9+ (Mali-G72 MP18 + Exynos 9810)
 * ============================================================================
 */
class Yolo11Ncnn {

    companion object {
        init {
            System.loadLibrary("yolo11ncnn")
        }

        const val MODEL_YOLO11N = 0
        const val MODEL_YOLO11S = 1
        const val MODEL_YOLO11M = 2

        const val COMPUTE_CPU = 0
        const val COMPUTE_VULKAN = 1
        const val COMPUTE_VULKAN_AHB = 2

        const val ZONE_NEAR = 0
        const val ZONE_MID = 1
        const val ZONE_FAR = 2
    }

    /**
     * Initializes NCNN with CPU, Vulkan, or direct Vulkan+AHB input mode.
     */
    external fun Init(
        mgr: AssetManager,
        modelTypeIndex: Int,
        targetSize: Int,
        computeMode: Int
    ): Boolean

    /**
     * Frame input method. The native backend selects CPU-mapped YUV, CPU-mapped
     * YUV with Vulkan inference, or direct Vulkan AHardwareBuffer import.
     * Directly fills pre-allocated [outBuffer] with detections from [hardwareBuffer].
     * @return Number of detected objects.
     */
    external fun DetectFrame(
        hardwareBuffer: HardwareBuffer,
        outBuffer: FloatArray,
        confThresh: Float,
        nmsThresh: Float,
        rotateType: Int = 1
    ): Int

    /**
     * Returns the highest raw prediction score found in the previous frame.
     */
    external fun GetTopScore(): Float

    /**
     * Returns the class ID corresponding to the highest raw prediction score.
     */
    external fun GetTopClass(): Int

    /**
     * Releases cached Vulkan pipelines and GPU textures.
     */
    external fun ClearCache()

    /**
     * Releases all native C++ and Vulkan resources.
     */
    external fun Destroy()
}
