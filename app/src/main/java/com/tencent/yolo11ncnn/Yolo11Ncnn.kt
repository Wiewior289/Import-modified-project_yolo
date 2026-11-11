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
        const val COMPUTE_VULKAN_AHB_SPLIT = 3
        const val COMPUTE_VULKAN_AHB_ASYNC = 4
        const val COMPUTE_AHB_COMPARE = 5
        const val COMPUTE_MATMUL_VULKAN_COMPARE = 6
        const val COMPUTE_VULKAN_AHB_MATMUL_CPU = 7
        const val COMPUTE_MATMUL_VULKAN = 8

        const val ZONE_NEAR = 0
        const val ZONE_MID = 1
        const val ZONE_FAR = 2
    }

    /**
     * Initializes NCNN with CPU, Vulkan, direct Vulkan+AHB, split modes, or bounded AHB/CPU comparison mode.
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
        rotateType: Int,
        frameId: Long,
        imageTimestampNs: Long
    ): Int

    /**
     * Runs AHB import, preprocessing, GPU graph prefix (layers 1..109), and
     * boundary readback. The camera Image may be closed after this returns.
     */
    external fun PrefixFrame(
        hardwareBuffer: HardwareBuffer,
        slotIndex: Int,
        rotateType: Int,
        frameId: Long,
        imageTimestampNs: Long
    ): Int

    /**
     * Runs the CPU-only suffix (layers 110..256) from the selected frame slot.
     */
    external fun SuffixFrame(
        slotIndex: Int,
        outBuffer: FloatArray,
        confThresh: Float,
        nmsThresh: Float
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
