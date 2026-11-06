package com.tencent.yolo11ncnn

import android.content.res.AssetManager
import android.hardware.HardwareBuffer

/**
 * ============================================================================
 * Yolo11Ncnn - Kotlin JNI Wrapper for Mali-G72 Zero-Copy Inference
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

        const val ZONE_NEAR = 0
        const val ZONE_MID = 1
        const val ZONE_FAR = 2
    }

    /**
     * Initializes NCNN, compiles Vulkan pipelines, and loads models from Android assets.
     */
    external fun Init(
        mgr: AssetManager,
        modelTypeIndex: Int,
        targetSize: Int,
        useGpu: Boolean
    ): Boolean

    /**
     * ULTRA ZERO-COPY & ZERO-GC METHOD:
     * Directly fills pre-allocated [outBuffer] with detections from [hardwareBuffer].
     * @return Number of detected objects.
     */
    external fun DetectAHBDirect(
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
