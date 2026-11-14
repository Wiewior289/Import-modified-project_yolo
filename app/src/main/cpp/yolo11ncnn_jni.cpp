/**
 * JNI interface for the YOLO11 AHB engine.
 * Async mode exposes a GPU-prefix call and a CPU-suffix call. The Java layer
 * owns frame scheduling; each native call only holds a shared engine reference.
 */
#include <jni.h>
#include <android/asset_manager_jni.h>
#include <android/hardware_buffer.h>
#include <android/hardware_buffer_jni.h>
#include <android/log.h>
#include <mutex>
#include <atomic>
#include <memory>
#include <array>
#include <vector>
#include <chrono>
#include "yolo11.h"

#ifndef YOLO_NCNN_SOURCE_SHA
#define YOLO_NCNN_SOURCE_SHA "UNVERIFIED"
#endif

#ifndef YOLO_NCNN_PATCHSET_SHA
#define YOLO_NCNN_PATCHSET_SHA "UNVERIFIED"
#endif

#define TAG "Yolo11Ncnn_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static std::shared_ptr<Yolo11> g_yolo;
static std::mutex g_lock;
static constexpr int MAX_DETECTION_FLOATS = 1 + 64 * 7;
static std::atomic<bool> g_first_ahb_logged(false);
static bool g_ncnn_gpu_instance_ready = false;

static std::shared_ptr<Yolo11> acquire_engine()
{
    std::lock_guard<std::mutex> lock(g_lock);
    return g_yolo;
}

static int copy_detections(JNIEnv* env, jfloatArray out_buffer_obj,
                           Yolo11& engine, const std::vector<Object>& objects)
{
    std::array<float, MAX_DETECTION_FLOATS> flat{};
    const int count = engine.pack_detections_flat(objects, flat.data(), MAX_DETECTION_FLOATS);
    if (count < 0) return count;
    env->SetFloatArrayRegion(out_buffer_obj, 0, 1 + count * 7, flat.data());
    return count;
}

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    LOGI("JNI_OnLoad called for YOLO11 AHB Import Mali-G72 Native Module");
    LOGI("NCNN_SOURCE_SHA=%s", YOLO_NCNN_SOURCE_SHA);
    LOGI("NCNN_PATCHSET_SHA=%s", YOLO_NCNN_PATCHSET_SHA);
    const int vk_ret = ncnn::create_gpu_instance();
    if (vk_ret != 0) {
        LOGE("ncnn::create_gpu_instance failed: %d", vk_ret);
        return JNI_VERSION_1_6;
    }
    g_ncnn_gpu_instance_ready = true;
    LOGI("NCNN Vulkan instance ready; detected GPU count=%d", ncnn::get_gpu_count());
    return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM* vm, void* reserved)
{
    std::shared_ptr<Yolo11> old;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        old.swap(g_yolo);
    }
    old.reset();
    if (g_ncnn_gpu_instance_ready) {
        ncnn::destroy_gpu_instance();
        g_ncnn_gpu_instance_ready = false;
    }
}

JNIEXPORT jboolean JNICALL
Java_com_tencent_yolo11ncnn_Yolo11Ncnn_Init(
    JNIEnv* env, jobject thiz, jobject asset_manager,
    jint model_type_index, jint model_input_size, jint compute_mode)
{
    AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
    const char* modeltype = model_type_index == 1 ? "s" : (model_type_index == 2 ? "m" : "n");

    std::shared_ptr<Yolo11> engine = std::make_shared<Yolo11>();
    const int ret = engine->load(mgr, modeltype, model_input_size, static_cast<int>(compute_mode));
    if (ret != 0) {
        LOGE("YOLO11 init failed with error code: %d", ret);
        return JNI_FALSE;
    }

    {
        std::lock_guard<std::mutex> lock(g_lock);
        g_yolo = engine;
    }
    LOGI("YOLO11 initialized (model=yolo11%s model_input_size=%d compute_mode=%d)",
         modeltype, model_input_size, static_cast<int>(compute_mode));
    return JNI_TRUE;
}

JNIEXPORT jint JNICALL
Java_com_tencent_yolo11ncnn_Yolo11Ncnn_DetectFrame(
    JNIEnv* env, jobject thiz, jobject hardware_buffer_obj, jfloatArray out_buffer_obj,
    jfloat conf_thresh, jfloat nms_thresh, jint rotate_type,
    jlong frame_id, jlong image_timestamp_ns)
{
    if (!hardware_buffer_obj || !out_buffer_obj) return -1;
    std::shared_ptr<Yolo11> engine = acquire_engine();
    if (!engine || !engine->is_ready()) return -2;

    AHardwareBuffer* ahb = AHardwareBuffer_fromHardwareBuffer(env, hardware_buffer_obj);
    if (!ahb) return -3;

    if (!g_first_ahb_logged.exchange(true)) {
        AHardwareBuffer_Desc desc{};
        AHardwareBuffer_describe(ahb, &desc);
        __android_log_print(ANDROID_LOG_INFO, "NCNN_BARE_METAL",
            "=== FIRST AHB === format=0x%X size=%ux%u layers=%u stride=%u usage=0x%llX",
            desc.format, desc.width, desc.height, desc.layers, desc.stride,
            static_cast<unsigned long long>(desc.usage));
    }

    const auto jni_frame_start = std::chrono::steady_clock::now();
    AHardwareBuffer_acquire(ahb);
    static thread_local std::vector<Object> objects;
    objects.clear();
    if (objects.capacity() < 64) objects.reserve(64);

    const auto detect_start = std::chrono::steady_clock::now();
    const int ret = engine->detect_frame(ahb, conf_thresh, nms_thresh, objects,
        rotate_type, static_cast<int64_t>(frame_id), static_cast<int64_t>(image_timestamp_ns));
    const double detect_frame_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - detect_start).count();
    AHardwareBuffer_release(ahb);

    if (ret != 0) {
        if (frame_id > 0 && frame_id % 30 == 0) {
            const double jni_total_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - jni_frame_start).count();
            LOGI("JNI_FRAME_TIMING frame=%lld detect_frame_ms=%.3f array_pack_copy_ms=0.000 jni_total_ms=%.3f ret=%d",
                 static_cast<long long>(frame_id), detect_frame_ms, jni_total_ms, ret);
        }
        LOGE("DetectFrame failed: %d", ret);
        return ret;
    }

    const auto copy_start = std::chrono::steady_clock::now();
    const int count = copy_detections(env, out_buffer_obj, *engine, objects);
    const double array_pack_copy_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - copy_start).count();
    if (frame_id > 0 && frame_id % 30 == 0) {
        const double jni_total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - jni_frame_start).count();
        LOGI("JNI_FRAME_TIMING frame=%lld detect_frame_ms=%.3f array_pack_copy_ms=%.3f jni_total_ms=%.3f detections=%d",
             static_cast<long long>(frame_id), detect_frame_ms, array_pack_copy_ms, jni_total_ms, count);
    }
    return count;
}

JNIEXPORT jint JNICALL
Java_com_tencent_yolo11ncnn_Yolo11Ncnn_PrefixFrame(
    JNIEnv* env, jobject thiz, jobject hardware_buffer_obj,
    jint slot_index, jint rotate_type, jlong frame_id, jlong image_timestamp_ns)
{
    if (!hardware_buffer_obj) return -50;
    std::shared_ptr<Yolo11> engine = acquire_engine();
    if (!engine || !engine->is_ready()) return -51;

    AHardwareBuffer* ahb = AHardwareBuffer_fromHardwareBuffer(env, hardware_buffer_obj);
    if (!ahb) return -52;

    AHardwareBuffer_acquire(ahb);
    const int ret = engine->gpu_prefix_frame(
        slot_index, ahb, rotate_type,
        static_cast<int64_t>(frame_id), static_cast<int64_t>(image_timestamp_ns));
    AHardwareBuffer_release(ahb);
    return ret;
}

JNIEXPORT jint JNICALL
Java_com_tencent_yolo11ncnn_Yolo11Ncnn_FullGpuFrame(
    JNIEnv* env, jobject thiz, jobject hardware_buffer_obj,
    jint slot_index, jint rotate_type, jlong frame_id, jlong image_timestamp_ns)
{
    if (!hardware_buffer_obj) return -80;
    std::shared_ptr<Yolo11> engine = acquire_engine();
    if (!engine || !engine->is_ready()) return -81;

    AHardwareBuffer* ahb = AHardwareBuffer_fromHardwareBuffer(env, hardware_buffer_obj);
    if (!ahb) return -82;

    AHardwareBuffer_acquire(ahb);
    const int ret = engine->gpu_full_frame(
        slot_index, ahb, rotate_type,
        static_cast<int64_t>(frame_id), static_cast<int64_t>(image_timestamp_ns));
    AHardwareBuffer_release(ahb);
    return ret;
}

JNIEXPORT jint JNICALL
Java_com_tencent_yolo11ncnn_Yolo11Ncnn_WaitGpuFrame(
    JNIEnv* env, jobject thiz, jint slot_index)
{
    std::shared_ptr<Yolo11> engine = acquire_engine();
    if (!engine) return -83;
    return engine->gpu_wait_frame(slot_index);
}

JNIEXPORT jint JNICALL
Java_com_tencent_yolo11ncnn_Yolo11Ncnn_PostprocessFrame(
    JNIEnv* env, jobject thiz, jint slot_index, jfloatArray out_buffer_obj,
    jfloat conf_thresh, jfloat nms_thresh)
{
    if (!out_buffer_obj) return -90;
    std::shared_ptr<Yolo11> engine = acquire_engine();
    if (!engine || !engine->is_ready()) return -91;

    static thread_local std::vector<Object> objects;
    objects.clear();
    if (objects.capacity() < 64) objects.reserve(64);
    const int ret = engine->cpu_postprocess_frame(slot_index, conf_thresh, nms_thresh, objects);
    if (ret != 0) return ret;
    return copy_detections(env, out_buffer_obj, *engine, objects);
}

JNIEXPORT jint JNICALL
Java_com_tencent_yolo11ncnn_Yolo11Ncnn_SuffixFrame(
    JNIEnv* env, jobject thiz, jint slot_index, jfloatArray out_buffer_obj,
    jfloat conf_thresh, jfloat nms_thresh)
{
    if (!out_buffer_obj) return -60;
    std::shared_ptr<Yolo11> engine = acquire_engine();
    if (!engine || !engine->is_ready()) return -61;

    static thread_local std::vector<Object> objects;
    objects.clear();
    if (objects.capacity() < 64) objects.reserve(64);
    const int ret = engine->cpu_suffix_frame(slot_index, conf_thresh, nms_thresh, objects);
    if (ret != 0) return ret;
    return copy_detections(env, out_buffer_obj, *engine, objects);
}

JNIEXPORT jfloat JNICALL
Java_com_tencent_yolo11ncnn_Yolo11Ncnn_GetTopScore(JNIEnv* env, jobject thiz)
{
    std::shared_ptr<Yolo11> engine = acquire_engine();
    return engine ? engine->get_top_candidate_score() : 0.0f;
}

JNIEXPORT jint JNICALL
Java_com_tencent_yolo11ncnn_Yolo11Ncnn_GetTopClass(JNIEnv* env, jobject thiz)
{
    std::shared_ptr<Yolo11> engine = acquire_engine();
    return engine ? engine->get_top_candidate_class() : -1;
}

JNIEXPORT void JNICALL
Java_com_tencent_yolo11ncnn_Yolo11Ncnn_ClearCache(JNIEnv* env, jobject thiz)
{
    std::shared_ptr<Yolo11> engine = acquire_engine();
    if (engine) engine->clear_ahb_cache();
}

JNIEXPORT void JNICALL
Java_com_tencent_yolo11ncnn_Yolo11Ncnn_Destroy(JNIEnv* env, jobject thiz)
{
    std::shared_ptr<Yolo11> old;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        old.swap(g_yolo);
    }
    old.reset();
    LOGI("YOLO11 native instance released");
}

} // extern "C"
