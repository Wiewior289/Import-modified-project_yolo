/**
 * JNI Interface for YOLO11: synchronous baseline plus split AHB frame stages.
 */

#include <jni.h>
#include <android/asset_manager_jni.h>
#include <android/hardware_buffer.h>
#include <android/hardware_buffer_jni.h>
#include <android/log.h>

#include <algorithm>
#include <array>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

#include "yolo11.h"

#define TAG "Yolo11Ncnn_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static constexpr int MAX_DETECTION_FLOATS = 1 + 64 * 7;

static std::shared_ptr<Yolo11> g_yolo;
static std::mutex g_lock;               // Protects only the engine handle/lifecycle.
static std::mutex g_sync_detect_lock;   // Legacy full-frame path uses engine scratchpads.
static std::mutex g_pending_lock;
static std::unordered_map<jlong, std::shared_ptr<Yolo11>> g_pending_engines;
static bool g_ncnn_gpu_instance_ready = false;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    LOGI("JNI_OnLoad called for YOLO11 AHB frame-overlap engine");

    const int vk_ret = ncnn::create_gpu_instance();
    if (vk_ret != 0) {
        LOGE("ncnn::create_gpu_instance failed: %d", vk_ret);
        return JNI_VERSION_1_6;
    }

    g_ncnn_gpu_instance_ready = true;
    LOGI("NCNN Vulkan instance ready; detected GPU count=%d", ncnn::get_gpu_count());
    return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL
JNI_OnUnload(JavaVM* vm, void* reserved)
{
    {
        std::lock_guard<std::mutex> lock(g_pending_lock);
        g_pending_engines.clear();
    }

    std::shared_ptr<Yolo11> old_engine;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        old_engine = std::move(g_yolo);
    }
    old_engine.reset();

    if (g_ncnn_gpu_instance_ready) {
        ncnn::destroy_gpu_instance();
        g_ncnn_gpu_instance_ready = false;
    }
}

JNIEXPORT jboolean JNICALL
Java_com_tencent_yolo11ncnn_Yolo11Ncnn_Init(
    JNIEnv* env,
    jobject thiz,
    jobject asset_manager,
    jint model_type_index,
    jint target_size,
    jint compute_mode)
{
    {
        std::lock_guard<std::mutex> pending_lock(g_pending_lock);
        if (!g_pending_engines.empty()) {
            LOGE("Init rejected: %zu async frame(s) are still pending",
                 g_pending_engines.size());
            return JNI_FALSE;
        }
    }

    std::lock_guard<std::mutex> lifecycle_lock(g_lock);

    AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
    if (!mgr) {
        LOGE("AAssetManager_fromJava returned null");
        return JNI_FALSE;
    }

    const char* modeltype = "n";
    if (model_type_index == 1) modeltype = "s";
    else if (model_type_index == 2) modeltype = "m";

    std::shared_ptr<Yolo11> next_engine;
    try {
        next_engine = std::make_shared<Yolo11>();
    } catch (const std::bad_alloc&) {
        LOGE("Unable to allocate Yolo11 engine");
        return JNI_FALSE;
    }

    const int ret = next_engine->load(
        mgr, modeltype, target_size, static_cast<int>(compute_mode));
    if (ret != 0) {
        LOGE("YOLO11 init failed with error code: %d", ret);
        return JNI_FALSE;
    }

    g_yolo = std::move(next_engine);
    LOGI("YOLO11 initialized (model=yolo11%s size=%d compute_mode=%d)",
         modeltype, target_size, static_cast<int>(compute_mode));
    return JNI_TRUE;
}

JNIEXPORT jint JNICALL
Java_com_tencent_yolo11ncnn_Yolo11Ncnn_DetectFrame(
    JNIEnv* env,
    jobject thiz,
    jobject hardware_buffer_obj,
    jfloatArray out_buffer_obj,
    jfloat conf_thresh,
    jfloat nms_thresh,
    jint rotate_type,
    jlong frame_id,
    jlong image_timestamp_ns)
{
    if (!hardware_buffer_obj || !out_buffer_obj) return -1;

    std::shared_ptr<Yolo11> engine;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        engine = g_yolo;
    }
    if (!engine || !engine->is_ready()) return -2;

    std::lock_guard<std::mutex> inference_lock(g_sync_detect_lock);

    AHardwareBuffer* ahb = AHardwareBuffer_fromHardwareBuffer(env, hardware_buffer_obj);
    if (!ahb) {
        LOGE("DetectFrame: failed to get AHardwareBuffer from Java object");
        return -3;
    }

    AHardwareBuffer_acquire(ahb);
    std::vector<Object> objects;
    objects.reserve(64);
    const int ret = engine->detect_frame(
        ahb, conf_thresh, nms_thresh, objects, rotate_type,
        static_cast<int64_t>(frame_id), static_cast<int64_t>(image_timestamp_ns));
    AHardwareBuffer_release(ahb);

    if (ret != 0) {
        LOGE("DetectFrame failed frame=%lld code=%d", (long long)frame_id, ret);
        return ret;
    }

    const jsize output_capacity = env->GetArrayLength(out_buffer_obj);
    if (output_capacity < 1) return -4;

    std::array<float, MAX_DETECTION_FLOATS> flat{};
    const int count = engine->pack_detections_flat(
        objects, flat.data(), std::min<int>(MAX_DETECTION_FLOATS, output_capacity));
    const int floats_to_copy = 1 + count * 7;
    env->SetFloatArrayRegion(out_buffer_obj, 0, floats_to_copy, flat.data());
    return count;
}

JNIEXPORT jlong JNICALL
Java_com_tencent_yolo11ncnn_Yolo11Ncnn_BeginFrame(
    JNIEnv* env,
    jobject thiz,
    jobject hardware_buffer_obj,
    jint rotate_type,
    jlong frame_id,
    jlong image_timestamp_ns)
{
    if (!hardware_buffer_obj || frame_id <= 0) return -1;

    std::shared_ptr<Yolo11> engine;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        engine = g_yolo;
    }
    if (!engine || !engine->is_ready()) return -2;

    {
        std::lock_guard<std::mutex> lock(g_pending_lock);
        if (g_pending_engines.find(frame_id) != g_pending_engines.end()) {
            LOGE("BeginFrame rejected duplicate frame id=%lld", (long long)frame_id);
            return -3;
        }
    }

    AHardwareBuffer* ahb = AHardwareBuffer_fromHardwareBuffer(env, hardware_buffer_obj);
    if (!ahb) return -4;

    AHardwareBuffer_acquire(ahb);
    const int ret = engine->begin_async_frame(
        ahb, rotate_type, static_cast<int64_t>(frame_id),
        static_cast<int64_t>(image_timestamp_ns));
    AHardwareBuffer_release(ahb);

    if (ret != 0) {
        LOGE("BeginFrame GPU prefix failed frame=%lld code=%d",
             (long long)frame_id, ret);
        return static_cast<jlong>(ret);
    }

    {
        std::lock_guard<std::mutex> lock(g_pending_lock);
        const auto inserted = g_pending_engines.emplace(frame_id, engine);
        if (!inserted.second) {
            LOGE("BeginFrame pending registry collision frame=%lld", (long long)frame_id);
            // The Yolo11 context is intentionally retained for explicit engine
            // teardown; this collision should be impossible with monotonic IDs.
            return -5;
        }
    }

    return frame_id;
}

JNIEXPORT jint JNICALL
Java_com_tencent_yolo11ncnn_Yolo11Ncnn_FinishFrame(
    JNIEnv* env,
    jobject thiz,
    jlong frame_id,
    jfloatArray out_buffer_obj,
    jfloat conf_thresh,
    jfloat nms_thresh)
{
    if (!out_buffer_obj || frame_id <= 0) return -1;

    std::shared_ptr<Yolo11> engine;
    {
        std::lock_guard<std::mutex> lock(g_pending_lock);
        auto it = g_pending_engines.find(frame_id);
        if (it == g_pending_engines.end()) {
            LOGE("FinishFrame has no pending engine for frame=%lld", (long long)frame_id);
            return -2;
        }
        engine = std::move(it->second);
        g_pending_engines.erase(it);
    }

    std::vector<Object> objects;
    objects.reserve(64);
    const int ret = engine->finish_async_frame(
        static_cast<int64_t>(frame_id), conf_thresh, nms_thresh, objects);
    if (ret != 0) {
        LOGE("FinishFrame CPU suffix failed frame=%lld code=%d",
             (long long)frame_id, ret);
        return ret;
    }

    const jsize output_capacity = env->GetArrayLength(out_buffer_obj);
    if (output_capacity < 1) return -3;

    std::array<float, MAX_DETECTION_FLOATS> flat{};
    const int count = engine->pack_detections_flat(
        objects, flat.data(), std::min<int>(MAX_DETECTION_FLOATS, output_capacity));
    const int floats_to_copy = 1 + count * 7;
    env->SetFloatArrayRegion(out_buffer_obj, 0, floats_to_copy, flat.data());
    return count;
}

JNIEXPORT void JNICALL
Java_com_tencent_yolo11ncnn_Yolo11Ncnn_CancelFrame(
    JNIEnv* env,
    jobject thiz,
    jlong frame_id)
{
    std::shared_ptr<Yolo11> engine;
    {
        std::lock_guard<std::mutex> lock(g_pending_lock);
        auto it = g_pending_engines.find(frame_id);
        if (it == g_pending_engines.end()) return;
        engine = std::move(it->second);
        g_pending_engines.erase(it);
    }
    engine->cancel_async_frame(static_cast<int64_t>(frame_id));
}

JNIEXPORT jfloat JNICALL
Java_com_tencent_yolo11ncnn_Yolo11Ncnn_GetTopScore(JNIEnv* env, jobject thiz)
{
    std::shared_ptr<Yolo11> engine;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        engine = g_yolo;
    }
    return engine ? engine->get_top_candidate_score() : 0.0f;
}

JNIEXPORT jint JNICALL
Java_com_tencent_yolo11ncnn_Yolo11Ncnn_GetTopClass(JNIEnv* env, jobject thiz)
{
    std::shared_ptr<Yolo11> engine;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        engine = g_yolo;
    }
    return engine ? engine->get_top_candidate_class() : -1;
}

JNIEXPORT void JNICALL
Java_com_tencent_yolo11ncnn_Yolo11Ncnn_ClearCache(JNIEnv* env, jobject thiz)
{
    std::shared_ptr<Yolo11> engine;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        engine = g_yolo;
    }
    if (engine) engine->clear_ahb_cache();
}

JNIEXPORT void JNICALL
Java_com_tencent_yolo11ncnn_Yolo11Ncnn_Destroy(JNIEnv* env, jobject thiz)
{
    {
        std::lock_guard<std::mutex> lock(g_pending_lock);
        if (!g_pending_engines.empty()) {
            LOGE("Destroy deferred: %zu async frame(s) are still pending",
                 g_pending_engines.size());
            return;
        }
    }

    std::shared_ptr<Yolo11> old_engine;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        old_engine = std::move(g_yolo);
    }
    old_engine.reset();
    LOGI("YOLO11 native instance destroyed");
}

} // extern "C"
