/**
 * ============================================================================
 * JNI Interface for YOLO11 AHardwareBuffer Vulkan Import Test Harness
 * Target: Samsung Galaxy S9+ (Mali-G72 MP18)
 * ============================================================================
 */

#include <jni.h>
#include <android/asset_manager_jni.h>
#include <android/hardware_buffer.h>
#include <android/hardware_buffer_jni.h>
#include <android/log.h>
#include <mutex>
#include <atomic>
#include "yolo11.h"

#define TAG "Yolo11Ncnn_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static Yolo11* g_yolo = nullptr;
static std::mutex g_lock;

static const int MAX_DETECTION_FLOATS = 1 + 64 * 7;
static float g_flat_buffer[MAX_DETECTION_FLOATS];
static std::vector<Object> g_objects_scratchpad;
static std::atomic<bool> g_first_ahb_logged(false);
static bool g_ncnn_gpu_instance_ready = false;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    LOGI("JNI_OnLoad called for YOLO11 AHB Import Mali-G72 Native Module");

    const int vk_ret = ncnn::create_gpu_instance();
    if (vk_ret != 0) {
        // Keep the Java diagnostics server usable; Init() will report Vulkan unavailable.
        LOGE("ncnn::create_gpu_instance failed: %d", vk_ret);
        return JNI_VERSION_1_6;
    }

    g_ncnn_gpu_instance_ready = true;
    const int gpu_count = ncnn::get_gpu_count();
    LOGI("NCNN Vulkan instance ready; detected GPU count=%d", gpu_count);
    return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL
JNI_OnUnload(JavaVM* vm, void* reserved)
{
    {
        std::lock_guard<std::mutex> lock(g_lock);
        if (g_yolo) {
            delete g_yolo;
            g_yolo = nullptr;
        }
    }

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
    std::lock_guard<std::mutex> lock(g_lock);

    if (g_yolo) {
        delete g_yolo;
        g_yolo = nullptr;
    }

    g_yolo = new Yolo11();
    AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);

    const char* modeltype = "n";
    if (model_type_index == 1) {
        modeltype = "s";
    } else if (model_type_index == 2) {
        modeltype = "m";
    }

    int ret = g_yolo->load(mgr, modeltype, target_size, static_cast<int>(compute_mode));
    if (ret != 0) {
        LOGE("YOLO11 init failed with error code: %d", ret);
        delete g_yolo;
        g_yolo = nullptr;
        return JNI_FALSE;
    }

    g_objects_scratchpad.reserve(64);
    LOGI("YOLO11 initialized successfully (Model: yolo11%s, Size: %d, compute_mode=%d)",
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
    jint rotate_type)
{
    if (!hardware_buffer_obj || !out_buffer_obj) {
        return -1;
    }

    std::lock_guard<std::mutex> lock(g_lock);

    if (!g_yolo || !g_yolo->is_ready()) {
        return -2;
    }

    AHardwareBuffer* ahb = AHardwareBuffer_fromHardwareBuffer(env, hardware_buffer_obj);                                                           
    if (!ahb) {
        LOGE("Failed to get AHardwareBuffer from Java object");
        return -3;
    }                                                                           
                                                                                
    // Pierwszy Log Diagnostyczny
    if (!g_first_ahb_logged.exchange(true)) {                                   
        AHardwareBuffer_Desc desc;                                              
        AHardwareBuffer_describe(ahb, &desc);                                   
                                                                                
        __android_log_print(ANDROID_LOG_INFO, "NCNN_BARE_METAL",                
            "=== ANALIZA PIERWSZEGO AHB COUPLING ===\n"                         
            "Format sprzetowy (HEX): 0x%X\n"                                    
            "Wymiary krzemu: %ux%u | Warstwy: %u\n"                             
            "Stride (Wyrownanie pamieci w bajtach): %u\n"                       
            "Usage Flags: 0x%llX\n"                                             
            "Czy CPU ma dostep do odczytu?: %s\n"                               
            "Czy GPU ma dostep bezposredni?: %s",                               
            desc.format, desc.width, desc.height, desc.layers, desc.stride,     
            (unsigned long long)desc.usage,                                                 
            (desc.usage & AHARDWAREBUFFER_USAGE_CPU_READ_MASK) ? "CPU_READ usage enabled" : "CPU_READ usage absent",
            (desc.usage & AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE) ? "GPU_SAMPLED_IMAGE enabled" : "GPU_SAMPLED_IMAGE absent"                                                             
        );                                                                      
    }                                                                           
                                                                                
    AHardwareBuffer_acquire(ahb);                                               
                                                                                
    g_objects_scratchpad.clear();                                               
    int ret = g_yolo->detect_frame(ahb, conf_thresh, nms_thresh, g_objects_scratchpad, rotate_type);                                         
    AHardwareBuffer_release(ahb);                                               
                                                                                
    if (ret != 0) {
        LOGE("detect_ahb failed with code %d", ret);
        g_flat_buffer[0] = 0.0f;
        return ret;
    }

    int count = g_yolo->pack_detections_flat(g_objects_scratchpad, g_flat_buffer, MAX_DETECTION_FLOATS);

    const int floats_to_copy = 1 + count * 7;
    env->SetFloatArrayRegion(out_buffer_obj, 0, floats_to_copy, g_flat_buffer);

    return count;
}

JNIEXPORT jfloat JNICALL
Java_com_tencent_yolo11ncnn_Yolo11Ncnn_GetTopScore(JNIEnv* env, jobject thiz)
{
    std::lock_guard<std::mutex> lock(g_lock);
    if (!g_yolo) return 0.0f;
    return g_yolo->get_top_candidate_score();
}

JNIEXPORT jint JNICALL
Java_com_tencent_yolo11ncnn_Yolo11Ncnn_GetTopClass(JNIEnv* env, jobject thiz)
{
    std::lock_guard<std::mutex> lock(g_lock);
    if (!g_yolo) return -1;
    return g_yolo->get_top_candidate_class();
}

JNIEXPORT void JNICALL
Java_com_tencent_yolo11ncnn_Yolo11Ncnn_Destroy(JNIEnv* env, jobject thiz)
{
    std::lock_guard<std::mutex> lock(g_lock);
    if (g_yolo) {
        delete g_yolo;
        g_yolo = nullptr;
    }
    LOGI("YOLO11 native instance destroyed");
}

} // extern "C"
