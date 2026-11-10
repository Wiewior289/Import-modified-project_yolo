/**
 * ============================================================================
 * YOLO11 NCNN AHardwareBuffer Vulkan Import Engine for Autonomous Robotic Mower
 * Platform: Samsung Galaxy S9+ (Samsung Exynos 9810 + Mali-G72 MP18 Bifrost GPU)
 * Actuator MCU: STM32F405 (UART/CAN control interface)
 * Framework: Tencent ncnn (Vulkan GPU Acceleration & AHardwareBuffer Zero-Copy)
 * ============================================================================
 */

#include "yolo11.h"

#include <android/log.h>
#include <chrono>
#include <new>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <sstream>
#include <sys/system_properties.h>
#include "blob.h"

#define TAG "Yolo11_AHB_MaliG72"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {
static int tensor_tap_property(){char v[PROP_VALUE_MAX]={};return __system_property_get("debug.yolo11.tensor_tap",v)>0?std::atoi(v):0;}
static float half_to_float(uint16_t h){
 const uint32_t sign=(uint32_t)(h&0x8000u)<<16;uint32_t exp=(h>>10)&31u,m=h&1023u,bits=0;
 if(exp==0){if(!m)bits=sign;else{int sh=0;while((m&1024u)==0){m<<=1;++sh;}m&=1023u;bits=sign|((uint32_t)(127-15-sh+1)<<23)|(m<<13);}}
 else if(exp==31)bits=sign|0x7f800000u|(m<<13);else bits=sign|((exp+112u)<<23)|(m<<13);
 float x;std::memcpy(&x,&bits,4);return x;
}
static float tensor_value(const unsigned char*p,size_t n){
 if(n==4){float x;std::memcpy(&x,p,4);return x;}
 if(n==2){uint16_t x;std::memcpy(&x,p,2);return half_to_float(x);}
 int8_t x;std::memcpy(&x,p,1);return (float)x;
}
static void log_tensor_stats(const char*name,const ncnn::Mat&m,int64_t frame,int64_t ts){
 if(m.empty()||!m.data||m.elempack<=0||!m.elemsize){LOGW("TENSOR_CPU frame=%lld ts=%lld name=%s EMPTY",(long long)frame,(long long)ts,name);return;}
 size_t n=m.total()*(size_t)m.elempack,bytes=m.elemsize/(size_t)m.elempack;
 if(!n||(bytes!=1&&bytes!=2&&bytes!=4)){LOGI("TENSOR_CPU frame=%lld ts=%lld name=%s shape=%d,%d,%d,%d elemsize=%zu pack=%d UNSUPPORTED",(long long)frame,(long long)ts,name,m.w,m.h,m.d,m.c,m.elemsize,m.elempack);return;}
 const unsigned char*p=(const unsigned char*)m.data;float mn=std::numeric_limits<float>::infinity(),mx=-mn,s[4]={};double sum=0;size_t nf=0;
 for(size_t i=0;i<n;i++){float x=tensor_value(p+i*bytes,bytes);if(!std::isfinite(x)){nf++;continue;}mn=std::min(mn,x);mx=std::max(mx,x);sum+=x;}
 size_t ix[4]={0,static_cast<size_t>(n>1?1:0),n/2,n-1};for(int i=0;i<4;i++)s[i]=tensor_value(p+ix[i]*bytes,bytes);
 size_t finite=n-nf;LOGI("TENSOR_CPU frame=%lld ts=%lld name=%s dims=%d shape=%d,%d,%d,%d elemsize=%zu pack=%d count=%zu min=%.7g max=%.7g mean=%.7g nonfinite=%zu samples=[%.7g,%.7g,%.7g,%.7g]",
 (long long)frame,(long long)ts,name,m.dims,m.w,m.h,m.d,m.c,m.elemsize,m.elempack,n,mn,mx,finite?sum/finite:0.0,nf,s[0],s[1],s[2],s[3]);
}
static void log_vkmat(const char*name,const ncnn::VkMat&m,int64_t frame,int64_t ts){
 LOGI("TENSOR_VK frame=%lld ts=%lld name=%s empty=%d dims=%d shape=%d,%d,%d,%d elemsize=%zu pack=%d total=%zu data=%p allocator=%p offset=%zu capacity=%zu",
 (long long)frame,(long long)ts,name,m.empty()?1:0,m.dims,m.w,m.h,m.d,m.c,m.elemsize,m.elempack,m.total(),
 (const void*)m.data,(const void*)m.allocator,m.buffer_offset(),m.buffer_capacity());
}

struct CompareStats {
    bool valid = false;
    size_t count = 0, finite = 0, nonfinite = 0;
    double min_value = 0.0, max_value = 0.0, mean = 0.0, stddev = 0.0;
};

static size_t tensor_scalar_count(const ncnn::Mat& m)
{
    const size_t w = static_cast<size_t>(m.w);
    const size_t h = m.dims >= 2 ? static_cast<size_t>(m.h) : 1u;
    const size_t d = m.dims >= 3 ? static_cast<size_t>(m.d) : 1u;
    const size_t c = m.dims >= 3 ? static_cast<size_t>(m.c) : 1u;
    const size_t n = m.dims == 4 ? static_cast<size_t>(m.n) : 1u;
    return w * h * d * c * n * static_cast<size_t>(m.elempack);
}

static const unsigned char* tensor_scalar_ptr(const ncnn::Mat& m, size_t scalar_index)
{
    const size_t pack = static_cast<size_t>(m.elempack);
    const size_t scalar_bytes = m.elemsize / pack;
    const size_t spatial = static_cast<size_t>(m.w) *
                           (m.dims >= 2 ? static_cast<size_t>(m.h) : 1u) *
                           (m.dims >= 3 ? static_cast<size_t>(m.d) : 1u);
    const size_t channel_scalars = spatial * pack;
    const size_t channel_index = scalar_index / channel_scalars;
    const size_t within = scalar_index % channel_scalars;
    const size_t packed_element = within / pack;
    const size_t lane = within % pack;

    size_t element_offset = packed_element;
    if (m.dims >= 3) {
        const size_t channels_per_batch = static_cast<size_t>(m.c);
        const size_t batch = channel_index / channels_per_batch;
        const size_t channel = channel_index % channels_per_batch;
        element_offset = batch * m.nstep + channel * m.cstep + packed_element;
    }
    return static_cast<const unsigned char*>(m.data) + element_offset * m.elemsize + lane * scalar_bytes;
}

static CompareStats measure_tensor(const ncnn::Mat& m)
{
    CompareStats st;
    if (m.empty() || !m.data || m.elempack <= 0 || m.elemsize == 0 ||
        (m.elemsize % static_cast<size_t>(m.elempack)) != 0) return st;
    const size_t scalar_bytes = m.elemsize / static_cast<size_t>(m.elempack);
    if (scalar_bytes != 1 && scalar_bytes != 2 && scalar_bytes != 4) return st;
    st.count = tensor_scalar_count(m);
    if (st.count == 0) return st;

    st.min_value = std::numeric_limits<double>::infinity();
    st.max_value = -std::numeric_limits<double>::infinity();
    double sum = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < st.count; ++i) {
        const float v = tensor_value(tensor_scalar_ptr(m, i), scalar_bytes);
        if (!std::isfinite(v)) { ++st.nonfinite; continue; }
        ++st.finite;
        st.min_value = std::min(st.min_value, static_cast<double>(v));
        st.max_value = std::max(st.max_value, static_cast<double>(v));
        sum += v;
        sum_sq += static_cast<double>(v) * v;
    }
    if (!st.finite) return st;
    st.mean = sum / static_cast<double>(st.finite);
    st.stddev = std::sqrt(std::max(0.0, sum_sq / static_cast<double>(st.finite) - st.mean * st.mean));
    st.valid = true;
    return st;
}

static void log_compare_tensor(const char* name, const ncnn::Mat& a, const ncnn::Mat& b,
                               int64_t frame_id, int64_t image_timestamp_ns)
{
    const CompareStats sa = measure_tensor(a), sb = measure_tensor(b);
    LOGI("AHBCMP_TENSOR frame=%lld ts=%lld name=%s A(shape=%d:%d,%d,%d,%d es=%zu pack=%d n=%zu min=%.6g max=%.6g mean=%.6g std=%.6g nf=%zu) B(shape=%d:%d,%d,%d,%d es=%zu pack=%d n=%zu min=%.6g max=%.6g mean=%.6g std=%.6g nf=%zu)",
         (long long)frame_id, (long long)image_timestamp_ns, name,
         a.dims,a.w,a.h,a.d,a.c,a.elemsize,a.elempack,sa.count,
         sa.valid?sa.min_value:0.0,sa.valid?sa.max_value:0.0,sa.valid?sa.mean:0.0,sa.valid?sa.stddev:0.0,sa.nonfinite,
         b.dims,b.w,b.h,b.d,b.c,b.elemsize,b.elempack,sb.count,
         sb.valid?sb.min_value:0.0,sb.valid?sb.max_value:0.0,sb.valid?sb.mean:0.0,sb.valid?sb.stddev:0.0,sb.nonfinite);

    const bool same_shape = a.dims == b.dims && a.w == b.w && a.h == b.h &&
                            a.d == b.d && a.c == b.c && a.n == b.n;
    if (!sa.valid || !sb.valid || sa.count != sb.count || !same_shape) {
        LOGW("AHBCMP_DIFF frame=%lld ts=%lld name=%s status=INCOMPATIBLE A_valid=%d B_valid=%d A_shape=%d:%d,%d,%d,%d,%d B_shape=%d:%d,%d,%d,%d,%d A_count=%zu B_count=%zu",
             (long long)frame_id,(long long)image_timestamp_ns,name,sa.valid?1:0,sb.valid?1:0,
             a.dims,a.w,a.h,a.d,a.c,a.n,b.dims,b.w,b.h,b.d,b.c,b.n,sa.count,sb.count);
        return;
    }
    const size_t ba=a.elemsize/static_cast<size_t>(a.elempack);
    const size_t bb=b.elemsize/static_cast<size_t>(b.elempack);
    double abs_sum=0.0,sq_sum=0.0,dot=0.0,norm_a=0.0,norm_b=0.0,max_abs=0.0;
    size_t compared=0,nonfinite=0;
    for(size_t i=0;i<sa.count;i++){
        const double va=tensor_value(tensor_scalar_ptr(a,i),ba);
        const double vb=tensor_value(tensor_scalar_ptr(b,i),bb);
        if(!std::isfinite(va)||!std::isfinite(vb)){++nonfinite;continue;}
        const double d=std::fabs(va-vb);
        abs_sum+=d;sq_sum+=d*d;max_abs=std::max(max_abs,d);
        dot+=va*vb;norm_a+=va*va;norm_b+=vb*vb;++compared;
    }
    const double mae=compared?abs_sum/compared:0.0;
    const double rmse=compared?std::sqrt(sq_sum/compared):0.0;
    const double cosine=(norm_a>0.0&&norm_b>0.0)?dot/std::sqrt(norm_a*norm_b):0.0;
    LOGI("AHBCMP_DIFF frame=%lld ts=%lld name=%s count=%zu compared=%zu nonfinite=%zu mae=%.7g rmse=%.7g max=%.7g cosine=%.9g",
         (long long)frame_id,(long long)image_timestamp_ns,name,sa.count,compared,nonfinite,mae,rmse,max_abs,cosine);
}


static void log_single_tensor(const char* name,const ncnn::Mat& m,int64_t frame_id,int64_t image_timestamp_ns)
{
    const CompareStats st=measure_tensor(m);
    LOGI("AHBCMP_SINGLE frame=%lld ts=%lld name=%s shape=%d:%d,%d,%d,%d elemsize=%zu pack=%d count=%zu min=%.6g max=%.6g mean=%.6g std=%.6g nonfinite=%zu valid=%d",
         (long long)frame_id,(long long)image_timestamp_ns,name,
         m.dims,m.w,m.h,m.d,m.c,m.elemsize,m.elempack,st.count,
         st.valid?st.min_value:0.0,st.valid?st.max_value:0.0,
         st.valid?st.mean:0.0,st.valid?st.stddev:0.0,st.nonfinite,st.valid?1:0);
}

static void log_rgb_samples(const ncnn::Mat& a, const ncnn::Mat& b,
                            int64_t frame_id, int64_t image_timestamp_ns)
{
    if(a.empty()||b.empty()||a.w!=b.w||a.h!=b.h||a.c<3||b.c<3||a.elempack!=1||b.elempack!=1){
        LOGW("AHBCMP_RGB_SAMPLES frame=%lld ts=%lld status=UNAVAILABLE",(long long)frame_id,(long long)image_timestamp_ns);return;
    }
    const int w=a.w,h=a.h;
    const int xy[8][2]={{0,0},{w/4,h/4},{w/2,h/2},{3*w/4,h/4},{w/4,3*h/4},{3*w/4,3*h/4},{w-1,h-1},{w/2,h/4}};
    std::ostringstream line;line<<"AHBCMP_RGB_SAMPLES frame="<<frame_id<<" ts="<<image_timestamp_ns;
    for(int i=0;i<8;i++){
        const size_t p=static_cast<size_t>(xy[i][1])*w+xy[i][0];
        line<<" P"<<i<<"("<<xy[i][0]<<","<<xy[i][1]<<") A=";
        for(int c=0;c<3;c++){if(c)line<<",";line<<tensor_value(tensor_scalar_ptr(a,static_cast<size_t>(c)*w*h+p),a.elemsize);}
        line<<" B=";
        for(int c=0;c<3;c++){if(c)line<<",";line<<tensor_value(tensor_scalar_ptr(b,static_cast<size_t>(c)*w*h+p),b.elemsize);}
    }
    LOGI("%s",line.str().c_str());
}

static void log_compare_detections(const char* name,const std::vector<Object>& objects,
                                   float top_score,int top_class,int64_t frame_id,int64_t image_timestamp_ns)
{
    std::ostringstream line;line<<"AHBCMP_DET frame="<<frame_id<<" ts="<<image_timestamp_ns
        <<" path="<<name<<" count="<<objects.size()<<" top="<<top_score<<" class="<<top_class<<" first=";
    const size_t count=std::min<size_t>(objects.size(),5);
    for(size_t i=0;i<count;i++){const Object&o=objects[i];if(i)line<<";";
        line<<o.label<<":"<<o.prob<<":"<<o.x<<","<<o.y<<","<<o.w<<","<<o.h;}
    LOGI("%s",line.str().c_str());
}

static std::string blob_edges(const std::vector<int>&ids,const std::vector<ncnn::Blob>&blobs){
 std::ostringstream o;o<<"[";for(size_t i=0;i<ids.size();i++){if(i)o<<";";int id=ids[i];if(id<0||(size_t)id>=blobs.size()){o<<id<<":INVALID";continue;}
 const ncnn::Blob&b=blobs[(size_t)id];o<<id<<":"<<b.name<<"{p="<<b.producer<<",c="<<b.consumer<<",shape="<<b.shape.w<<"x"<<b.shape.h<<"x"<<b.shape.d<<"x"<<b.shape.c<<",es="<<b.shape.elemsize<<",pack="<<b.shape.elempack<<"}";}
 o<<"]";return o.str();
}
}

Yolo11::Yolo11()
    : vkdev(nullptr), scale_layer(nullptr), ahb_cache_clock(0),
      ahb_cache_hits(0), ahb_cache_misses(0), ahb_cache_evictions(0),
      ahb_cache_rotate_rebuilds(0), ahb_compare_samples_done(0),
      target_w(320), target_h(320), num_classes(80),
      model_loaded(false), use_vulkan_compute(true), compute_mode(YOLO_COMPUTE_VULKAN_AHB),
      last_top_score(0.0f), last_top_class(-1)
{
    std::memset(dfl_distribution_scratchpad, 0, sizeof(dfl_distribution_scratchpad));
}

Yolo11::~Yolo11()
{
    clear_ahb_cache();
    if (scale_layer) {
        if (vkdev) {
            ncnn::Option opt = net.opt;
            scale_layer->destroy_pipeline(opt);
        }
        delete scale_layer;
        scale_layer = nullptr;
    }
    net.clear();
    cpu_net.clear();
}

void Yolo11::destroy_ahb_cache_entry(AHBImportCacheEntry* entry)
{
    if (!entry) return;
    delete entry->pipeline;
    entry->pipeline = nullptr;
    entry->src = ncnn::VkImageMat();
    delete entry->allocator;
    entry->allocator = nullptr;
    if (entry->ahb) {
        AHardwareBuffer_release(entry->ahb);
        entry->ahb = nullptr;
    }
    delete entry;
}

void Yolo11::clear_ahb_cache()
{
    for (auto& item : ahb_import_cache) destroy_ahb_cache_entry(item.second);
    ahb_import_cache.clear();
    ahb_cache_clock = 0;
    ahb_cache_hits = 0;
    ahb_cache_misses = 0;
    ahb_cache_evictions = 0;
    ahb_cache_rotate_rebuilds = 0;
    ahb_compare_samples_done = 0;
}

int Yolo11::load(AAssetManager* mgr, const char* modeltype, int target_size, int requested_compute_mode)
{
    if (!mgr || !modeltype) {
        LOGE("Invalid AssetManager or modeltype pointer");
        return -1;
    }
    if (target_size < 320 || target_size > 480 || (target_size % 32) != 0) {
        LOGE("Unsupported target size %d; expected 320..480 in steps of 32", target_size);
        return -4;
    }

    clear_ahb_cache();
    net.clear();
    cpu_net.clear();
    for (auto& slot : async_slots) {
        std::lock_guard<std::mutex> slot_lock(slot.mutex);
        for (auto& boundary : slot.boundary) boundary.release();
        slot.state = AsyncFrameSlot::FREE;
        slot.frame_id = 0;
        slot.image_timestamp_ns = 0;
    }

    target_w = target_size;
    target_h = target_size;
    if (requested_compute_mode < YOLO_COMPUTE_CPU ||
        requested_compute_mode > YOLO_COMPUTE_AHB_COMPARE) {
        LOGE("Invalid compute mode: %d", requested_compute_mode);
        return -4;
    }

    compute_mode = requested_compute_mode;
    ahb_compare_samples_done = 0;
    use_vulkan_compute = (compute_mode != YOLO_COMPUTE_CPU);
    model_loaded = false;
    last_top_score = 0.0f;
    last_top_class = -1;
    vkdev = nullptr;

    if (use_vulkan_compute) {
        const int gpu_count = ncnn::get_gpu_count();
        if (gpu_count <= 0) {
            LOGE("Vulkan requested but no NCNN Vulkan device is available");
            return -4;
        }
        vkdev = ncnn::get_gpu_device(0);
        if (!vkdev || !vkdev->is_valid()) {
            LOGE("No valid Vulkan device available");
            return -4;
        }

        if (compute_mode == YOLO_COMPUTE_VULKAN_AHB || compute_mode == YOLO_COMPUTE_VULKAN_AHB_SPLIT || compute_mode == YOLO_COMPUTE_VULKAN_AHB_ASYNC || compute_mode == YOLO_COMPUTE_AHB_COMPARE) {
            LOGI("AHB_CACHE_LIMIT=%zu", AHB_CACHE_LIMIT);
            const int ahb_ext = vkdev->info.support_VK_ANDROID_external_memory_android_hardware_buffer();
            if (ahb_ext <= 0) {
                LOGE("Vulkan device lacks VK_ANDROID_external_memory_android_hardware_buffer (version %d)", ahb_ext);
                return -4;
            }
            LOGI("AHB import extension available (spec version %d)", ahb_ext);
        }
        const char* dn=vkdev->info.device_name();const char* dr=vkdev->info.driver_name();
        LOGI("VULKAN_DEVICE name=%s driver=%s api=0x%08x driver_version=0x%08x vendor=0x%04x device=0x%04x heap_budget_mb=%u",
             dn?dn:"(null)",dr?dr:"(null)",vkdev->info.api_version(),vkdev->info.driver_version(),
             vkdev->info.vendor_id(),vkdev->info.device_id(),vkdev->get_heap_budget());
        net.set_vulkan_device(vkdev);
    }

    net.opt.use_vulkan_compute  = use_vulkan_compute;
    net.opt.use_fp16_packed     = use_vulkan_compute;
    net.opt.use_fp16_storage    = use_vulkan_compute;
    net.opt.use_fp16_arithmetic = use_vulkan_compute;
    net.opt.lightmode           = true;
    net.opt.num_threads         = 4;

    char parampath[256];
    char binpath[256];
    snprintf(parampath, sizeof(parampath), "yolo11%s.param", modeltype);
    snprintf(binpath, sizeof(binpath), "yolo11%s.bin", modeltype);

    LOGI("Loading YOLO11: %s, %s (target_size: %dx%d, compute_mode=%d)",
         parampath, binpath, target_w, target_h, compute_mode);

    int ret_param = net.load_param(mgr, parampath);
    if (ret_param != 0) {
        LOGE("Failed to load param: %s (code %d)", parampath, ret_param);
        return -2;
    }

    int ret_bin = net.load_model(mgr, binpath);
    if (ret_bin != 0) {
        LOGE("Failed to load model: %s (code %d)", binpath, ret_bin);
        return -3;
    }

    if (compute_mode == YOLO_COMPUTE_VULKAN_AHB_SPLIT || compute_mode == YOLO_COMPUTE_VULKAN_AHB_ASYNC) {
        cpu_net.opt.use_vulkan_compute = false;
        cpu_net.opt.use_fp16_packed = false;
        cpu_net.opt.use_fp16_storage = false;
        cpu_net.opt.use_fp16_arithmetic = false;
        cpu_net.opt.use_packing_layout = true;
        cpu_net.opt.lightmode = true;
        cpu_net.opt.num_threads = 4;

        const int cpu_param_ret = cpu_net.load_param(mgr, parampath);
        if (cpu_param_ret != 0) {
            LOGE("CPU suffix Net failed to load param: %s (code %d)", parampath, cpu_param_ret);
            return -14;
        }
        const int cpu_model_ret = cpu_net.load_model(mgr, binpath);
        if (cpu_model_ret != 0) {
            LOGE("CPU suffix Net failed to load model: %s (code %d)", binpath, cpu_model_ret);
            return -15;
        }
        LOGI("ASYNC_GRAPH_READY gpu_layers=%zu cpu_layers=%zu boundary_blobs=10",
             net.layers().size(), cpu_net.layers().size());

        static const int boundary_blob_ids[10] = {37, 72, 123, 125, 128, 129, 133, 135, 136, 137};
        static const int boundary_producers[10] = {31, 60, 102, 103, 105, 105, 107, 108, 108, 109};
        if ((std::strcmp(modeltype, "n") != 0 && std::strcmp(modeltype, "s") != 0) ||
            net.layers().size() != 257 || cpu_net.layers().size() != 257 ||
            net.blobs().size() != 307 || cpu_net.blobs().size() != 307) {
            LOGE("Async split supports only YOLO11n/YOLO11s graph signature (257 layers / 307 blobs)");
            return -16;
        }
        for (int i = 0; i < 10; ++i) {
            const int blob_id = boundary_blob_ids[i];
            if (net.blobs()[static_cast<size_t>(blob_id)].producer != boundary_producers[i] ||
                cpu_net.blobs()[static_cast<size_t>(blob_id)].producer != boundary_producers[i]) {
                LOGE("Async graph cut mismatch blob=%d expected_producer=%d gpu=%d cpu=%d",
                     blob_id, boundary_producers[i],
                     net.blobs()[static_cast<size_t>(blob_id)].producer,
                     cpu_net.blobs()[static_cast<size_t>(blob_id)].producer);
                return -17;
            }
        }
        for (size_t i = 1; i <= 109; ++i) {
            const ncnn::Layer* layer = net.layers()[i];
            if (!layer || !layer->support_vulkan) {
                LOGE("Async GPU prefix contains a non-Vulkan layer at idx=%zu", i);
                return -18;
            }
        }

        // Prove that the selected ten blobs are the complete graph cut:
        // no suffix layer may consume a prefix-produced blob outside this set.
        for (size_t layer_index = 110; layer_index < net.layers().size(); ++layer_index) {
            const ncnn::Layer* layer = net.layers()[layer_index];
            if (!layer) return -19;
            for (size_t bottom_index = 0; bottom_index < layer->bottoms.size(); ++bottom_index) {
                const int blob_id = layer->bottoms[bottom_index];
                if (blob_id < 0 || static_cast<size_t>(blob_id) >= net.blobs().size()) return -19;
                const int producer = net.blobs()[static_cast<size_t>(blob_id)].producer;
                if (producer <= 109) {
                    bool is_boundary = false;
                    for (int boundary_id : boundary_blob_ids) {
                        if (boundary_id == blob_id) {
                            is_boundary = true;
                            break;
                        }
                    }
                    if (!is_boundary) {
                        LOGE("Async graph cut is incomplete: suffix layer=%zu consumes prefix blob=%d produced_by=%d",
                             layer_index, blob_id, producer);
                        return -20;
                    }
                }
            }
        }

        for (int boundary_id : boundary_blob_ids) {
            bool consumed_in_prefix = false;
            bool consumed_in_suffix = false;
            for (size_t layer_index = 1; layer_index < net.layers().size(); ++layer_index) {
                const ncnn::Layer* layer = net.layers()[layer_index];
                if (!layer) return -21;
                for (int bottom_id : layer->bottoms) {
                    if (bottom_id == boundary_id) {
                        if (layer_index <= 109) consumed_in_prefix = true;
                        else consumed_in_suffix = true;
                    }
                }
            }
            if (consumed_in_prefix || !consumed_in_suffix) {
                LOGE("Async boundary blob=%d has invalid consumer placement prefix=%d suffix=%d",
                     boundary_id, consumed_in_prefix ? 1 : 0, consumed_in_suffix ? 1 : 0);
                return -22;
            }
        }
    }

    if (use_vulkan_compute && !vkdev) {
        LOGE("Vulkan initialization did not complete");
        return -5;
    }

    if (compute_mode == YOLO_COMPUTE_VULKAN_AHB || compute_mode == YOLO_COMPUTE_VULKAN_AHB_SPLIT || compute_mode == YOLO_COMPUTE_VULKAN_AHB_ASYNC || compute_mode == YOLO_COMPUTE_AHB_COMPARE) {
    if (scale_layer) {
        scale_layer->destroy_pipeline(net.opt);
        delete scale_layer;
        scale_layer = nullptr;
    }

    scale_layer = ncnn::create_layer(ncnn::LayerType::Scale);
    if (!scale_layer) {
        LOGE("Failed to create ncnn Scale layer");
        return -5;
    }
    scale_layer->vkdev = vkdev;

    ncnn::ParamDict pd;
    pd.set(0, 1);
    pd.set(1, 0);
    if (scale_layer->load_param(pd) != 0) {
        LOGE("Failed to load Scale layer parameters");
        delete scale_layer;
        scale_layer = nullptr;
        return -5;
    }

    ncnn::Mat weights[1];
    weights[0].create(1);
    if (weights[0].empty()) {
        LOGE("Failed to allocate Scale layer weights");
        delete scale_layer;
        scale_layer = nullptr;
        return -5;
    }
    weights[0][0] = 1.0f / 255.0f;
    ncnn::ModelBinFromMatArray mb(weights);
    if (scale_layer->load_model(mb) != 0) {
        LOGE("Failed to load Scale layer weights");
        delete scale_layer;
        scale_layer = nullptr;
        return -5;
    }

    ncnn::VkAllocator* scale_blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* scale_staging_allocator = vkdev->acquire_staging_allocator();
    if (!scale_blob_allocator || !scale_staging_allocator) {
        if (scale_blob_allocator) vkdev->reclaim_blob_allocator(scale_blob_allocator);
        if (scale_staging_allocator) vkdev->reclaim_staging_allocator(scale_staging_allocator);
        LOGE("Failed to acquire allocators for Scale layer upload");
        delete scale_layer;
        scale_layer = nullptr;
        return -5;
    }

    int scale_init_ret = 0;
    {
        ncnn::Option scale_opt = net.opt;
        scale_opt.blob_vkallocator = scale_blob_allocator;
        scale_opt.staging_vkallocator = scale_staging_allocator;

        ncnn::VkTransfer cmd_transfer(vkdev);
        scale_init_ret = scale_layer->upload_model(cmd_transfer, scale_opt);
        if (scale_init_ret == 0) scale_init_ret = cmd_transfer.submit_and_wait();
        if (scale_init_ret == 0) scale_init_ret = scale_layer->create_pipeline(scale_opt);
    }

    vkdev->reclaim_blob_allocator(scale_blob_allocator);
    vkdev->reclaim_staging_allocator(scale_staging_allocator);

    if (scale_init_ret != 0) {
        LOGE("GPU Scale layer initialization failed: %d", scale_init_ret);
        scale_layer->destroy_pipeline(net.opt);
        delete scale_layer;
        scale_layer = nullptr;
        return -5;
    }

    LOGI("GPU Scale layer initialized: RGB [0,255] -> [0,1]");
    }



    // Static graph/capabilities only; does not claim runtime backend placement.
    const auto& model_layers=net.layers();const auto& model_blobs=net.blobs();
    LOGI("NCNN_GRAPH_INVENTORY layers=%zu blobs=%zu mode=%d",model_layers.size(),model_blobs.size(),compute_mode);
    for(size_t i=0;i<model_layers.size();i++){const ncnn::Layer*l=model_layers[i];if(!l)continue;
        const std::string b=blob_edges(l->bottoms,model_blobs),t=blob_edges(l->tops,model_blobs);
        LOGI("NCNN_GRAPH_LAYER idx=%zu type=%s name=%s hint=%s vulkan=%d pack=%d vkpack=%d fp16=%d int8=%d tensor=%d anypack=%d vkanypack=%d bottoms=%s tops=%s",
             i,l->type.c_str(),l->name.c_str(),l->support_vulkan?"VULKAN_CAPABLE_RUNTIME_UNVERIFIED":"CPU_ONLY_CANDIDATE",
             l->support_vulkan?1:0,l->support_packing?1:0,l->support_vulkan_packing?1:0,
             l->support_fp16_storage?1:0,l->support_int8_storage?1:0,l->support_tensor_storage?1:0,
             l->support_any_packing?1:0,l->support_vulkan_any_packing?1:0,b.c_str(),t.c_str());
    }
    LOGI("NCNN_RUNTIME_PLACEMENT_NOTE: static layer capability is not proof of runtime execution, VkMat/Mat copies or synchronization.");

    generate_grids_and_stride(target_w, target_h);

    const size_t total_anchors = grid_strides.size();
    proposal_scratchpad.reserve(total_anchors);
    candidates_scratchpad.reserve(256);
    nms_indices_scratchpad.reserve(256);

    model_loaded = true;
    LOGI("YOLO11 loaded successfully. Total anchors: %zu", total_anchors);
    return 0;
}

void Yolo11::generate_grids_and_stride(int target_w, int target_h)
{
    grid_strides.clear();
    const int strides[3] = {8, 16, 32};

    size_t total_count = 0;
    for (int s : strides) {
        int gw = (target_w + s - 1) / s;
        int gh = (target_h + s - 1) / s;
        total_count += static_cast<size_t>(gw * gh);
    }
    grid_strides.reserve(total_count);

    for (int stride : strides) {
        int num_grid_x = target_w / stride;
        int num_grid_y = target_h / stride;
        for (int g1 = 0; g1 < num_grid_y; g1++) {
            for (int g0 = 0; g0 < num_grid_x; g0++) {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}

int Yolo11::detect_frame(AHardwareBuffer* ahb,float conf_thresh,float nms_thresh,std::vector<Object>&objects,int rotate_type,int64_t frame_id,int64_t image_timestamp_ns){
 if(compute_mode==YOLO_COMPUTE_VULKAN_AHB_SPLIT || compute_mode==YOLO_COMPUTE_VULKAN_AHB_ASYNC)return -13;
 if(compute_mode==YOLO_COMPUTE_AHB_COMPARE)return detect_ahb_compare(ahb,conf_thresh,nms_thresh,objects,rotate_type,frame_id,image_timestamp_ns);
 if(compute_mode==YOLO_COMPUTE_VULKAN_AHB)return detect_ahb(ahb,conf_thresh,nms_thresh,objects,rotate_type,frame_id,image_timestamp_ns);
 return detect_cpu_or_vulkan(ahb,conf_thresh,nms_thresh,objects,rotate_type,frame_id,image_timestamp_ns);
}

int Yolo11::detect_cpu_or_vulkan(AHardwareBuffer* ahb,float conf_thresh,float nms_thresh,std::vector<Object>&objects,int rotate_type,int64_t frame_id,int64_t image_timestamp_ns,AhbCompareSnapshot* snapshot)
{
    last_top_score = 0.0f;
    last_top_class = -1;
    objects.clear();
    if (!model_loaded) return -1;
    if (!ahb) return -2;
    if (compute_mode != YOLO_COMPUTE_CPU && compute_mode != YOLO_COMPUTE_VULKAN) return -3;
    if (compute_mode == YOLO_COMPUTE_VULKAN && (!use_vulkan_compute || !vkdev)) return -3;

    const auto frame_start = std::chrono::steady_clock::now();
    AHardwareBuffer_Desc desc{};
    AHardwareBuffer_describe(ahb, &desc);
    if (desc.width == 0 || desc.height == 0 ||
        (desc.usage & AHARDWAREBUFFER_USAGE_CPU_READ_MASK) == 0) {
        LOGE("CPU input path requires CPU-readable AHB; size=%ux%u usage=0x%llx",
             desc.width, desc.height, static_cast<unsigned long long>(desc.usage));
        return -4;
    }

    const auto preprocess_start = std::chrono::steady_clock::now();
    AHardwareBuffer_Planes planes{};
    int ret = AHardwareBuffer_lockPlanes(ahb, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN,
                                         -1, nullptr, &planes);
    if (ret != 0) {
        LOGE("AHardwareBuffer_lockPlanes failed: %d", ret);
        return -5;
    }

    if (desc.format != AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420 || planes.planeCount < 3) {
        AHardwareBuffer_unlock(ahb, nullptr);
        LOGE("Unsupported CPU YUV format: format=%u planeCount=%u", desc.format, planes.planeCount);
        return -6;
    }

    const uint32_t src_w = desc.width;
    const uint32_t src_h = desc.height;
    const auto& yp = planes.planes[0];
    const auto& cbp = planes.planes[1];
    const auto& crp = planes.planes[2];
    if (snapshot) LOGI("AHBCMP_YUV frame=%lld ts=%lld size=%ux%u Y(rs=%u ps=%u) U(rs=%u ps=%u) V(rs=%u ps=%u) planes=%u",
        (long long)frame_id,(long long)image_timestamp_ns,desc.width,desc.height,
        yp.rowStride,yp.pixelStride,cbp.rowStride,cbp.pixelStride,crp.rowStride,crp.pixelStride,planes.planeCount);
    if (!yp.data || !cbp.data || !crp.data ||
        yp.rowStride == 0 || cbp.rowStride == 0 || crp.rowStride == 0 ||
        yp.pixelStride == 0 || cbp.pixelStride == 0 || crp.pixelStride == 0) {
        AHardwareBuffer_unlock(ahb, nullptr);
        LOGE("Invalid YUV plane metadata");
        return -7;
    }

    const bool rotate_90 = (rotate_type == 6 || rotate_type == 8);
    const uint32_t rotated_w = rotate_90 ? src_h : src_w;
    const uint32_t rotated_h = rotate_90 ? src_w : src_h;
    cpu_rgb_scratchpad.resize(static_cast<size_t>(rotated_w) * rotated_h * 3u);
    unsigned char* rgb_data = cpu_rgb_scratchpad.data();

    const unsigned char* y_data = static_cast<const unsigned char*>(yp.data);
    const unsigned char* cb_data = static_cast<const unsigned char*>(cbp.data);
    const unsigned char* cr_data = static_cast<const unsigned char*>(crp.data);

    // BT.601 limited-range YUV420 to packed RGB. CPU and Vulkan baseline modes
    // share this exact preprocessing path so backend comparison is controlled.
    for (uint32_t sy = 0; sy < src_h; ++sy) {
        const unsigned char* y_row = y_data + static_cast<size_t>(sy) * yp.rowStride;
        const unsigned char* cb_row = cb_data + static_cast<size_t>(sy / 2u) * cbp.rowStride;
        const unsigned char* cr_row = cr_data + static_cast<size_t>(sy / 2u) * crp.rowStride;
        for (uint32_t sx = 0; sx < src_w; ++sx) {
            const int yy = std::max(0, static_cast<int>(y_row[sx * yp.pixelStride]) - 16);
            const int uu = static_cast<int>(cb_row[(sx / 2u) * cbp.pixelStride]) - 128;
            const int vv = static_cast<int>(cr_row[(sx / 2u) * crp.pixelStride]) - 128;
            const int r = (298 * yy + 409 * vv + 128) >> 8;
            const int g = (298 * yy - 100 * uu - 208 * vv + 128) >> 8;
            const int b = (298 * yy + 516 * uu + 128) >> 8;

            uint32_t dx = sx;
            uint32_t dy = sy;
            if (rotate_type == 6) { dx = src_h - 1u - sy; dy = sx; }
            else if (rotate_type == 3) { dx = src_w - 1u - sx; dy = src_h - 1u - sy; }
            else if (rotate_type == 8) { dx = sy; dy = src_w - 1u - sx; }

            const size_t dst = (static_cast<size_t>(dy) * rotated_w + dx) * 3u;
            rgb_data[dst + 0] = static_cast<unsigned char>(std::max(0, std::min(255, r)));
            rgb_data[dst + 1] = static_cast<unsigned char>(std::max(0, std::min(255, g)));
            rgb_data[dst + 2] = static_cast<unsigned char>(std::max(0, std::min(255, b)));
        }
    }

    const int unlock_ret = AHardwareBuffer_unlock(ahb, nullptr);
    if (unlock_ret != 0) {
        LOGW("AHardwareBuffer_unlock returned %d", unlock_ret);
    }

    ncnn::Mat input = ncnn::Mat::from_pixels_resize(
        rgb_data, ncnn::Mat::PIXEL_RGB, static_cast<int>(rotated_w),
        static_cast<int>(rotated_h), target_w, target_h);
    if (input.empty()) {
        LOGE("CPU YUV preprocess produced an empty NCNN Mat");
        return -8;
    }
    if (snapshot) snapshot->rgb_0_255 = input.clone();
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    input.substract_mean_normalize(mean_vals, norm_vals);
    if (snapshot) snapshot->input_after_norm = input.clone();
    const double preprocess_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - preprocess_start).count();

    ncnn::Mat out;
    int extract_ret = 0;
    double infer_extract_ms = 0.0;
    ncnn::VkAllocator* blob_allocator = nullptr;
    ncnn::VkAllocator* staging_allocator = nullptr;
    if (compute_mode == YOLO_COMPUTE_VULKAN) {
        blob_allocator = vkdev->acquire_blob_allocator();
        staging_allocator = vkdev->acquire_staging_allocator();
        if (!blob_allocator || !staging_allocator) {
            if (blob_allocator) vkdev->reclaim_blob_allocator(blob_allocator);
            if (staging_allocator) vkdev->reclaim_staging_allocator(staging_allocator);
            LOGE("Failed to acquire Vulkan allocators for Mat input");
            return -9;
        }
    }

    {
        ncnn::Extractor ex = net.create_extractor();
        if (compute_mode == YOLO_COMPUTE_VULKAN) {
            ex.set_blob_vkallocator(blob_allocator);
            ex.set_workspace_vkallocator(blob_allocator);
            ex.set_staging_vkallocator(staging_allocator);
        }
        extract_ret = ex.input("in0", input);
        if (extract_ret == 0) {
            const auto t = std::chrono::steady_clock::now();
            extract_ret = ex.extract("out0", out);
            infer_extract_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t).count();
        }
    }

    if (compute_mode == YOLO_COMPUTE_VULKAN) {
        vkdev->reclaim_blob_allocator(blob_allocator);
        vkdev->reclaim_staging_allocator(staging_allocator);
    }
    if (extract_ret != 0 || out.empty()) {
        LOGE("Extractor::extract(out0, Mat) failed in mode=%d ret=%d", compute_mode, extract_ret);
        return -10;
    }

    if(frame_id>0&&frame_id%30==0)log_tensor_stats("out0",out,frame_id,image_timestamp_ns);
    const auto post_start = std::chrono::steady_clock::now();
    proposal_scratchpad.clear();
    decode_dfl_and_filter(out, conf_thresh);
    objects.clear();
    nms_sorted_bboxes(nms_thresh, objects);
    const double post_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - post_start).count();
    const double frame_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - frame_start).count();

    static int s_cpu_vk_frame_log = 0;
    if (++s_cpu_vk_frame_log % 30 == 0) {
        LOGI("Pipeline mode=%s frame=%lld imageTs=%lld input=AHB_CPU_LOCK format=YUV420_888 preprocess_cpu=%.3fms infer_extract_wall=%.3fms post=%.3fms frame_total=%.3fms out(w=%d h=%d elemsize=%zu pack=%d) proposals=%zu detections=%zu top=%.3f cls=%d",
             compute_mode==YOLO_COMPUTE_CPU?"CPU":"VULKAN",(long long)frame_id,(long long)image_timestamp_ns,
             preprocess_ms,infer_extract_ms,post_ms,frame_ms,out.w,out.h,out.elemsize,out.elempack,
             proposal_scratchpad.size(),objects.size(),last_top_score,last_top_class);
    }
    if (snapshot) {
        snapshot->out0=out; snapshot->detections=objects;
        snapshot->top_score=last_top_score; snapshot->top_class=last_top_class;
        snapshot->preprocess_ms=preprocess_ms;
        snapshot->infer_ms=infer_extract_ms;
        snapshot->post_ms=post_ms;
    }
    return 0;
}


int Yolo11::detect_ahb_compare(AHardwareBuffer* ahb,float conf_thresh,float nms_thresh,
                               std::vector<Object>& objects,int rotate_type,
                               int64_t frame_id,int64_t image_timestamp_ns)
{
    if (!model_loaded) return -1;
    if (!ahb) return -2;
    const int original_mode=compute_mode;
    struct ModeRestore { int& mode; int original; ~ModeRestore(){mode=original;} } restore{compute_mode,original_mode};

    static const int64_t sample_interval=30;
    static const uint32_t max_samples=10;
    if (frame_id<=0 || frame_id%sample_interval!=0 || ahb_compare_samples_done>=max_samples) {
        compute_mode=YOLO_COMPUTE_VULKAN_AHB;
        return detect_ahb(ahb,conf_thresh,nms_thresh,objects,rotate_type,frame_id,image_timestamp_ns);
    }

    ++ahb_compare_samples_done;
    const uint32_t sample_number=ahb_compare_samples_done;
    AhbCompareSnapshot ahb_snapshot,cpu_snapshot;
    std::vector<Object> ahb_objects,cpu_objects;

    AHardwareBuffer_Desc frame_desc{};
    AHardwareBuffer_describe(ahb,&frame_desc);
    LOGI("AHBCMP_BEGIN sample=%u frame=%lld ts=%lld source=%ux%u target=%dx%d stride=%u format=0x%x usage=0x%llx rotate=%d",
         sample_number,(long long)frame_id,(long long)image_timestamp_ns,
         frame_desc.width,frame_desc.height,target_w,target_h,frame_desc.stride,frame_desc.format,
         (unsigned long long)frame_desc.usage,rotate_type);

    compute_mode=YOLO_COMPUTE_VULKAN_AHB;
    const int ret_a=detect_ahb(ahb,conf_thresh,nms_thresh,ahb_objects,rotate_type,
                               frame_id,image_timestamp_ns,&ahb_snapshot);
    if(ret_a!=0){
        LOGE("AHBCMP_END sample=%u frame=%lld status=FAIL_A ret=%d",sample_number,(long long)frame_id,ret_a);
        return ret_a;
    }

    compute_mode=YOLO_COMPUTE_VULKAN;
    const int ret_b=detect_cpu_or_vulkan(ahb,conf_thresh,nms_thresh,cpu_objects,rotate_type,
                                         frame_id,image_timestamp_ns,&cpu_snapshot);
    if(ret_b!=0){
        LOGE("AHBCMP_END sample=%u frame=%lld status=FAIL_B ret=%d",sample_number,(long long)frame_id,ret_b);
        return ret_b;
    }

    objects=ahb_objects;
    last_top_score=ahb_snapshot.top_score;
    last_top_class=ahb_snapshot.top_class;
    log_compare_tensor("RGB_0_255",ahb_snapshot.rgb_0_255,cpu_snapshot.rgb_0_255,frame_id,image_timestamp_ns);
    log_rgb_samples(ahb_snapshot.rgb_0_255,cpu_snapshot.rgb_0_255,frame_id,image_timestamp_ns);
    log_compare_tensor("INPUT_NORMALIZED_FP32",ahb_snapshot.input_after_norm,cpu_snapshot.input_after_norm,frame_id,image_timestamp_ns);
    log_single_tensor("AHB_INPUT_AFTER_CAST",ahb_snapshot.input_after_cast,frame_id,image_timestamp_ns);
    log_compare_tensor("OUT0",ahb_snapshot.out0,cpu_snapshot.out0,frame_id,image_timestamp_ns);
    log_compare_detections("AHB",ahb_snapshot.detections,ahb_snapshot.top_score,ahb_snapshot.top_class,frame_id,image_timestamp_ns);
    log_compare_detections("CPU_YUV",cpu_snapshot.detections,cpu_snapshot.top_score,cpu_snapshot.top_class,frame_id,image_timestamp_ns);
    LOGI("AHBCMP_TIMING frame=%lld ts=%lld diag_readback=1 AHB(preprocess=%.3fms inference=%.3fms post=%.3fms) CPU_YUV(preprocess=%.3fms inference=%.3fms post=%.3fms)",
         (long long)frame_id,(long long)image_timestamp_ns,
         ahb_snapshot.preprocess_ms,ahb_snapshot.infer_ms,ahb_snapshot.post_ms,
         cpu_snapshot.preprocess_ms,cpu_snapshot.infer_ms,cpu_snapshot.post_ms);
    LOGI("AHBCMP_END sample=%u frame=%lld ts=%lld status=OK",
         sample_number,(long long)frame_id,(long long)image_timestamp_ns);
    return 0;
}

int Yolo11::detect_ahb(AHardwareBuffer* ahb,float conf_thresh,float nms_thresh,std::vector<Object>&objects,int rotate_type,int64_t frame_id,int64_t image_timestamp_ns,AhbCompareSnapshot* snapshot)
{
    last_top_score = 0.0f;
    last_top_class = -1;
    objects.clear();
    if (!model_loaded) return -1;
    if (!ahb) return -2;
    if (compute_mode != YOLO_COMPUTE_VULKAN_AHB || !use_vulkan_compute || !vkdev) return -3;

    const auto frame_start = std::chrono::steady_clock::now();

    AHardwareBuffer_Desc desc{};
    AHardwareBuffer_describe(ahb, &desc);
    if (desc.width == 0 || desc.height == 0 ||
        (desc.usage & AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE) == 0) {
        LOGE("AHB not importable: %ux%u usage=0x%llx; GPU_SAMPLED_IMAGE required",
             desc.width, desc.height, static_cast<unsigned long long>(desc.usage));
        return -4;
    }

    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = vkdev->acquire_staging_allocator();
    if (!blob_allocator || !staging_allocator) {
        if (blob_allocator) vkdev->reclaim_blob_allocator(blob_allocator);
        if (staging_allocator) vkdev->reclaim_staging_allocator(staging_allocator);
        LOGE("Failed to acquire Vulkan frame allocators");
        return -5;
    }

    ncnn::Mat out;
    int execute_ret = 0;

    // Cache timings are intentionally separated. The old cache_create_ms only
    // measured pipe.create(), hiding allocator construction and AHB image import.
    double cache_lookup_ms = 0.0;
    double allocator_create_ms = 0.0;
    double image_import_ms = 0.0;
    double pipeline_create_ms = 0.0;
    double preprocess_record_ms=0.0,preprocess_import_record_ms=0.0,preprocess_scale_record_ms=0.0,preprocess_cast_record_ms=0.0;
    double preprocess_wait_ms=0.0;uint64_t preprocess_dispatches=0;double infer_extract_ms=0.0;
    double postprocess_ms = 0.0;
    bool cache_hit = false;

    {
        ncnn::Option opt = net.opt;
        opt.use_vulkan_compute = true;
        opt.blob_vkallocator = blob_allocator;
        opt.staging_vkallocator = staging_allocator;

        const auto lookup_start = std::chrono::steady_clock::now();
        auto cache_it = ahb_import_cache.find(ahb);
        cache_lookup_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - lookup_start).count();

        if (cache_it == ahb_import_cache.end()) {
            ++ahb_cache_misses;

            if (ahb_import_cache.size() >= AHB_CACHE_LIMIT) {
                auto oldest = ahb_import_cache.begin();
                for (auto it = ahb_import_cache.begin(); it != ahb_import_cache.end(); ++it) {
                    if (it->second->last_used < oldest->second->last_used) oldest = it;
                }

                ++ahb_cache_evictions;
                LOGW("AHB_CACHE_EVICT ptr=%p entries_before=%zu hits=%llu misses=%llu evictions=%llu",
                     static_cast<void*>(oldest->first), ahb_import_cache.size(),
                     static_cast<unsigned long long>(ahb_cache_hits),
                     static_cast<unsigned long long>(ahb_cache_misses),
                     static_cast<unsigned long long>(ahb_cache_evictions));

                destroy_ahb_cache_entry(oldest->second);
                ahb_import_cache.erase(oldest);
            }

            auto* entry = new (std::nothrow) AHBImportCacheEntry();
            if (!entry) {
                execute_ret = -6;
            } else {
                AHardwareBuffer_acquire(ahb);
                entry->ahb = ahb;

                auto t = std::chrono::steady_clock::now();
                entry->allocator = new (std::nothrow) ncnn::VkAndroidHardwareBufferImageAllocator(vkdev, ahb);
                allocator_create_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t).count();

                if (!entry->allocator) {
                    destroy_ahb_cache_entry(entry);
                    execute_ret = -6;
                } else {
                    t = std::chrono::steady_clock::now();
                    entry->src = ncnn::VkImageMat::from_android_hardware_buffer(entry->allocator);
                    image_import_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t).count();

                    if (entry->src.empty()) {
                        LOGE("VkImageMat::from_android_hardware_buffer returned empty image");
                        destroy_ahb_cache_entry(entry);
                        execute_ret = -6;
                    } else {
                        const auto& fp=entry->allocator->bufferFormatProperties;
                        const auto& bp=entry->allocator->bufferProperties;
                        LOGI("AHB_YCBCR_PROPERTIES ptr=%p format=%d externalFormat=0x%llx features=0x%08x model=%d range=%d chromaX=%d chromaY=%d components=%d,%d,%d,%d allocationSize=%llu memoryTypeBits=0x%08x",
                             (void*)ahb,(int)fp.format,(unsigned long long)fp.externalFormat,(unsigned int)fp.formatFeatures,
                             (int)fp.suggestedYcbcrModel,(int)fp.suggestedYcbcrRange,(int)fp.suggestedXChromaOffset,(int)fp.suggestedYChromaOffset,
                             (int)fp.samplerYcbcrConversionComponents.r,(int)fp.samplerYcbcrConversionComponents.g,
                             (int)fp.samplerYcbcrConversionComponents.b,(int)fp.samplerYcbcrConversionComponents.a,
                             (unsigned long long)bp.allocationSize,(unsigned int)bp.memoryTypeBits);
                        entry->pipeline = new (std::nothrow) ncnn::ImportAndroidHardwareBufferPipeline(vkdev);
                        if (!entry->pipeline) {
                            destroy_ahb_cache_entry(entry);
                            execute_ret = -6;
                        } else {
                            ncnn::Option pipeline_opt = opt;
                            pipeline_opt.use_fp16_storage = false;
                            pipeline_opt.use_fp16_packed = false;

                            t = std::chrono::steady_clock::now();
                            const int pr = entry->pipeline->create(
                                entry->allocator, 1, rotate_type,
                                target_w, target_h, pipeline_opt);
                            pipeline_create_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t).count();

                            if (pr != 0) {
                                LOGE("ImportAndroidHardwareBufferPipeline::create failed: %d", pr);
                                destroy_ahb_cache_entry(entry);
                                execute_ret = -6;
                            } else {
                                entry->rotate_type = rotate_type;
                                entry->last_used = ++ahb_cache_clock;
                                ahb_import_cache.emplace(ahb, entry);
                                cache_it = ahb_import_cache.find(ahb);

                                LOGI("AHB_CACHE_MISS ptr=%p %ux%u stride=%u fmt=%u layers=%u usage=0x%llx "
                                     "alloc=%.3fms import=%.3fms pipe=%.3fms entries=%zu",
                                     static_cast<void*>(ahb), desc.width, desc.height, desc.stride,
                                     desc.format, desc.layers,
                                     static_cast<unsigned long long>(desc.usage),
                                     allocator_create_ms, image_import_ms, pipeline_create_ms,
                                     ahb_import_cache.size());
                            }
                        }
                    }
                }
            }
        } else {
            cache_hit = true;
            ++ahb_cache_hits;
            cache_it->second->last_used = ++ahb_cache_clock;
        }

        if (execute_ret == 0) {
            AHBImportCacheEntry* entry = cache_it->second;
            if (snapshot) {
                const auto& fp = entry->allocator->bufferFormatProperties;
                LOGI("AHBCMP_AHB_YCBCR frame=%lld ts=%lld externalFormat=0x%llx features=0x%08x model=%d range=%d chromaX=%d chromaY=%d components=%d,%d,%d,%d",
                     (long long)frame_id,(long long)image_timestamp_ns,
                     (unsigned long long)fp.externalFormat,(unsigned int)fp.formatFeatures,
                     (int)fp.suggestedYcbcrModel,(int)fp.suggestedYcbcrRange,
                     (int)fp.suggestedXChromaOffset,(int)fp.suggestedYChromaOffset,
                     (int)fp.samplerYcbcrConversionComponents.r,(int)fp.samplerYcbcrConversionComponents.g,
                     (int)fp.samplerYcbcrConversionComponents.b,(int)fp.samplerYcbcrConversionComponents.a);
            }

            if (!entry->pipeline || entry->rotate_type != rotate_type) {
                ++ahb_cache_rotate_rebuilds;
                delete entry->pipeline;
                entry->pipeline = nullptr;

                ncnn::Option pipeline_opt = opt;
                pipeline_opt.use_fp16_storage = false;
                pipeline_opt.use_fp16_packed = false;

                const auto t = std::chrono::steady_clock::now();
                entry->pipeline = new (std::nothrow) ncnn::ImportAndroidHardwareBufferPipeline(vkdev);
                const int pr = entry->pipeline
                    ? entry->pipeline->create(entry->allocator, 1, rotate_type,
                                              target_w, target_h, pipeline_opt)
                    : -1;
                pipeline_create_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t).count();

                if (pr != 0) {
                    LOGE("Rebuilding AHB import pipeline failed: %d", pr);
                    delete entry->pipeline;
                    entry->pipeline = nullptr;
                    execute_ret = -7;
                } else {
                    entry->rotate_type = rotate_type;
                }
            }

            if (execute_ret == 0) {
                ncnn::VkMat dst;
                dst.create(target_w, target_h, 3, (size_t)4u, 1, blob_allocator);
                if (dst.empty()) {
                    execute_ret = -8;
                } else {
                    ncnn::VkMat dst_normalized;
                    ncnn::VkMat in_gpu;

                    // Stage 1: AHB import + preprocessing. This command still waits
                    // before inference because this diagnostic build isolates phases.
                    auto t = std::chrono::steady_clock::now();
                    {
                        ncnn::VkCompute preprocess_cmd(vkdev);
                        t = std::chrono::steady_clock::now();

                        auto st=std::chrono::steady_clock::now();
                        preprocess_cmd.record_import_android_hardware_buffer(entry->pipeline,entry->src,dst);
                        preprocess_import_record_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-st).count();
                        st=std::chrono::steady_clock::now();
                        execute_ret=scale_layer?scale_layer->forward(dst,dst_normalized,preprocess_cmd,opt):-9;
                        preprocess_scale_record_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-st).count();
                        if(execute_ret==0){
                            const int pack=1,cast=(opt.use_fp16_storage||opt.use_fp16_packed)?2:0;
                            st=std::chrono::steady_clock::now();
                            vkdev->convert_packing(dst_normalized,in_gpu,pack,cast,preprocess_cmd,opt);
                            preprocess_cast_record_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-st).count();
                            if (snapshot) {
                                preprocess_cmd.record_clone(dst,snapshot->rgb_0_255,opt);
                                preprocess_cmd.record_clone(dst_normalized,snapshot->input_after_norm,opt);
                                preprocess_cmd.record_clone(in_gpu,snapshot->input_after_cast,opt);
                            }
                        }
                        preprocess_record_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();
                        preprocess_dispatches=preprocess_cmd.pending_dispatch_total();

                        if (execute_ret == 0) {
                            t = std::chrono::steady_clock::now();
                            execute_ret = preprocess_cmd.submit_and_wait();
                            preprocess_wait_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t).count();
                            if (execute_ret != 0) {
                                LOGE("Preprocess submit_and_wait failed: %d", execute_ret);
                                execute_ret = -10;
                            }
                        }
                    } // Preprocess queue is complete; descriptors below do not read tensor memory.
                    if(frame_id>0&&frame_id%30==0){
                        log_vkmat("ahb_rgb_0_255",dst,frame_id,image_timestamp_ns);
                        log_vkmat("ahb_scaled_0_1",dst_normalized,frame_id,image_timestamp_ns);
                        log_vkmat("ncnn_input_after_cast",in_gpu,frame_id,image_timestamp_ns);
                    }
                    // Opt-in diagnostic VkMat->Mat clones. This adds GPU->CPU copies and a wait.
                    const int tap_mode=tensor_tap_property();
                    static int prev_tap_mode=-1;static bool one_shot_latched=false;
                    if(tap_mode!=prev_tap_mode){if(tap_mode==1)one_shot_latched=false;prev_tap_mode=tap_mode;}
                    if(tap_mode==0)one_shot_latched=false;
                    const bool do_tap=(tap_mode==1&&!one_shot_latched)||(tap_mode==2&&frame_id>0&&frame_id%30==0);
                    if(tap_mode==1&&do_tap)one_shot_latched=true;
                    if(do_tap){
                        ncnn::Mat tap_rgb,tap_scaled,tap_input;ncnn::VkCompute tap_cmd(vkdev);
                        const auto ts0=std::chrono::steady_clock::now();
                        tap_cmd.record_clone(dst,tap_rgb,opt);tap_cmd.record_clone(dst_normalized,tap_scaled,opt);tap_cmd.record_clone(in_gpu,tap_input,opt);
                        uint64_t d=tap_cmd.pending_dispatch_total();int rc=tap_cmd.submit_and_wait();
                        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts0).count();
                        LOGI("TENSOR_TAP frame=%lld imageTs=%lld mode=%d ret=%d dispatches=%llu wall=%.3fms PERTURBS_PIPELINE=1",
                             (long long)frame_id,(long long)image_timestamp_ns,tap_mode,rc,(unsigned long long)d,ms);
                        if(rc==0){log_tensor_stats("ahb_rgb_0_255_snapshot",tap_rgb,frame_id,image_timestamp_ns);
                            log_tensor_stats("ahb_scaled_0_1_snapshot",tap_scaled,frame_id,image_timestamp_ns);
                            log_tensor_stats("ncnn_input_snapshot",tap_input,frame_id,image_timestamp_ns);}
                    }

                    // Stage 2: use NCNN's high-level CPU extraction path.
                    // This is the last known-good path used before the diagnostic
                    // VkMat/manual-download change. Extractor handles GPU output
                    // conversion, packing and readback for ncnn::Mat.
                    if (execute_ret == 0) {
                        ncnn::Extractor ex = net.create_extractor();
                        ex.set_blob_vkallocator(blob_allocator);
                        ex.set_workspace_vkallocator(blob_allocator);
                        ex.set_staging_vkallocator(staging_allocator);
                        const int in_ret=ex.input("in0",in_gpu);
                        if(in_ret!=0){LOGE("Extractor::input(in0,VkMat) failed frame=%lld ret=%d",(long long)frame_id,in_ret);execute_ret=-12;}
                        else{t=std::chrono::steady_clock::now();execute_ret=ex.extract("out0",out);
                            infer_extract_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();}
                        if(execute_ret!=0){LOGE("Extractor input/extract failed frame=%lld ret=%d",(long long)frame_id,execute_ret);execute_ret=-11;}
                        else {
                            LOGD("Tensor CPU extracted dims=%d w=%d h=%d c=%d elemsize=%zu elempack=%d",
                                 out.dims, out.w, out.h, out.c, out.elemsize, out.elempack);
                        }
                    }
                }
            }
        }
    } // VkMat / VkCompute / Extractor references die before allocator reclaim.

    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);

    if (execute_ret != 0) {
        LOGE("AHB frame failed at GPU stage (code=%d)", execute_ret);
        return -11;
    }

    if(frame_id>0&&frame_id%30==0)log_tensor_stats("out0",out,frame_id,image_timestamp_ns);
    const auto post_start = std::chrono::steady_clock::now();
    proposal_scratchpad.clear();
    decode_dfl_and_filter(out, conf_thresh);
    objects.clear();
    nms_sorted_bboxes(nms_thresh, objects);
    postprocess_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - post_start).count();

    const double preprocess_total_ms = preprocess_record_ms + preprocess_wait_ms;
    const double infer_total_ms = infer_extract_ms; // Extractor wall time includes GPU execution, sync and output readback.
    const double cache_setup_ms = allocator_create_ms + image_import_ms + pipeline_create_ms;
    const double frame_total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - frame_start).count();

    static int s_frame_log = 0;
    if (++s_frame_log % 30 == 0) {
        LOGI("AHB_CACHE frame=%d ptr=%p state=%s hits=%llu misses=%llu evictions=%llu "
             "rotate_rebuilds=%llu entries=%zu lookup=%.3fms alloc=%.3fms import=%.3fms pipe=%.3fms",
             s_frame_log, static_cast<void*>(ahb), cache_hit ? "HIT" : "MISS",
             static_cast<unsigned long long>(ahb_cache_hits),
             static_cast<unsigned long long>(ahb_cache_misses),
             static_cast<unsigned long long>(ahb_cache_evictions),
             static_cast<unsigned long long>(ahb_cache_rotate_rebuilds),
             ahb_import_cache.size(), cache_lookup_ms, allocator_create_ms,
             image_import_ms, pipeline_create_ms);

        LOGI("Pipeline frame=%lld imageTs=%lld record_total=%.3fms import_record=%.3fms scale_record=%.3fms cast_record=%.3fms dispatches=%llu wait=%.3fms preprocess_total=%.3fms infer_extract_wall=%.3fms post=%.3fms cache_setup=%.3fms frame_total=%.3fms",
             (long long)frame_id,(long long)image_timestamp_ns,preprocess_record_ms,preprocess_import_record_ms,preprocess_scale_record_ms,preprocess_cast_record_ms,
             (unsigned long long)preprocess_dispatches,preprocess_wait_ms,preprocess_total_ms,infer_total_ms,postprocess_ms,cache_setup_ms,frame_total_ms);

        LOGI("Tensor out0 frame=%lld imageTs=%lld dims=%d shape=%d,%d,%d,%d elemsize=%zu pack=%d proposals=%zu detections=%zu top=%.3f cls=%d",
             (long long)frame_id,(long long)image_timestamp_ns,out.dims,out.w,out.h,out.d,out.c,out.elemsize,out.elempack,
             proposal_scratchpad.size(),objects.size(),last_top_score,last_top_class);

        const size_t detail_count = std::min<size_t>(objects.size(), 8);
        for (size_t i = 0; i < detail_count; i++) {
            const Object& obj = objects[i];
            LOGD("det[%zu] cls=%d score=%.3f xywh=(%.4f,%.4f,%.4f,%.4f) zone=%d",
                 i, obj.label, obj.prob,
                 obj.x / target_w, obj.y / target_h, obj.w / target_w, obj.h / target_h,
                 obj.zone_id);
        }
    }

    if (snapshot) {
        snapshot->out0=out; snapshot->detections=objects;
        snapshot->top_score=last_top_score; snapshot->top_class=last_top_class;
        snapshot->preprocess_ms=preprocess_total_ms;
        snapshot->infer_ms=infer_total_ms;
        snapshot->post_ms=postprocess_ms;
    }
    return 0;
}


int Yolo11::gpu_prefix_frame(int slot_index, AHardwareBuffer* ahb, int rotate_type,
                             int64_t frame_id, int64_t image_timestamp_ns)
{
    if (slot_index < 0 || slot_index >= static_cast<int>(async_slots.size())) return -20;
    AsyncFrameSlot& slot = async_slots[static_cast<size_t>(slot_index)];
    std::lock_guard<std::mutex> slot_lock(slot.mutex);

    if (!model_loaded || (compute_mode != YOLO_COMPUTE_VULKAN_AHB_SPLIT && compute_mode != YOLO_COMPUTE_VULKAN_AHB_ASYNC) || !vkdev || !scale_layer) return -21;
    if (!ahb) return -22;
    if (slot.state != AsyncFrameSlot::FREE) return -23;

    slot.state = AsyncFrameSlot::GPU_RUNNING;
    slot.frame_id = frame_id;
    slot.image_timestamp_ns = image_timestamp_ns;
    for (auto& boundary : slot.boundary) boundary.release();

    auto fail_slot = [&](int code) {
        for (auto& boundary : slot.boundary) boundary.release();
        slot.state = AsyncFrameSlot::FREE;
        return code;
    };

    AHardwareBuffer_Desc desc{};
    AHardwareBuffer_describe(ahb, &desc);
    if (desc.width == 0 || desc.height == 0 ||
        (desc.usage & AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE) == 0) {
        LOGE("Async prefix: AHB not importable %ux%u usage=0x%llx",
             desc.width, desc.height, static_cast<unsigned long long>(desc.usage));
        return fail_slot(-24);
    }

    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = vkdev->acquire_staging_allocator();
    if (!blob_allocator || !staging_allocator) {
        if (blob_allocator) vkdev->reclaim_blob_allocator(blob_allocator);
        if (staging_allocator) vkdev->reclaim_staging_allocator(staging_allocator);
        return fail_slot(-25);
    }

    int execute_ret = 0;
    {
        ncnn::Option opt = net.opt;
        opt.use_vulkan_compute = true;
        opt.blob_vkallocator = blob_allocator;
        opt.workspace_vkallocator = blob_allocator;
        opt.staging_vkallocator = staging_allocator;

        // The cache is single-owner: only the serial GPU worker calls this path.
        AHBImportCacheEntry* entry = nullptr;
        auto cache_it = ahb_import_cache.find(ahb);
        if (cache_it == ahb_import_cache.end()) {
            if (ahb_import_cache.size() >= AHB_CACHE_LIMIT) {
                auto oldest = ahb_import_cache.begin();
                for (auto it = ahb_import_cache.begin(); it != ahb_import_cache.end(); ++it) {
                    if (it->second->last_used < oldest->second->last_used) oldest = it;
                }
                destroy_ahb_cache_entry(oldest->second);
                ahb_import_cache.erase(oldest);
                ++ahb_cache_evictions;
            }

            entry = new (std::nothrow) AHBImportCacheEntry();
            if (!entry) {
                execute_ret = -26;
            } else {
                AHardwareBuffer_acquire(ahb);
                entry->ahb = ahb;
                entry->allocator = new (std::nothrow) ncnn::VkAndroidHardwareBufferImageAllocator(vkdev, ahb);
                if (!entry->allocator) {
                    destroy_ahb_cache_entry(entry);
                    entry = nullptr;
                    execute_ret = -26;
                } else {
                    entry->src = ncnn::VkImageMat::from_android_hardware_buffer(entry->allocator);
                    if (entry->src.empty()) {
                        destroy_ahb_cache_entry(entry);
                        entry = nullptr;
                        execute_ret = -27;
                    } else {
                        entry->pipeline = new (std::nothrow) ncnn::ImportAndroidHardwareBufferPipeline(vkdev);
                        if (!entry->pipeline) {
                            destroy_ahb_cache_entry(entry);
                            entry = nullptr;
                            execute_ret = -28;
                        } else {
                            ncnn::Option pipeline_opt = opt;
                            pipeline_opt.use_fp16_storage = false;
                            pipeline_opt.use_fp16_packed = false;
                            const int pipeline_ret = entry->pipeline->create(
                                entry->allocator, 1, rotate_type, target_w, target_h, pipeline_opt);
                            if (pipeline_ret != 0) {
                                destroy_ahb_cache_entry(entry);
                                entry = nullptr;
                                execute_ret = -29;
                            } else {
                                entry->rotate_type = rotate_type;
                                entry->last_used = ++ahb_cache_clock;
                                ahb_import_cache.emplace(ahb, entry);
                                ++ahb_cache_misses;
                            }
                        }
                    }
                }
            }
        } else {
            entry = cache_it->second;
            ++ahb_cache_hits;
            entry->last_used = ++ahb_cache_clock;
        }

        if (execute_ret == 0 && entry && entry->rotate_type != rotate_type) {
            delete entry->pipeline;
            entry->pipeline = nullptr;
            ncnn::Option pipeline_opt = opt;
            pipeline_opt.use_fp16_storage = false;
            pipeline_opt.use_fp16_packed = false;
            entry->pipeline = new (std::nothrow) ncnn::ImportAndroidHardwareBufferPipeline(vkdev);
            const int pipeline_ret = entry->pipeline
                ? entry->pipeline->create(entry->allocator, 1, rotate_type, target_w, target_h, pipeline_opt)
                : -1;
            if (pipeline_ret != 0) {
                delete entry->pipeline;
                entry->pipeline = nullptr;
                execute_ret = -30;
            } else {
                entry->rotate_type = rotate_type;
                ++ahb_cache_rotate_rebuilds;
            }
        }

        if (execute_ret == 0 && entry && entry->pipeline) {
            ncnn::VkMat dst;
            dst.create(target_w, target_h, 3, (size_t)4u, 1, blob_allocator);
            if (dst.empty()) {
                execute_ret = -31;
            } else {
                ncnn::VkMat dst_normalized;
                ncnn::VkMat input_gpu;
                ncnn::VkCompute cmd(vkdev);

                cmd.record_import_android_hardware_buffer(entry->pipeline, entry->src, dst);
                execute_ret = scale_layer->forward(dst, dst_normalized, cmd, opt);
                if (execute_ret == 0) {
                    const int cast_type = (opt.use_fp16_storage || opt.use_fp16_packed) ? 2 : 0;
                    vkdev->convert_packing(dst_normalized, input_gpu, 1, cast_type, cmd, opt);
                    if (input_gpu.empty()) execute_ret = -32;
                }

                if (execute_ret == 0) {
                    ncnn::Extractor gpu_ex = net.create_extractor();
                    gpu_ex.set_light_mode(false);
                    gpu_ex.set_blob_vkallocator(blob_allocator);
                    gpu_ex.set_workspace_vkallocator(blob_allocator);
                    gpu_ex.set_staging_vkallocator(staging_allocator);
                    execute_ret = gpu_ex.input("in0", input_gpu);
                    if (execute_ret == 0) {
                        static const int boundary_blob_ids[10] = {37, 72, 123, 125, 128, 129, 133, 135, 136, 137};
                        ncnn::VkMat boundary_gpu[10];
                        for (int i = 0; i < 10 && execute_ret == 0; ++i) {
                            execute_ret = gpu_ex.extract(boundary_blob_ids[i], boundary_gpu[i], cmd);
                            if (execute_ret == 0 && boundary_gpu[i].empty()) execute_ret = -33;
                        }

                        if (execute_ret == 0) {
                            ncnn::Option download_opt = opt;
                            download_opt.use_packing_layout = false;
                            for (int i = 0; i < 10; ++i) {
                                cmd.record_download(boundary_gpu[i], slot.boundary[static_cast<size_t>(i)], download_opt);
                            }
                            execute_ret = cmd.submit_and_wait();
                            if (execute_ret == 0) {
                                for (int i = 0; i < 10; ++i) {
                                    const ncnn::Mat& boundary = slot.boundary[static_cast<size_t>(i)];
                                    if (boundary.empty()) {
                                        execute_ret = -34;
                                        break;
                                    }
                                    if (boundary.elemsize != 4u || boundary.elempack != 1) {
                                        LOGE("Async boundary format mismatch blob_index=%d elemsize=%zu elempack=%d; expected FP32 pack1",
                                             boundary_blob_ids[i], boundary.elemsize, boundary.elempack);
                                        execute_ret = -36;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else if (execute_ret == 0) {
            execute_ret = -35;
        }
    }

    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);

    if (execute_ret != 0) {
        LOGE("Async GPU prefix failed frame=%lld slot=%d ret=%d",
             static_cast<long long>(frame_id), slot_index, execute_ret);
        return fail_slot(execute_ret);
    }

    slot.state = AsyncFrameSlot::CPU_READY;
    LOGD("ASYNC_PREFIX_READY frame=%lld slot=%d boundary_count=10",
         static_cast<long long>(frame_id), slot_index);
    return 0;
}

int Yolo11::cpu_suffix_frame(int slot_index, float conf_thresh, float nms_thresh,
                             std::vector<Object>& objects)
{
    if (slot_index < 0 || slot_index >= static_cast<int>(async_slots.size())) return -40;
    AsyncFrameSlot& slot = async_slots[static_cast<size_t>(slot_index)];
    std::lock_guard<std::mutex> slot_lock(slot.mutex);

    if (!model_loaded || (compute_mode != YOLO_COMPUTE_VULKAN_AHB_SPLIT && compute_mode != YOLO_COMPUTE_VULKAN_AHB_ASYNC)) return -41;
    if (slot.state != AsyncFrameSlot::CPU_READY) return -42;

    slot.state = AsyncFrameSlot::CPU_RUNNING;
    const int64_t frame_id = slot.frame_id;
    const int64_t image_timestamp_ns = slot.image_timestamp_ns;
    last_top_score = 0.0f;
    last_top_class = -1;
    objects.clear();

    int execute_ret = 0;
    ncnn::Mat out;
    {
        ncnn::Extractor cpu_ex = cpu_net.create_extractor();
        cpu_ex.set_light_mode(true);
        static const int boundary_blob_ids[10] = {37, 72, 123, 125, 128, 129, 133, 135, 136, 137};
        for (int i = 0; i < 10 && execute_ret == 0; ++i) {
            const ncnn::Mat& boundary = slot.boundary[static_cast<size_t>(i)];
            if (boundary.empty()) {
                execute_ret = -43;
                break;
            }
            execute_ret = cpu_ex.input(boundary_blob_ids[i], boundary);
        }
        if (execute_ret == 0) execute_ret = cpu_ex.extract("out0", out);
    }

    for (auto& boundary : slot.boundary) boundary.release();

    if (execute_ret != 0 || out.empty()) {
        slot.state = AsyncFrameSlot::FREE;
        LOGE("Async CPU suffix failed frame=%lld slot=%d ret=%d",
             static_cast<long long>(frame_id), slot_index, execute_ret);
        return execute_ret != 0 ? execute_ret : -44;
    }

    proposal_scratchpad.clear();
    decode_dfl_and_filter(out, conf_thresh);
    objects.clear();
    nms_sorted_bboxes(nms_thresh, objects);

    if (frame_id > 0 && frame_id % 30 == 0) {
        log_tensor_stats("async_out0", out, frame_id, image_timestamp_ns);
        LOGI("ASYNC_SUFFIX_DONE frame=%lld slot=%d detections=%zu top=%.4f class=%d",
             static_cast<long long>(frame_id), slot_index, objects.size(),
             last_top_score, last_top_class);
    }
    slot.state = AsyncFrameSlot::FREE;
    return 0;
}

void Yolo11::decode_dfl_and_filter(const ncnn::Mat& out_blob, float conf_thresh)
{
    last_top_score = 0.0f;
    last_top_class = -1;

    if (out_blob.empty()) {
        LOGE("decode_dfl_and_filter: out_blob is empty!");
        return;
    }

    ncnn::Mat out_flat;
    if (out_blob.dims == 3) {
        if (out_blob.c == 1) out_flat = out_blob.reshape(out_blob.w, out_blob.h);
        else if (out_blob.w == 1) out_flat = out_blob.reshape(out_blob.h, out_blob.c);
        else out_flat = out_blob.reshape(out_blob.w * out_blob.h, out_blob.c);
    } else {
        out_flat = out_blob;
    }

    if (out_flat.w > out_flat.h && out_flat.h >= 68) {
        ncnn::Mat out_t;
        out_t.create(out_flat.h, out_flat.w, (size_t)4u);
        for (int r = 0; r < out_flat.h; r++) {
            const float* src_row = out_flat.row(r);
            for (int c = 0; c < out_flat.w; c++) {
                out_t.row(c)[r] = src_row[c];
            }
        }
        out_flat = out_t;
    }

    const int total_anchors = out_flat.h;
    if (total_anchors <= 0 || out_flat.w < 68) {
        LOGW("decode_dfl_and_filter: invalid shape out_flat(w=%d, h=%d, dims=%d)",
             out_flat.w, out_flat.h, out_flat.dims);
        return;
    }

    const int cls_count = out_flat.w - 64;
    if (cls_count != num_classes || total_anchors != static_cast<int>(grid_strides.size())) {
        LOGE("Unexpected YOLO output layout: features=%d expected=%d anchors=%d expected=%zu",
             out_flat.w, 64 + num_classes, total_anchors, grid_strides.size());
        return;
    }

    for (int i = 0; i < total_anchors; i++) {
        const float* feat_ptr = out_flat.row(i);
        const float* cls_ptr = feat_ptr + 64;

        int best_class_id = -1;
        float best_score = -1.0f;

        for (int c = 0; c < cls_count; c++) {
            float logit = cls_ptr[c];
            if (!std::isfinite(logit)) continue;
            float score = 1.0f / (1.0f + std::exp(-logit));
            if (score > best_score) {
                best_score = score;
                best_class_id = c;
            }
        }

        if (best_score > last_top_score) {
            last_top_score = best_score;
            last_top_class = best_class_id;
        }

        if (best_score < conf_thresh) {
            continue;
        }

        float pred_ltrb[4];
        bool valid_dfl = true;
        for (int k = 0; k < 4; k++) {
            const float* dfl_reg = feat_ptr + (k * REG_MAX);

            float max_val = dfl_reg[0];
            if (!std::isfinite(max_val)) { valid_dfl = false; break; }
            for (int j = 1; j < REG_MAX; j++) {
                if (!std::isfinite(dfl_reg[j])) { valid_dfl = false; break; }
                if (dfl_reg[j] > max_val) max_val = dfl_reg[j];
            }
            if (!valid_dfl) break;

            float sum_exp = 0.0f;
            for (int j = 0; j < REG_MAX; j++) {
                dfl_distribution_scratchpad[j] = std::exp(dfl_reg[j] - max_val);
                sum_exp += dfl_distribution_scratchpad[j];
            }

            if (sum_exp <= 0.0f || !std::isfinite(sum_exp)) { valid_dfl = false; break; }
            const float inv_sum = 1.0f / sum_exp;
            float expected_val = 0.0f;
            for (int j = 0; j < REG_MAX; j++) {
                expected_val += static_cast<float>(j) * (dfl_distribution_scratchpad[j] * inv_sum);
            }
            pred_ltrb[k] = expected_val;
        }
        if (!valid_dfl) continue;

        const float stride = static_cast<float>(grid_strides[i].stride);
        const float grid0  = static_cast<float>(grid_strides[i].grid0);
        const float grid1  = static_cast<float>(grid_strides[i].grid1);

        float x0 = std::max(0.0f, std::min(static_cast<float>(target_w), (grid0 + 0.5f - pred_ltrb[0]) * stride));
        float y0 = std::max(0.0f, std::min(static_cast<float>(target_h), (grid1 + 0.5f - pred_ltrb[1]) * stride));
        float x1 = std::max(0.0f, std::min(static_cast<float>(target_w), (grid0 + 0.5f + pred_ltrb[2]) * stride));
        float y1 = std::max(0.0f, std::min(static_cast<float>(target_h), (grid1 + 0.5f + pred_ltrb[3]) * stride));

        float bw = std::max(0.0f, x1 - x0);
        float bh = std::max(0.0f, y1 - y0);

        if (bw < 1.0f || bh < 1.0f) {
            continue;
        }

        Object obj;
        obj.label = best_class_id;
        obj.prob = best_score;
        obj.x = x0;
        obj.y = y0;
        obj.w = bw;
        obj.h = bh;
        obj.zone_id = ZONE_MID;

        proposal_scratchpad.push_back(obj);
    }
}

void Yolo11::nms_sorted_bboxes(float nms_thresh, std::vector<Object>& output)
{
    if (proposal_scratchpad.empty()) {
        return;
    }

    // Wykorzystanie lambdy do czystego sortowania malejaco po prawdopodobienstwie
    std::sort(proposal_scratchpad.begin(), proposal_scratchpad.end(),
              [](const Object& a, const Object& b) {
                  return a.prob > b.prob;
              });

    candidates_scratchpad.clear();

    const size_t total = proposal_scratchpad.size();
    for (size_t i = 0; i < total; i++) {
        const Object& a = proposal_scratchpad[i];

        bool keep = true;
        for (size_t j = 0; j < candidates_scratchpad.size(); j++) {
            const Object& b = candidates_scratchpad[j];

            if (a.label != b.label) {
                continue;
            }

            float ix1 = std::max(a.x, b.x);
            float iy1 = std::max(a.y, b.y);
            float ix2 = std::min(a.x + a.w, b.x + b.w);
            float iy2 = std::min(a.y + a.h, b.y + b.h);

            float iw = std::max(0.0f, ix2 - ix1);
            float ih = std::max(0.0f, iy2 - iy1);

            float inter_area = iw * ih;
            float union_area = a.w * a.h + b.w * b.h - inter_area;

            if (union_area > 0.0f) {
                float iou = inter_area / union_area;
                if (iou > nms_thresh) {
                    keep = false;
                    break;
                }
            }
        }

        if (keep) {
            candidates_scratchpad.push_back(a);
            if (candidates_scratchpad.size() >= 64) {
                break;
            }
        }
    }

    output.clear();

    for (size_t i = 0; i < candidates_scratchpad.size(); i++) {
        Object obj = candidates_scratchpad[i];
        const float norm_bottom_y = (obj.y + obj.h) / static_cast<float>(target_h);
        if (norm_bottom_y > 0.70f) obj.zone_id = ZONE_NEAR;
        else if (norm_bottom_y >= 0.35f) obj.zone_id = ZONE_MID;
        else obj.zone_id = ZONE_FAR;

        output.push_back(obj);
    }
}

int Yolo11::pack_detections_flat(const std::vector<Object>& objects, float* out_buffer, int max_capacity)
{
    if (!out_buffer || max_capacity < 1) {
        return 0;
    }

    int count = static_cast<int>(objects.size());
    const int max_objects = (max_capacity - 1) / 7;
    if (count > max_objects) {
        count = max_objects;
    }

    out_buffer[0] = static_cast<float>(count);

    const float inv_w = 1.0f / static_cast<float>(target_w);
    const float inv_h = 1.0f / static_cast<float>(target_h);

    for (int i = 0; i < count; i++) {
        const int base = 1 + (i * 7);
        out_buffer[base + 0] = static_cast<float>(objects[i].zone_id);
        out_buffer[base + 1] = static_cast<float>(objects[i].label);
        out_buffer[base + 2] = objects[i].prob;
        out_buffer[base + 3] = objects[i].x * inv_w;
        out_buffer[base + 4] = objects[i].y * inv_h;
        out_buffer[base + 5] = objects[i].w * inv_w;
        out_buffer[base + 6] = objects[i].h * inv_h;
    }

    return count;
}
