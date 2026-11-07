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

#define TAG "Yolo11_AHB_MaliG72"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

Yolo11::Yolo11()
    : vkdev(nullptr), scale_layer(nullptr), ahb_cache_clock(0),
      ahb_cache_hits(0), ahb_cache_misses(0), ahb_cache_evictions(0),
      ahb_cache_rotate_rebuilds(0),
      target_w(320), target_h(320), num_classes(80),
      model_loaded(false), use_vulkan_compute(true),
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
}

int Yolo11::load(AAssetManager* mgr, const char* modeltype, int target_size, bool use_gpu)
{
    if (!mgr || !modeltype) {
        LOGE("Invalid AssetManager or modeltype pointer");
        return -1;
    }

    clear_ahb_cache();
    net.clear();

    target_w = target_size;
    target_h = target_size;
    use_vulkan_compute = use_gpu;
    model_loaded = false;
    last_top_score = 0.0f;
    last_top_class = -1;

    const int gpu_count = ncnn::get_gpu_count();
    if (!use_gpu || gpu_count <= 0) {
        LOGE("AHB camera inference requires Vulkan; CPU fallback is not implemented in this path");
        return -4;
    }
    vkdev = ncnn::get_gpu_device(0);
    if (!vkdev || !vkdev->is_valid()) {
        LOGE("No valid Vulkan device available for AHB import");
        return -4;
    }
    const int ahb_ext = vkdev->info.support_VK_ANDROID_external_memory_android_hardware_buffer();
    if (ahb_ext <= 0) {
        LOGE("Vulkan device lacks VK_ANDROID_external_memory_android_hardware_buffer (version %d)", ahb_ext);
        return -4;
    }
    use_vulkan_compute = true;
    net.set_vulkan_device(vkdev);
    LOGI("AHB import extension available (spec version %d)", ahb_ext);

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

    LOGI("Loading YOLO11: %s, %s (target_size: %dx%d, Vulkan: %d)",
         parampath, binpath, target_w, target_h, use_vulkan_compute ? 1 : 0);

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
                                                                                
    if (!use_vulkan_compute || !vkdev) {
        LOGE("Vulkan initialization did not complete");
        return -5;
    }

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

int Yolo11::detect_ahb(AHardwareBuffer* ahb, float conf_thresh, float nms_thresh,
                         std::vector<Object>& objects, int rotate_type)
{
    last_top_score = 0.0f;
    last_top_class = -1;
    objects.clear();
    if (!model_loaded) return -1;
    if (!ahb) return -2;
    if (!use_vulkan_compute || !vkdev) return -3;

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
    double preprocess_record_ms = 0.0;
    double preprocess_wait_ms = 0.0;
    double infer_extract_ms = 0.0;
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

                        preprocess_cmd.record_import_android_hardware_buffer(entry->pipeline, entry->src, dst);
                        execute_ret = scale_layer
                            ? scale_layer->forward(dst, dst_normalized, preprocess_cmd, opt)
                            : -9;

                        if (execute_ret == 0) {
                            const int pack = 1;
                            const int cast = (opt.use_fp16_storage || opt.use_fp16_packed) ? 2 : 0;
                            vkdev->convert_packing(dst_normalized, in_gpu, pack, cast, preprocess_cmd, opt);
                        }

                        preprocess_record_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t).count();

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
                    } // Release the completed preprocessing command before inference.

                    // Stage 2: use NCNN's high-level CPU extraction path.
                    // This is the last known-good path used before the diagnostic
                    // VkMat/manual-download change. Extractor handles GPU output
                    // conversion, packing and readback for ncnn::Mat.
                    if (execute_ret == 0) {
                        ncnn::Extractor ex = net.create_extractor();
                        ex.set_blob_vkallocator(blob_allocator);
                        ex.set_workspace_vkallocator(blob_allocator);
                        ex.set_staging_vkallocator(staging_allocator);
                        ex.input("in0", in_gpu);

                        t = std::chrono::steady_clock::now();
                        execute_ret = ex.extract("out0", out);
                        infer_extract_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t).count();

                        if (execute_ret != 0) {
                            LOGE("Extractor::extract(out0, Mat) failed: %d", execute_ret);
                            execute_ret = -11;
                        } else {
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

    const auto post_start = std::chrono::steady_clock::now();
    proposal_scratchpad.clear();
    decode_dfl_and_filter(out, conf_thresh);
    objects.clear();
    nms_sorted_bboxes(nms_thresh, objects);
    postprocess_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - post_start).count();

    const double preprocess_total_ms = preprocess_record_ms + preprocess_wait_ms;
    const double infer_total_ms = infer_extract_ms;
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

        LOGI("Pipeline frame=%d preprocess_record=%.3fms preprocess_wait=%.3fms preprocess_total=%.3fms "
             "infer_extract_total=%.3fms post=%.3fms cache_setup=%.3fms frame_total=%.3fms",
             s_frame_log, preprocess_record_ms, preprocess_wait_ms, preprocess_total_ms,
             infer_total_ms, postprocess_ms, cache_setup_ms, frame_total_ms);

        LOGI("Tensor out(dims=%d w=%d h=%d c=%d elemsize=%zu elempack=%d) "
             "proposals=%zu detections=%zu top=%.3f cls=%d",
             out.dims, out.w, out.h, out.c, out.elemsize, out.elempack,
             proposal_scratchpad.size(), objects.size(), last_top_score, last_top_class);

        const size_t detail_count = std::min<size_t>(objects.size(), 8);
        for (size_t i = 0; i < detail_count; i++) {
            const Object& obj = objects[i];
            LOGD("det[%zu] cls=%d score=%.3f xywh=(%.4f,%.4f,%.4f,%.4f) zone=%d",
                 i, obj.label, obj.prob,
                 obj.x / target_w, obj.y / target_h, obj.w / target_w, obj.h / target_h,
                 obj.zone_id);
        }
    }

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
