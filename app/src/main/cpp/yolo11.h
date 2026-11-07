#ifndef YOLO11_H
#define YOLO11_H

/**
 * ============================================================================
 * YOLO11 NCNN AHardwareBuffer Vulkan Import Engine for Autonomous Robotic Mower
 * Platform: Samsung Galaxy S9+ (Samsung Exynos 9810 + Mali-G72 MP18 Bifrost GPU)
 * Actuator MCU: STM32F405 (UART/CAN control interface)
 * Framework: Tencent ncnn (Vulkan GPU Acceleration & AHardwareBuffer Zero-Copy)
 * ============================================================================
 */

#include <android/hardware_buffer.h>
#include <android/asset_manager.h>
#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>

// NCNN includes
#include "net.h"
#include "gpu.h"
#include "command.h"
#include "pipeline.h"
#include "layer.h"
#include "layer_type.h"

// Diagnostic image bands only. These are image-relative regions, not safety distances.
// Zone 0: NEAR (bottom_y > 0.70)
// Zone 1: MID  (0.35 <= bottom_y <= 0.70)
// Zone 2: FAR  (bottom_y < 0.35)
enum ObstacleZone {
    ZONE_NEAR = 0,
    ZONE_MID  = 1,
    ZONE_FAR  = 2
};

struct Object {
    int zone_id;       // 0: Near image band, 1: Mid image band, 2: Far image band
    int label;         // Class index (COCO or custom mower obstacle class)
    float prob;        // Detection confidence [0.0, 1.0]
    float x;           // Bounding box left (pixel coordinate)
    float y;           // Bounding box top (pixel coordinate)
    float w;           // Bounding box width (pixel)
    float h;           // Bounding box height (pixel)
};

struct GridAndStride {
    int grid0;
    int grid1;
    int stride;
};

class Yolo11 {
public:
    Yolo11();
    ~Yolo11();

    /**
     * Initializes NCNN, Vulkan compute engine, Mali-G72 shader optimizations,
     * and loads model weights from Android assets.
     */
    int load(AAssetManager* mgr, const char* modeltype, int target_size, bool use_gpu = true);

    /**
     * Imports a Camera2 AHardwareBuffer into Vulkan and runs YOLO on its VkMat.
     * The caller keeps the originating Image alive for this call. Cache entries
     * retain their own AHardwareBuffer reference beyond the call.
     */
    int detect_ahb(AHardwareBuffer* ahb, float conf_thresh, float nms_thresh,
                   std::vector<Object>& objects, int rotate_type = 1);

    /**
     * Packs detection results into a flat primitive float array for zero-overhead JNI transfer.
     * Layout: [count, zone_id0, class0, score0, norm_x0, norm_y0, norm_w0, norm_h0, zone_id1, ...]
     */
    int pack_detections_flat(const std::vector<Object>& objects, float* out_buffer, int max_capacity);

    /**
     * Diagnostics: Returns highest raw score and class seen in last frame (before conf_thresh filtering)
     */
    float get_top_candidate_score() const { return last_top_score; }
    int get_top_candidate_class() const { return last_top_class; }

    /**
     * Clears all cached Vulkan pipelines and GPU resources.
     */
    void clear_ahb_cache();

    /**
     * Returns configured model target resolution.
     */
    int get_target_width() const { return target_w; }
    int get_target_height() const { return target_h; }
    bool is_ready() const { return model_loaded; }

private:
    void generate_grids_and_stride(int target_w, int target_h);
    void decode_dfl_and_filter(const ncnn::Mat& out_blob, float conf_thresh);
    void nms_sorted_bboxes(float nms_thresh, std::vector<Object>& output);

    // NCNN network & Vulkan device
    ncnn::Net net;
    ncnn::VulkanDevice* vkdev;
    ncnn::Layer* scale_layer;

    // Cache entries are keyed by the actual AHardwareBuffer pointer. Each entry
    // owns an AHB reference and keeps allocator, imported VkImageMat and pipeline
    // alive together for the lifetime of the camera buffer.
    struct AHBImportCacheEntry {
        AHardwareBuffer* ahb;
        ncnn::VkAndroidHardwareBufferImageAllocator* allocator;
        ncnn::ImportAndroidHardwareBufferPipeline* pipeline;
        ncnn::VkImageMat src;
        int rotate_type;
        uint64_t last_used;

        AHBImportCacheEntry()
            : ahb(nullptr), allocator(nullptr), pipeline(nullptr),
              rotate_type(1), last_used(0) {}
    };

    static constexpr size_t AHB_CACHE_LIMIT = 8;
    std::unordered_map<AHardwareBuffer*, AHBImportCacheEntry*> ahb_import_cache;
    uint64_t ahb_cache_clock;
    uint64_t ahb_cache_hits;
    uint64_t ahb_cache_misses;
    uint64_t ahb_cache_evictions;
    uint64_t ahb_cache_rotate_rebuilds;
    void destroy_ahb_cache_entry(AHBImportCacheEntry* entry);

    // Pre-allocated static scratchpads to guarantee ZERO allocations in inference loop
    std::vector<GridAndStride> grid_strides;
    std::vector<Object> proposal_scratchpad;
    std::vector<Object> candidates_scratchpad;
    std::vector<int> nms_indices_scratchpad;

    // Model configurations
    int target_w;
    int target_h;
    int num_classes;
    bool model_loaded;
    bool use_vulkan_compute;

    // Live diagnostics for raw score inspection
    float last_top_score;
    int last_top_class;

    // Statically allocated softmax distribution lookup / scratchpad for DFL
    static const int REG_MAX = 16;
    float dfl_distribution_scratchpad[REG_MAX];
};

#endif // YOLO11_H