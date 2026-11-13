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
#include <array>
#include <mutex>
#include <memory>
#include <atomic>

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
enum YoloComputeMode {
    YOLO_COMPUTE_CPU = 0,
    YOLO_COMPUTE_VULKAN = 1,
    YOLO_COMPUTE_VULKAN_AHB = 2,
    YOLO_COMPUTE_VULKAN_AHB_SPLIT = 3,
    YOLO_COMPUTE_VULKAN_AHB_ASYNC = 4,
    YOLO_COMPUTE_AHB_COMPARE = 5,
    YOLO_COMPUTE_MATMUL_VULKAN_COMPARE = 6,
    YOLO_COMPUTE_VULKAN_AHB_MATMUL_CPU = 7,
    YOLO_COMPUTE_MATMUL_VULKAN = 8
};

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
    int load(AAssetManager* mgr, const char* modeltype, int target_size,
             int requested_compute_mode = YOLO_COMPUTE_VULKAN_AHB);

    /**
     * Imports a Camera2 AHardwareBuffer into Vulkan and runs YOLO on its VkMat.
     * The caller keeps the originating Image alive for this call. Cache entries
     * retain their own AHardwareBuffer reference beyond the call.
     */
    int detect_frame(AHardwareBuffer* ahb, float conf_thresh, float nms_thresh,
                     std::vector<Object>& objects, int rotate_type,
                     int64_t frame_id, int64_t image_timestamp_ns);

    // Stage 1: synchronous GPU prefix and boundary readback. The camera buffer
    // may be released after this method returns successfully.
    int gpu_prefix_frame(int slot_index, AHardwareBuffer* ahb, int rotate_type,
                         int64_t frame_id, int64_t image_timestamp_ns);

    // Stage 2: CPU-only suffix and postprocess from the slot's ten boundary Mats.
    int cpu_suffix_frame(int slot_index, float conf_thresh, float nms_thresh,
                         std::vector<Object>& objects);

    // Full-GPU mode 8: finish the complete Vulkan graph and read back out0 into
    // a frame slot. CPU DFL/NMS is deliberately deferred to the CPU worker.
    int gpu_full_frame(int slot_index, AHardwareBuffer* ahb, int rotate_type,
                       int64_t frame_id, int64_t image_timestamp_ns);
    int gpu_wait_frame(int slot_index);
    int cpu_postprocess_frame(int slot_index, float conf_thresh, float nms_thresh,
                              std::vector<Object>& objects);

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
    struct AhbCompareSnapshot {
        ncnn::Mat rgb_0_255;
        ncnn::Mat input_after_norm;
        ncnn::Mat input_after_cast;
        ncnn::Mat out0;
        std::vector<Object> detections;
        float top_score = 0.f;
        int top_class = -1;
        double preprocess_ms = 0.0;
        double infer_ms = 0.0;
        double post_ms = 0.0;
    };

    int detect_ahb_compare(AHardwareBuffer* ahb, float conf_thresh, float nms_thresh,
                           std::vector<Object>& objects, int rotate_type,
                           int64_t frame_id, int64_t image_timestamp_ns);
    struct AsyncFrameSlot;
    struct AHBImportCacheEntry;

    int detect_ahb(AHardwareBuffer* ahb, float conf_thresh, float nms_thresh,
                   std::vector<Object>& objects, int rotate_type,
                   int64_t frame_id, int64_t image_timestamp_ns,
                   AhbCompareSnapshot* snapshot = nullptr,
                   ncnn::Mat* deferred_out0 = nullptr,
                   AsyncFrameSlot* async_slot = nullptr);
    int detect_cpu_or_vulkan(AHardwareBuffer* ahb, float conf_thresh, float nms_thresh,
                             std::vector<Object>& objects, int rotate_type,
                             int64_t frame_id, int64_t image_timestamp_ns,
                             AhbCompareSnapshot* snapshot = nullptr);
    void generate_grids_and_stride(int target_w, int target_h);
    void decode_dfl_and_filter(const ncnn::Mat& out_blob, float conf_thresh);
    void nms_sorted_bboxes(float nms_thresh, std::vector<Object>& output);

    // NCNN network & Vulkan device
    ncnn::Net net;
    ncnn::Net cpu_net;
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
        std::atomic<uint32_t> in_flight;

        AHBImportCacheEntry()
            : ahb(nullptr), allocator(nullptr), pipeline(nullptr),
              rotate_type(1), last_used(0), in_flight(0) {}
    };

    struct AsyncGpuExecution {
        std::unique_ptr<ncnn::VkCompute> command;
        std::unique_ptr<ncnn::Extractor> extractor;
        ncnn::VkMat ahb_scaled;
        ncnn::VkMat input;
        ncnn::VkMat output;
        ncnn::VkAllocator* blob_allocator = nullptr;
        ncnn::VkAllocator* staging_allocator = nullptr;
        AHBImportCacheEntry* cache_entry = nullptr;
        bool timestamp_queries_ready = false;
    };

    struct AsyncFrameSlot {
        enum State { FREE = 0, GPU_RUNNING = 1, GPU_SUBMITTED = 2,
                     CPU_READY = 3, CPU_RUNNING = 4 };
        std::mutex mutex;
        State state;
        std::array<ncnn::Mat, 10> boundary;
        ncnn::Mat out0;
        std::unique_ptr<AsyncGpuExecution> gpu_execution;
        int64_t frame_id;
        int64_t image_timestamp_ns;
        AsyncFrameSlot() : state(FREE), frame_id(0), image_timestamp_ns(0) {}
    };
    std::array<AsyncFrameSlot, 2> async_slots;

    static constexpr size_t AHB_CACHE_LIMIT = 12;
    std::unordered_map<AHardwareBuffer*, AHBImportCacheEntry*> ahb_import_cache;
    uint64_t ahb_cache_clock;
    uint64_t ahb_cache_hits;
    uint64_t ahb_cache_misses;
    uint64_t ahb_cache_evictions;
    uint64_t ahb_cache_rotate_rebuilds;
    uint32_t ahb_compare_samples_done;
    void destroy_ahb_cache_entry(AHBImportCacheEntry* entry);

    // Pre-allocated static scratchpads to guarantee ZERO allocations in inference loop
    std::vector<GridAndStride> grid_strides;
    std::vector<unsigned char> cpu_rgb_scratchpad;
    std::vector<Object> proposal_scratchpad;
    std::vector<Object> candidates_scratchpad;
    std::vector<int> nms_indices_scratchpad;

    // Model configurations
    int target_w;
    int target_h;
    int num_classes;
    bool model_loaded;
    bool cpu_net_loaded;
    bool use_vulkan_compute;
    int compute_mode;

    // Live diagnostics for raw score inspection
    float last_top_score;
    int last_top_class;

    // Statically allocated softmax distribution lookup / scratchpad for DFL
    static const int REG_MAX = 16;
    float dfl_distribution_scratchpad[REG_MAX];
};

#endif // YOLO11_H