# Async frame compute: implementation contract

## Target

Implement frame-level N/N+1 overlap for the YOLO11 AHB path on the pinned NCNN revision:

- GPU worker: AHB import + preprocess + YOLO graph prefix through layer 109 for frame N+1.
- CPU worker: graph suffix from layer 110 through layer 256 for frame N.
- No GPU -> CPU -> GPU island around layers 110-113.
- Keep the original .param and .bin intact. The split is a runtime execution boundary, not a physical model split.

## Verified NCNN contracts

Pinned NCNN source: 6a1bf000f363714839a36793addc8c879d3d899e.

- Extractor::extract(int, VkMat&, VkCompute&) and Extractor::input(int, const Mat&) exist in src/net.h.
- VkCompute::submit_and_wait() obtains a queue through VulkanDevice::acquire_queue(), submits, returns the queue to the pool, then waits on its own fence.
- VulkanDevice::acquire_queue() and reclaim_queue() protect queue-pool ownership with NCNN mutex/condition-variable locks. Two independent VkCompute objects can therefore submit from different host threads without concurrent host access to the same leased queue handle.
- VkBlobAllocator and VkStagingAllocator are not internally synchronized for concurrent use. Each in-flight GPU frame must own distinct blob and staging allocator instances. NCNN's device-level allocator pools are synchronized when allocators are acquired/reclaimed.
- Extractor owns per-execution blob state. Keep the GPU Net confined to the GPU worker and the CPU Net confined to the CPU worker; do not run the same Net concurrently from both workers.

## Graph cut

The GPU side must materialize these ten boundary blobs. Their producers are all at or before layer 109; their consumers are in the CPU suffix:

| Blob | Producer | First consumer |
|---:|---:|---:|
| 37 | 31 | 145 |
| 72 | 60 | 130 |
| 123 | 102 | 125 |
| 125 | 103 | 119 |
| 128 | 105 | 114 |
| 129 | 105 | 115 |
| 133 | 107 | 110 |
| 135 | 108 | 115 |
| 136 | 108 | 113 |
| 137 | 109 | 110 |

The GPU Extractor must run with light mode disabled so that boundary values remain available while the ten outputs are collected. Record the ten downloads into one VkCompute command context and wait once at the boundary. The CPU Extractor receives those ten Mat values through input(blob_index, Mat) before extracting out0.

## Frame ownership and scheduling

Use two explicit frame slots with the state transition:

FREE -> GPU_RUNNING -> CPU_READY -> CPU_RUNNING -> FREE.

- One serial GPU worker owns AHB import cache access, GPU Net, GPU Extractors, VkCompute objects, and per-slot Vulkan allocators.
- One serial CPU worker owns CPU Net, CPU Extractors, postprocessing scratchpads, and detection output for its active slot.
- Slot 0 and slot 1 have independent boundary Mat storage and output buffers.
- A slot cannot be reused until CPU suffix/postprocessing has completed.
- The camera Image and Java HardwareBuffer wrapper must remain alive through GPU prefix completion. They may be closed immediately after the prefix fence completes because the CPU suffix consumes only copied boundary Mat objects.
- AHardwareBuffer_acquire() extends object lifetime; it does not prove that GPU reads have completed. Fence completion is the content-lifetime boundary.

The overlap is between separate stage workers. NCNN's public VkCompute::submit_and_wait() is synchronous; this design does not pretend it is an asynchronous submit API.

## Implementation status in this branch

The opt-in compute mode 3 now has a first implementation pass:

- Native Yolo11 owns a Vulkan Net for the prefix and a CPU-only Net for the suffix. The async mode refuses any model other than the pinned YOLO11n graph signature and checks the ten boundary producers plus Vulkan capability for layers 1-109.
- gpu_prefix_frame() imports/preprocesses AHB, evaluates the prefix, records ten downloads in one VkCompute context, waits for completion, and transitions the frame slot to CPU_READY.
- cpu_suffix_frame() injects those ten host Mats into the CPU Net, extracts out0, then performs DFL decode and NMS.
- JNI now snapshots a shared_ptr under g_lock and releases the lock before inference. Detection vectors and output arrays are per-thread/per-call rather than global shared buffers.
- Kotlin now has one serial GPU executor, one serial CPU executor, two reserved frame slots, per-slot output arrays, and an explicit producer-before-consumer drain on mode changes and teardown.
- Camera Image/HardwareBuffer lifetime ends after the GPU prefix returns. ImageReader teardown is delayed until the GPU worker has released all acquired Images.

Ownership assumptions that remain deliberate:

- The AHB cache is owned by the single serial GPU worker. ClearCache/Destroy must only run after that worker has drained.
- CPU postprocess scratchpads and last-score fields are shared members, but only the single serial CPU worker touches them in async mode.
- The same Net is never shared between the GPU and CPU workers.
- This is stage-level asynchronous overlap, not a nonblocking NCNN Vulkan submit API. The GPU prefix call waits for its boundary fence before the CPU suffix is queued.

The mode is opt-in and is not the default. The code has not been built or run on the target Android device yet. The validation gates below are still mandatory before treating the implementation as correct.

## AHB correctness patch

patches/ncnn/0001-ahb-normalized-ycbcr-sampling.patch targets only the pinned NCNN commit. It:

1. sets unnormalizedCoordinates = VK_FALSE when YCbCr conversion is active;
2. converts shader coordinates from pixel-space to normalized texel-center UVs;
3. uses the half-pixel resize mapping used by ncnn::resize_bilinear_c3();
4. removes the misplaced VkExternalFormatANDROID from VkSamplerYcbcrConversionInfo attachments while preserving it in image creation and YCbCr conversion creation chains;
5. derives chroma/min/mag filter choices from the AHB format feature flags. The supplied Mali-G72 log reports formatFeatures=0x00861001: linear YCbCr filtering is present, separate reconstruction filtering is absent, so the sampler and chroma filters must match. The patch selects LINEAR for both on that reported format.

Apply only to the exact NCNN checkout with:

    ./scripts/apply_ncnn_ahb_normalized_sampling.sh /path/to/ncnn

The script refuses any other NCNN HEAD. It is separate from apply_ncnn_runtime_trace.sh; apply it to the same checkout after the runtime trace script if that diagnostic patch is present.

## Validation gates before enabling async mode by default

1. Build NCNN with the AHB patch and run Vulkan validation if the driver exposes it.
2. Compare the same captured AHB frame against CPU YUV conversion and ordinary Vulkan input. Test rotations 1, 3, 6, and 8, plus resize and no-resize cases.
3. Validate boundary tensor dimensions, element sizes, packing, and finite-value/hash checks before CPU suffix execution.
4. Compare Point B (GPU 1-109 + CPU 110-256) detections and out0 against the full CPU baseline on identical input frames.
5. Only then enable the two-slot N/N+1 scheduler and measure overlap, dropped frames, queue wait, CPU suffix time, and end-to-end frame latency separately.

No .param/.bin split, precision-policy change, or broad Logcat expansion is part of this patch.
