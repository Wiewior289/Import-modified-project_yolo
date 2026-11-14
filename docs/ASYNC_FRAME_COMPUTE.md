# Async frame compute: implementation contract

## Architecture

Mode 8 is full-GPU YOLO11 inference with two frame slots and three ordered stages:

1. GPU submit worker records AHB import, preprocessing, the complete NCNN Vulkan graph (including MatMul 110 and 113), and final `out0` download, then submits without waiting.
2. GPU completion worker waits for the corresponding frame fence, performs deferred readback/casts, reads timestamp queries, releases frame resources and unpins the AHB cache entry.
3. CPU worker runs DFL decode/filtering and NMS after `out0` is ready.

Modes 3/4 remain legacy GPU-prefix/CPU-suffix diagnostic paths. Mode 8 does not delegate MatMul to CPU. The model `.param` and `.bin` are unchanged.

## Pinned NCNN source contracts

Pinned upstream NCNN: `6a1bf000f363714839a36793addc8c879d3d899e`.

- `Extractor::extract(int, VkMat&, VkCompute&)` records graph work into the caller's command context; it does not submit or wait.
- The current milestone source already records AHB import, Scale, cast, Extractor input/graph and final `out0` download into one `VkCompute`. There is no separate preprocessing fence in this path.
- `VkCompute::submit_and_wait()` ends/replays the command buffer, acquires a compute queue, submits, returns the queue handle to the device pool, and then waits on that VkCompute fence.
- The async patch adds `VkCompute::submit()` and `VkCompute::wait()`. `submit()` reuses NCNN's existing recording/replay and queue-submit code but returns immediately after successful `vkQueueSubmit`. `wait()` waits on that frame's fence and performs the deferred download/cast records.
- A submitted context cannot be reused before its fence completes and it is reset. `reset()` rejects a pending submission. The private destructor has a last-resort fence wait, but normal code must call `wait()` so deferred readback is materialized.
- `VulkanDevice::acquire_blob_allocator()` and `acquire_staging_allocator()` return a free allocator from their pools or create another. Their reclaim functions return it to the pool. The individual VkBlobAllocator/VkStagingAllocator instances do not provide per-instance allocation locking; each in-flight frame retains its own allocator until its fence and readback complete.
- Queue-pool acquire/reclaim is synchronized. The queue handle is returned immediately after `vkQueueSubmit`, so the host may submit another frame while the previous submission is executing. GPU submissions on the same queue remain ordered.

## Per-frame resource lifetime

Each async slot retains its `VkCompute`, `Extractor`, AHB-scaled/input/output VkMats, blob and staging allocators, timestamp-query state, and a pointer to its AHB cache entry. The entry is pinned by an atomic `in_flight` count.

Only after `VkCompute::wait()` succeeds does the completion worker:
- release Extractor and VkMat references;
- destroy the completed command context;
- return blob/staging allocators to the device pools;
- decrement the AHB cache pin.

The Camera2 `Image` and Java `HardwareBuffer` wrapper remain open from `FullGpuFrame()` until `WaitGpuFrame()` succeeds. An AHB reference protects object lifetime, not the camera producer's right to recycle the buffer contents. Cache eviction skips pinned entries; cache clear refuses to destroy pinned entries.

## Dispatch-threshold safety

NCNN internal dispatch-threshold flushes in `NetPrivate::forward_layer()` remain synchronous. Mode 8 raises small thresholds to a minimum of 512K to avoid repeated small flushes seen in earlier traces. Existing higher rough-score tiers (for example 1MB) remain unchanged. Any remaining flush must be visible through `NCNN_VK_FLUSH`.

`pending_dispatch_total` is accumulated X×Y×Z workgroup count since the last submit/reset. It is not a dispatch-call count, frame count, or duration. If a graph is split by flushes, each flush reports only its current segment.

## Worker synchronization

The GPU submit worker and GPU completion worker are separate executors. Waiting for frame N's fence must not block recording/submission of frame N+1. CPU postprocessing is queued only after the corresponding fence and deferred readback. CPU interval N can overlap GPU execution of N+1.

This is host-side pipeline overlap, not simultaneous execution of two inference command buffers on one GPU queue. Vulkan still orders submissions on that queue. The goal is to keep it fed while host readback and CPU postprocessing happen independently.

If both slots are occupied, a new camera frame is dropped rather than reusing a slot still owned by a worker. Reinitialization drains GPU submission, then GPU completion, then CPU tasks before destroying native state.

## Historical logs and branch history

The old `yolo_layers.txt` baseline showed `CPU_FALLBACK_DOWNLOAD` before MatMul 110 and 113. That is a real GPU→CPU→GPU boundary: the GPU-produced inputs had to be downloaded and synchronized before CPU MatMul; downstream GPU work could not consume the CPU result until it was uploaded again.

The current `milestone_matmul_diag(1).txt` shows `support_vulkan=1` for both MatMul layers, whereas the baseline showed `vulkan=0`. `VULKAN_CAPABLE_RUNTIME_UNVERIFIED` is the graph inventory's conservative label, not a CPU fallback marker and not runtime dispatch proof.

The closest tracked branch head before the 9-Nov diagnostic log is `feat/async-frame-compute`, commit `129efb9157d1ad71ce9bd62694948dd160559324`. Its tracked change is boundary-blob validation in `yolo11.cpp`; it does not add the `NCNN_RT` logger. The diagnostic branch's `NCNN_RUNTIME_TRACE.md` states that it did not modify NCNN source and used `NCNN_BENCHMARK=OFF`. I did not find the `NCNN_RT`/`TENSOR_TAP` implementation in the tracked NCNN patches examined, so those markers in `yolo_layers.txt` came from a local/untracked NCNN instrumentation build or another patch absent from the repository history.

The tracked instrumentation provides direct runtime evidence:
- `MATMUL_ROUTE` / `MATMUL_ROUTE_DONE`: actual route and tensor residency for layers 110/113.
- `MATMUL_VK_FORWARD` / `MATMUL_VK_DISPATCH_RECORDED`: custom MatMul Vulkan forward and recorded dispatch.
- `NCNN_BENCHMARK_CPU` / `NCNN_BENCHMARK_GPU`: CPU timings and Vulkan timestamp timings.
- `NCNN_VK_SYNC` / `NCNN_VK_FLUSH`: queue/fence timing and reason/size of synchronous threshold flushes.

The build script previously checked `runtime_sync_timing.patch` but did not apply it, while including it in `PATCHSET_SHA`. It now applies that patch and then `async_vk_submit.patch`; the patchset hash includes both.

## Model input resolution contract

`modelInputSize` is the single model-resolution setting in Kotlin. The same value is passed through `Yolo11Ncnn.Init()` and JNI as `model_input_size`, stored once in `Yolo11`, and reused for AHB preprocessing output, NCNN input dimensions, parser grid generation, DFL coordinate clipping and detection normalization. The supported square sizes remain 320..480 in steps of 32. The parser grid is built from that same value with strides 8, 16 and 32; for 480×480 this gives 4,725 anchors.

The Camera2 YUV output is a separate source stream (`cameraSourceSize`), not the model tensor size. Its selection is now based on the chosen `modelInputSize` instead of a fixed 320×320 area target. The importer still resizes the source AHardwareBuffer to the one canonical model input size before inference. The log `CAMERA2_STREAM_CONFIG` prints both dimensions so a HAL-supported source size is not mistaken for the parser input.

## Runtime telemetry

- `ASYNC_FRAME_QUEUED`: admitted frame, camera callback time, queue time and source dimensions.
- `ASYNC_GPU_SUBMIT_BEGIN/DONE`: GPU submit executor start/end, including executor queue delay. This is host-side recording and queue-submit time, not GPU execution time.
- `ASYNC_GPU_WAIT_BEGIN/ASYNC_GPU_COMPLETE_DONE`: completion executor start/end around native fence wait, deferred readback and resource release; includes completion-executor queue delay.
- `ASYNC_CPU_STAGE_BEGIN/DONE`: CPU postprocess executor start/end, including CPU executor queue delay.
- `ASYNC_FRAME_SLOT_RELEASE`: end-to-end host time from frame admission until the frame slot becomes reusable.
- `NCNN_VK_ASYNC_SUBMIT`: command-buffer replay/end, queue acquisition and queue-submit timing.
- `NCNN_VK_ASYNC_WAIT`: fence wait and deferred CPU readback/cast timing.
- `ASYNC_GPU_COMPLETE_DONE`: completion worker interval including fence wait and resource release.
- `ASYNC_CPU_STAGE_DONE`: DFL/NMS interval.
- `FRAME_ALLOCATORS_ACQUIRED/RECLAIM`: per-frame allocator leases.
- `AHB_CACHE_PIN/UNPIN`: AHB import lifetime pin count.

## Reading N/N+1 overlap

The Kotlin `start_ns`/`end_ns` fields use `SystemClock.elapsedRealtimeNanos()` and are comparable across the three executors. If the interval for `ASYNC_GPU_SUBMIT_BEGIN/DONE` of frame N+1 intersects the interval `ASYNC_GPU_WAIT_BEGIN/ASYNC_GPU_COMPLETE_DONE` of frame N, the host submitted N+1 while N's fence was still pending. If CPU stage N intersects submit or wait activity for N+1, the CPU postprocess is overlapping the next frame's host/GPU work. `executor_queue_wait_ms` / `completion_queue_wait_ms` / `queue_wait_ms` separate worker backlog from the stage's own duration.

This establishes host-side N/N+1 pipelining and queue submission before N completes. It does not mean two inference command buffers execute simultaneously: submissions on the same Vulkan queue remain ordered.

In full-GPU mode 8, runtime per-layer benchmarking is disabled (`net.opt.use_benchmark=false`). This suppresses per-layer CPU/GPU timings, MatMul shader dispatch details and timestamp-query allocation/readback. The low-volume `[MATMUL_ROUTE]` marker remains active for the MatMul layers, so runtime Vulkan-vs-CPU route selection can still be checked without collecting a timing line for every model layer. Compare/control modes retain their benchmark switch. The build may keep `NCNN_BENCHMARK=ON`; detailed profiling is simply inactive in mode 8.

## Validation gates

1. Build the pinned NCNN source with the complete patchset and rebuild the APK; do not reuse old Gradle/NDK native outputs.
2. Confirm `NCNN_VK_ASYNC_SUBMIT` and `NCNN_VK_ASYNC_WAIT` appear.
3. Confirm layers 110/113 produce `MATMUL_ROUTE`, `MATMUL_VK_DISPATCH_RECORDED`, and GPU timestamp pairs; verify no `NCNN_BENCHMARK_CPU` for the full mode-8 graph.
4. Check `NCNN_VK_FLUSH` at the selected resolution; compare workgroup count with the selected threshold (minimum 512K; higher tiers are preserved).
5. Compare detections and `out0` against the previous synchronous full-GPU path on identical AHB frames.
6. Confirm CPU interval N overlaps GPU execution/completion of N+1 and that AHB pins and per-frame allocator ownership remain valid.
7. Test rotations 1, 3, 6, 8 and resize/no-resize for long enough to detect allocator growth, stale AHB reuse, fence failures and slot starvation.

The async source changes are an implementation candidate until the pinned NCNN patch applies, the Android build succeeds and target-device validation passes.
