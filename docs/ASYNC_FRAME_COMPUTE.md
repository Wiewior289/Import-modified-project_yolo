# Async frame compute: implementation contract

## Target and mode separation

The active mode-8 path is full-GPU YOLO11 inference with bounded two-slot frame overlap:

- GPU worker: Camera2 AHB import, preprocessing, the complete NCNN Vulkan graph (including MatMul 110 and 113), and final `out0` readback for frame N+1.
- CPU worker: DFL decode/filtering and NMS for frame N.
- MatMul is not delegated to CPU in mode 8.
- The original YOLO `.param` and `.bin` are unchanged.

Modes 3/4 are separate legacy diagnostic paths. They split the graph at the CPU island around layers 110–113. They are not the mode-8 execution path.

## Pinned NCNN contracts

Pinned upstream NCNN source: `6a1bf000f363714839a36793addc8c879d3d899e`.

- `Extractor::extract(int, VkMat&, VkCompute&)` records the requested graph dependency closure into the caller's command context. It does not create a separate command context or wait for completion.
- `VkCompute::record_download(const VkMat&, Mat&, const Option&)` is the same readback primitive used by the high-level `Extractor::extract(int, Mat&)` implementation in this pinned source. It records conversion to a staging tensor and a device-to-staging transfer. After the fence, NCNN invalidates mapped memory, copies the tensor, and performs FP16/BF16-to-FP32 conversion where required.
- `VkCompute::submit_and_wait()` ends the command buffer, acquires a compute queue, submits it, returns the queue to the pool, and then waits on that VkCompute object's fence. The queue handle is therefore not held during the fence wait.
- `VkCompute::reset()` releases staging/download buffers and descriptor sets, processes image command references, and resets the command buffer and fence. It must not run while commands that reference those resources are still in flight.
- NCNN's Vulkan blob and staging allocator instances are not internally synchronized for concurrent use. Each independently in-flight GPU execution needs distinct allocator instances. The device-level acquire/reclaim pools provide the synchronization for allocator ownership.
- The Extractor owns per-execution blob state. The same Net must not be concurrently driven from GPU and CPU workers.

## Historical CPU-island graph cut (modes 3/4 only)

The older GPU-prefix/CPU-suffix implementation used the following ten boundary blobs. Their producers are at or before layer 109; their consumers are in the CPU suffix:

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

The later `feat/gpu-cpu-gpu-ring-extractor` experiment narrowed the CPU island input transfer to blobs 133, 137, and 136. It retained seven GPU carry blobs (37, 72, 123, 125, 128, 129, 135) in a persistent per-frame GPU Extractor, with per-slot Vulkan allocators. The CPU delegate produced bridge blob 141, which was injected into the same Extractor for GPU continuation.

That design directly addresses the lifetime dependency in a GPU→CPU→GPU graph: the GPU-produced tensors needed after the CPU island remain alive while the CPU computes the bridge tensor. It is not merged into mode 8 because the ring experiment delegates MatMul to CPU, while mode 8 keeps MatMul 110 and 113 on Vulkan.

## Mode-8 command flow

The mode-8 frame path records preprocessing and the full model graph into one `VkCompute` context:

```text
Camera2 AHB
  -> record_import_android_hardware_buffer
  -> Scale
  -> convert_packing / input cast
  -> Extractor::input("in0", VkMat)
  -> Extractor::extract("out0", VkMat, same VkCompute)
  -> record_download(out0, CPU Mat)
  -> submit_and_wait / final fence
  -> CPU DFL decode + NMS
```

The explicit preprocessing `submit_and_wait()` boundary has been removed. The commands for AHB import, scale, cast, and the model graph are recorded in order into the same command context. The final `out0` readback still requires a completed fence before the CPU can consume the host Mat.

NCNN's internal dispatch-threshold flushes remain synchronous. When the pending workgroup count crosses the rough-score-derived threshold, NCNN submits the currently recorded command buffer, waits for its fence, resets the command context, and continues graph recording. This guard is intentionally retained.

The application uses two slots and two serial workers:

```text
GPU worker (serial): frame N+1 full Vulkan inference -> slot[N+1].out0
CPU worker (serial): slot[N].out0 -> DFL/NMS -> detections
```

The GPU worker owns the AHB cache, Vulkan Net/Extractor, and Vulkan allocators. The CPU worker owns DFL/NMS scratchpads, score state, and detection packing. The deferred full-GPU path does not modify the CPU postprocessing scratchpads or score fields. Each slot retains its own `out0` Mat until CPU postprocessing completes.

The mode-8 listener bypasses the synchronous single-frame gate. If both slots are occupied, the incoming frame is dropped rather than reusing a slot still owned by a worker. Pipeline reinitialization drains GPU tasks first, then CPU tasks, before destroying native state. Teardown uses the same producer-before-consumer ordering.

This is stage-level frame overlap, not two simultaneously executing GPU inference graphs and not a nonblocking NCNN submit API.

## What the traces establish

The old `yolo_layers.txt` CPU-delegated baseline showed `CPU_FALLBACK_DOWNLOAD` before MatMul 110 and 113. Those transitions required downloading GPU tensors to CPU, waiting for GPU completion, executing CPU MatMul, and later uploading CPU-produced tensors for dependent GPU layers. That is a genuine GPU→CPU→GPU dependency.

The current full-GPU `ncnn_milestone.txt` contains `MATMUL_VK_DISPATCH_RECORDED` for the custom Vulkan MatMul implementation. Its `NCNN_VK_FLUSH` events are dispatch-threshold flushes, not CPU MatMul downloads. The old CPU-island dependency is therefore absent from the current mode-8 graph.

The queue handle is returned to NCNN's queue pool before `vkWaitForFences()`. The host thread still waits for the submitted GPU batch to finish. A gap between batches while the host wakes, resets/re-records, and submits the next chunk is plausible, but fence-wait duration alone does not prove GPU idle time. It must be measured with queue/submit timing or a separately controlled asynchronous submission experiment.

`pending_dispatch_total` is the accumulated workgroup count (sum of X×Y×Z group counts), not a count of dispatch calls, frames, or milliseconds. The threshold guard is retained.

## Telemetry and validation

Mode 8 records monotonic host timestamps every 30 frames:

- `ASYNC_FRAME_QUEUED.queued_ns`: slot reservation and GPU-task enqueue.
- `ASYNC_GPU_STAGE_DONE.start_ns/end_ns`: complete GPU-stage interval, including NCNN threshold flushes and final readback.
- `ASYNC_CPU_STAGE_DONE.start_ns/end_ns`: CPU DFL/NMS interval.

CPU interval N should intersect GPU interval N+1 when stage overlap is active. The stage wall times are host intervals, not pure GPU execution times. Use `NCNN_BENCHMARK_GPU` for per-layer GPU timestamps and `NCNN_VK_SYNC` / `NCNN_VK_FLUSH` for NCNN submission and wait analysis.

Validation gates:

1. Build the pinned NCNN source with the complete local patchset and rebuild the APK against that package. Do not reuse old `.cxx`/Gradle native build outputs.
2. Compare mode-8 detections and `out0` against the prior synchronous full-GPU path on identical captured AHB frames.
3. Verify `MATMUL_VK_DISPATCH_RECORDED` for layers 110 and 113, GPU timestamp pairs for both, and absence of `NCNN_BENCHMARK_CPU` records for the full graph.
4. Compare `preprocess_host_wait` (expected zero except perturbative tensor taps), `infer_record_wall`, `final_submit_wait_wall`, and `NCNN_VK_SYNC` counts/waits with the previous build.
5. Confirm CPU interval N overlaps GPU interval N+1. Check slot drops, AHB lifetime, slot reuse, and long-run stability.
6. Test AHB rotations 1, 3, 6, and 8 plus resize/no-resize.

The mode-8 source changes are committed but have not yet been built or executed on the target Android device. Treat them as an implementation candidate until these validation gates pass.

