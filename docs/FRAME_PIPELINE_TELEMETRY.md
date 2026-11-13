# Full-GPU AHB frame telemetry

This milestone is a **measurement and UI-consolidation step** on top of
`fix/matmul-runtime-timestamps`. The selected runtime path is fixed to
Vulkan + Camera2 AHardwareBuffer + Vulkan MatMul (compute mode 8). The
compute-mode selector has been removed from the Android activity UI. The
repository's `dashboard.html` did not contain a compute-mode selector.

This milestone does **not** implement multi-frame GPU submission yet. The
Camera2 listener still uses `isProcessingFrame` as a single-frame gate. That
is intentional: timing data must be collected before changing submission,
resource lifetime, and slot ownership rules.

## Log groups

Collect a complete trace with:

```sh
adb logcat -v threadtime | grep -E 'PipelineStages|JNI_FRAME_TIMING|APP_FRAME_TIMING|NCNN_VK_SYNC|NCNN_VK_FLUSH|NCNN_BENCHMARK_GPU|NCNN_BENCHMARK|AHB_CACHE|MATMUL_ROUTE'
```

Aggregate frame logs are sampled every 30 frames. Vulkan submit/wait and forced
flush logs are emitted per event and can perturb timings; compare the measured
run with a normal run without verbose tracing.

### Native frame stages: `PipelineStages`

Emitted every 30 frames by `Yolo11::detect_ahb()`.

- `allocator_acquire`: acquire NCNN blob and staging allocators.
- `cache_lookup`: AHB import-cache lookup.
- `cache_setup`: cache miss work (allocator construction, image import, and
  import-pipeline creation).
- `preprocess_record`: CPU time recording the AHB import, resize, and packing
  commands into the preprocessing command buffer.
- `import_record`, `scale_record`, `cast_record`: sub-times within
  `preprocess_record`; do not add these again to the total.
- `dispatches`: dispatch workgroup count recorded by preprocessing. It is not
  a count of frames, dispatch calls, or milliseconds.
- `preprocess_submit_wait_wall`: wall time of preprocessing
  `VkCompute::submit_and_wait()`. It is not a pure fence-wait measurement.
- `preprocess_total`: preprocessing record time plus preprocessing
  submit-and-wait wall time.
- `extractor_setup`: CPU time to construct the NCNN Extractor and attach the
  frame blob/workspace/staging allocators.
- `extractor_input`: CPU wall time of `ex.input("in0", in_gpu)`, which binds
  the preprocessed Vulkan tensor as the graph input.
- `infer_extract_wall`: wall time of `ex.extract("out0", out)`. It includes
  graph traversal/execution, any internal command-buffer submissions and
  synchronization, and output readback to CPU `ncnn::Mat`. It is not the
  GPU-only execution time.
- `dfl_decode`: CPU DFL decode/filter after `out0` is available.
- `nms`: CPU sort and non-maximum suppression.
- `post_total`: `dfl_decode + nms`.
- `native_frame_total`: wall time for the native AHB frame path.
- `unaccounted_wall`: residual between the frame total and the explicitly
  accounted stages. Treat it as **unattributed time**, not automatically as a
  wait or scheduler stall.

The GPU time of individual inference layers is logged separately as
`NCNN_BENCHMARK_GPU`. Preprocessing import/scale/cast GPU durations are not
yet split into per-operation GPU timestamps in this milestone; those are
currently covered by the preprocessing submit-and-wait wall interval.

### Per-layer GPU timing: `NCNN_BENCHMARK_GPU`

Mode 8 now enables the existing NCNN benchmark path. With the pinned NCNN
build configured with `NCNN_BENCHMARK=ON`, the final extractor submit emits
one record per available layer timestamp pair:

```text
[NCNN_BENCHMARK_GPU] idx=... type=... name=... duration=...us
```

These are GPU timestamp intervals around layer dispatches, not CPU wall time.
They should be summed only for the measured GPU layer set; a sum does not
include all host-side work, queue gaps, preprocessing, or readback.

If a layer in a Vulkan graph cannot execute on Vulkan and is evaluated by the
CPU path, the patched NCNN runtime emits:

```text
[NCNN_BENCHMARK_CPU] idx=... type=... name=... duration=...ms
```

That is the explicit per-layer CPU fallback record. In the intended mode 8
full-GPU graph, this log should not appear. MatMul routing is also recorded
separately through `MATMUL_ROUTE` / `MATMUL_ROUTE_DONE`.

### Vulkan synchronization: `NCNN_VK_SYNC`

Emitted by the patched `VkCompute::submit_and_wait()` for each successful
submit/wait transaction.

- `record_replay_ms`: CPU time replaying deferred Vulkan command records when
  the device does not support push descriptors.
- `command_buffer_end_ms`: host time ending the command buffer.
- `queue_acquire_ms`: host time obtaining an NCNN compute queue.
- `queue_submit_ms`: time spent inside the `vkQueueSubmit()` call. This is
  submission-call latency, not GPU execution time.
- `fence_wait_wall_ms`: wall time inside `vkWaitForFences()`. This usually
  includes waiting for the submitted GPU work to complete. It is **not** pure
  fence overhead.
- `post_download_cpu_ms`: CPU-side invalidation, mapped-memory copies, and
  FP16/BF16-to-FP32 casts performed after the fence signals.
- `total_wall_ms`: total wall time of this `submit_and_wait()` invocation.
- `pending_dispatches`: accumulated workgroup count at entry, not elapsed time.

A Vulkan staging transfer itself executes on the GPU and is therefore part of
the submitted GPU work and fence-wait interval. The post-download field only
measures the subsequent CPU-side handling.

### NCNN forced flushes: `NCNN_VK_FLUSH`

NCNN can force `submit_and_wait()` inside graph traversal for two existing
reasons:

1. `cpu_layer_download`: a CPU layer needs a GPU-produced input downloaded.
2. `dispatch_threshold`: the pending workgroup count crossed NCNN's
   rough-score-dependent driver-protection threshold.

The log reports the layer index/name, rough score, pending dispatch count,
threshold, and trigger. `NCNN_VK_FLUSH_DONE` reports the complete
submit-and-wait transaction wall time. The threshold is deliberately retained;
this milestone instruments it rather than disabling driver protection.

### JNI and application timing

`JNI_FRAME_TIMING` separates:

- `detect_frame_ms`: complete native detection path, including preprocessing,
  inference, output readback, DFL and NMS.
- `array_pack_copy_ms`: packing detections and copying the result to the
  Java float array.
- `jni_total_ms`: JNI call wall time.

`APP_FRAME_TIMING` separates:

- `callback_to_task_ms`: Camera2 callback start to executor task start.
- `executor_queue_ms`: time waiting in the selected executor queue.
- `native_detect_wall_ms`: Java-side wall time around `DetectFrame()`.
- `ahb_wrapper_close_ms`: time closing the Java `HardwareBuffer` wrapper.
- `json_build_ms`, `websocket_broadcast_ms`, `stm32_send_ms`: application
  output work after detection.
- `callback_to_done_ms`: Camera2 callback-to-task completion wall time.
- `busy_dropped_total`: number of frames discarded while a previous frame
  holds the single-frame processing gate.

The native/JNI/app fields are nested measurements, not additive buckets.
For example, `native_detect_wall_ms` contains the native `PipelineStages`
and `JNI_FRAME_TIMING.detect_frame_ms` intervals. Likewise,
`extractor_setup`, `extractor_input`, and `infer_extract_wall` are separate
sequential portions of the native frame path; the per-layer GPU timestamps and
Vulkan submit/fence timings occur inside the Extractor interval and must not be
added to it as if they were extra frame stages.

## Current synchronization boundary

The pipeline remains:

```text
Camera2 AHB
  -> Vulkan preprocessing VkCompute
  -> submit_and_wait
  -> NCNN Extractor: full Vulkan graph + Vulkan MatMul
  -> out0 readback to CPU Mat
  -> DFL decode
  -> NMS
  -> JNI array copy
  -> telemetry / application output
```

The preprocessing wait is still a hard host synchronization boundary. The
Extractor path also waits as needed before returning `out0`. The next milestone
can use the collected data to decide which boundary to remove first and then
introduce per-frame slots/fences without allowing any slot's AHB, VkMat,
allocator, or output buffers to be reused before completion.

## Reading the trace

For a given frame ID:

1. Find `PipelineStages` for native stage totals.
2. Find `NCNN_BENCHMARK_GPU` records emitted by the final extractor for
   per-layer GPU execution.
3. Correlate `NCNN_VK_SYNC` sequence events to separate queue submission,
   GPU completion wait, and CPU post-readback work.
4. Use `NCNN_VK_FLUSH` to identify internal graph flushes and why they happen.
5. Compare `JNI_FRAME_TIMING` and `APP_FRAME_TIMING` for bridge/application
   overhead and dropped-frame pressure.

Do not add the nested wall-time measurements together. Use the layer timestamps
for GPU layer time and the outer frame wall time for throughput/latency.
