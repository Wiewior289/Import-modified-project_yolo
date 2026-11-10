# NCNN backend comparison and profiling

## Compute modes

The selector exposes the existing compute modes plus a bounded same-frame AHB comparison mode:

| Mode | Camera input path | Model execution |
|---|---|---|
| CPU | AHardwareBuffer CPU lockPlanes -> YUV420 to RGB on CPU -> resize / normalize | NCNN CPU |
| GPU Vulkan | Same CPU-mapped YUV and CPU preprocessing as CPU mode | NCNN Vulkan |
| GPU Vulkan + AHB | Import the Camera2 AHardwareBuffer into Vulkan -> GPU YCbCr conversion / rotate / resize / normalize | NCNN Vulkan |
| DIAG: AHB vs CPU YUV + Vulkan | Same Camera2 Image/AHB, sequential AHB GPU preprocessing and CPU YUV preprocessing | Same loaded Net and Vulkan inference for both paths |

The CPU and GPU Vulkan baselines intentionally share the same input conversion and preprocessing code. Their model backend is the controlled variable. The AHB mode remains the direct-import implementation.

The CPU conversion currently assumes BT.601 limited-range YUV420. The Vulkan YCbCr sampler path can use a driver-selected conversion. The diagnostic mode compares both preprocessing paths from the same acquired Camera2 Image: AHB preprocessing first, then CPU plane locking/conversion, followed by Vulkan inference through the same loaded Net. It samples every 30th app frame, at most 10 samples per model initialization. RGB and normalized FP32 inputs are compared separately; AHB FP16 input statistics are logged on their own so dtype conversion is not confused with preprocessing error.

Every mode retains the high-level, known-good output path:
```cpp
ex.extract("out0", out);
```
Do not replace it with manual VkMat output conversion/download unless the tensor layout is independently validated.

## Layer capability inventory

At model load, the app logs one `NCNN_LAYER` record per network layer with:
- layer index, type and name
- `support_vulkan`
- `support_packing`
- `support_vulkan_packing`

This is a capability inventory, not measured execution placement. A layer that supports Vulkan may still use a CPU path under some runtime/configuration conditions.

## Enabling actual per-layer timings

NCNN's CPU and Vulkan per-layer timers are compiled into the NCNN library behind `NCNN_BENCHMARK`. The reference build supplied with this project reports `NCNN_BENCHMARK=0`. Defining the macro only for this Android app does not change an already-built `libncnn`.

Rebuild the NCNN Android arm64 package used by this project with at least:

```sh
-DNCNN_VULKAN=ON
-DNCNN_BENCHMARK=ON
-DNCNN_STRING=ON
```

Keep the same Android ABI, Android API level, NDK toolchain and Vulkan configuration as the package currently linked by the app. Point `ncnn_DIR` in the app build at the rebuilt package. Do not mix headers from one NCNN version with a library from another.

With the benchmark-enabled library:
- CPU-fallback layer durations are measured with host timing.
- Vulkan layer durations are measured with Vulkan timestamp queries.
- NCNN emits its own per-layer timing logs while the extractor runs.

The app cannot turn this facility on at runtime when the linked NCNN binary was built without it.

## What the current wall-clock measurements mean

- `preprocess_cpu`: CPU lock, plane conversion, rotation, resize and normalization for CPU / GPU Vulkan modes.
- `preprocess_record` and `preprocess_wait`: CPU command recording and the explicit preprocess queue wait for Vulkan+AHB.
- `infer_extract_wall`: wall time around `Extractor::extract("out0", out)`. For Vulkan this includes whatever graph work, internal queue submissions/waits, input upload (CPU-input Vulkan mode), output download and synchronization NCNN performs internally.
- `post`: DFL decode and NMS.
- `frame_total`: end-to-end native frame wall time.

The public Extractor API does not expose separate graph, queue-wait and output-readback durations. Splitting those accurately requires instrumentation inside the exact NCNN source version linked by the APK. NCNN's existing layer timestamps provide GPU layer execution time, but not a standalone readback duration. The app therefore reports `infer_extract_wall` and does not fabricate separate graph/wait/readback values.

A deeper Extractor instrumentation pass should place timers around:
1. graph evaluation into the GPU output VkMat,
2. output-download command recording,
3. final queue submit/wait,
4. GPU timestamp interval around the download command.

This diagnostic branch deliberately leaves NCNN source untouched and pins the linked library to the clean upstream commit in `ncnn.commit`. The same-frame AHB/CPU comparison is implemented in the app layer. Deeper Extractor or queue instrumentation would require a separate rebuild of that exact pinned NCNN source and is outside this diagnostic change.
