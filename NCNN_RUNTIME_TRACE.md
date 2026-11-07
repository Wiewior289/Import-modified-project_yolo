# YOLO11 / NCNN runtime investigation trace

This commit adds app-side evidence collection for Camera2 -> AHardwareBuffer -> Vulkan preprocessing -> NCNN inference. It does not change model weights, shader math, thresholds, output decoder, cache capacity or scheduling.

## Camera / frame correlation
- `CAMERA_IMAGE`: app frame id, ImageReader timestamp, dimensions and crop rectangle.
- `CAMERA_SENSOR`: Camera2 capture-result frame number, sensor timestamp, exposure, frame duration, ISO, AE FPS range and AE state.
- Native frame timing and tensor records carry the app frame id and Image.timestamp.

## GPU and imported AHB conversion
- `VULKAN_DEVICE`: GPU/driver identity, Vulkan API and driver versions, vendor/device IDs, heap-budget report.
- `AHB_YCBCR_PROPERTIES`: VkFormat/externalFormat, feature flags, suggested YCbCr model/range, chroma offsets, channel mapping, allocation size and memoryTypeBits.
- These values are driver/Vulkan sampler metadata, not evidence that the shader itself implements a BT.601/709 matrix.

## NCNN graph / operators
- `NCNN_GRAPH_LAYER`: operator index/type/name, bottom/top blob ids, blob names, producer/consumer indices, shape hints and capability flags.
- `CPU_ONLY_CANDIDATE` means the operator advertises `support_vulkan=0`.
- `VULKAN_CAPABLE_RUNTIME_UNVERIFIED` means it advertises Vulkan support. It is not proof that this execution actually used the Vulkan implementation.
- Topology/capability does not prove runtime blob residency, actual placement or transfer behavior.

## Tensor logs
- `TENSOR_VK` reports VkMat descriptors: dimensions, shape, element size/packing, data/allocator pointer, buffer offset and capacity. It does not read device memory.
- `TENSOR_CPU name=out0` summarizes the CPU Mat already returned from the extractor every 30 app frames.
- AHB preprocessing timings split host recording time for import, Scale and packing/dtype conversion. Dispatch count is collected before submission. Wait time is wall time around `submit_and_wait()`, not pure GPU time.
- `infer_extract_wall` remains wall time around canonical `Extractor::extract("out0", Mat)`; it includes the internal work and synchronization performed by the linked NCNN library.

## Optional explicit GPU tensor snapshots
Default disabled.

One-shot next AHB frame:
```sh
adb shell setprop debug.yolo11.tensor_tap 1
```

Disable/re-arm:
```sh
adb shell setprop debug.yolo11.tensor_tap 0
```

Periodic every 30 app frames:
```sh
adb shell setprop debug.yolo11.tensor_tap 2
```

The tap uses `VkCompute::record_clone(VkMat, Mat, Option)` to snapshot:
1. `ahb_rgb_0_255_snapshot` - imported/converted/rotated/resized input;
2. `ahb_scaled_0_1_snapshot` - after Scale;
3. `ncnn_input_snapshot` - after dtype/packing conversion before Extractor input.

It adds explicit GPU->CPU copies and another submit/wait; logs mark `PERTURBS_PIPELINE=1`. Do not use tap runs for FPS, latency, power or thermal benchmarking. This is a correctness/debug probe only.

## Logcat
```sh
adb logcat -c
adb logcat -v threadtime -s Yolo11_AHB_MaliG72:I Yolo11Ncnn_JNI:I MainActivity:I ncnn:V '*:S'
```

Use a rooted shell if Android blocks writing the debug system property from adb shell.

## What is still outside app visibility
The app links NCNN as an external package via `find_package(ncnn REQUIRED)`. NCNN graph executor, allocator internals and Vulkan command implementation are not compiled in this app target. This patch therefore cannot conclusively expose:
- actual runtime CPU/Vulkan placement per layer;
- implicit Mat/VkMat conversion, upload/download and residency changes;
- internal barriers, command-buffer boundaries, queue submits/waits, fences or overlapping work;
- exact dispatch route of the MatMul operators.

Those require instrumenting and rebuilding the **exact NCNN source revision** corresponding to the linked headers and library. Trace graph forward selection, blob residency transitions, CPU/GPU conversion paths, command recording, barriers and queue/fence submission. Keep NDK, ABI, Android API, Vulkan flags and headers/library versions identical. NCNN built with `NCNN_BENCHMARK=ON` is required for built-in per-layer timing; defining it only in the app is not enough.

The supplied AHB shader samples the YCbCr image and multiplies sampled RGB by 255; the YCbCr sampler conversion model/range and component mapping come from Vulkan/driver metadata. The app scales [0,255] to [0,1]. Direct AHB import avoids the ordinary CPU-mapped YUV->RGB->host Mat->GPU upload input path, but is not end-to-end zero-copy: the output tensor is still read back to CPU for decoding/NMS and telemetry.
