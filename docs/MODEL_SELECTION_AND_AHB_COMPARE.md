# YOLO11 model selection and clean AHB comparison

## NCNN source baseline

The diagnostic branch uses the original upstream Tencent/ncnn source at this exact commit:

`6a1bf000f363714839a36793addc8c879d3d899e`

The pin is stored in the repository-root file `ncnn.commit`. This is the same SHA that was previously required by the AHB patch script. The patch is deliberately no longer applied.

The upstream source was checked directly at the pinned commit. Its original AHB YCbCr sampler uses `unnormalizedCoordinates = VK_TRUE`, and its original `convert_ycbcr.comp` supplies pixel-space coordinates. These upstream files are built unmodified. Do not copy local NCNN source edits or apply the removed normalized-sampling patch.

Build the pinned source for Android:

```bash
export ANDROID_NDK_HOME=/path/to/android-ndk
bash scripts/build_pinned_ncnn_android.sh
```

The script:
- clones or resets only the managed checkout under `.deps/ncnn-<SHA>`;
- checks out the exact SHA in detached-HEAD state;
- runs `git clean -fdx` and verifies an empty working tree;
- builds upstream NCNN with Vulkan enabled for `arm64-v8a`;
- removes the old generated NCNN package before installing the clean build;
- writes `ncnn-source-sha.txt` beside `ncnnConfig.cmake`.

The app CMake configuration refuses to use an NCNN package without this marker or with a different SHA. JNI logs `NCNN_SOURCE_SHA=...` when the native library loads. This catches accidentally linking a stale package.

After the NCNN builder succeeds:

```bash
rm -rf app/.cxx app/build
./gradlew clean assembleDebug --no-configuration-cache
```

The NCNN sources themselves are not patched by this branch. The diagnostic changes are in the Android application layer.

## AHB versus CPU-YUV comparison

Select this mode in the app's compute selector:

`DIAG: AHB vs CPU YUV + Vulkan (10 samples)`

The comparison is performed inside the existing synchronous `DetectFrame` JNI call. Kotlin obtains one Camera2 `Image`, passes its `HardwareBuffer` to JNI, and keeps the `Image` alive until JNI returns. Native code uses that same AHardwareBuffer sequentially:

1. Path A imports the AHB into Vulkan, runs the existing GPU YCbCr/RGB preprocessing, and executes the model with Vulkan.
2. Once Path A has completed its GPU work and extraction, Path B locks the YUV planes of the same AHB, converts YUV to RGB on CPU, and executes the same loaded NCNN `Net` with Vulkan.
3. Native code compares the two snapshots and returns Path A detections to the normal app pipeline.
4. The JNI call returns; Kotlin closes the HardwareBuffer wrapper and then the originating Camera2 Image.

Both inferences use the same loaded model, same Vulkan backend, target resolution, rotation, thresholds, and source frame. The CPU path refers only to YUV-to-RGB preprocessing, not CPU inference. The pre-cast snapshot reports its actual element size; the next stage converts both normalized Mats to FP16 with the same NCNN cast utility when FP16 storage is enabled, then compares those tensors separately.

For log volume control, A/B comparison runs on every 30th camera frame, up to 10 samples per model initialization. Other frames continue through the regular AHB path. Each sample logs compact statistics, not tensor dumps.

## Comparison log records

Filter Android logcat to the native tags:

```bash
adb logcat -s Yolo11_AHB_MaliG72:I Yolo11Ncnn_JNI:I
```

Records beginning with `AHBCMP_` include:

- `BEGIN`: frame ID, Camera2 timestamp, AHB size/stride/format/usage and rotation.
- `YUV`: CPU plane rowStride and pixelStride for Y, U and V.
- `TENSOR`: RGB before normalization, normalized model input, and `out0`; shape, type, min/max/mean/stddev and non-finite count for A and B.
- `DIFF`: element-wise MAE, RMSE, maximum absolute difference and cosine similarity.
- `RGB_SAMPLES`: eight fixed spatial positions with RGB values for both paths.
- `DET`: detection count, top candidate and up to five decoded boxes for each path.
- `END`: completion or the path that failed.

A comparison is useful only when `AHBCMP_BEGIN` and `AHBCMP_END status=OK` exist for the same sample. Compare the stages in order:
1. RGB difference identifies conversion, sampling, rotation or resize divergence.
2. INPUT difference isolates normalization, packing or precision.
3. OUT0 difference after matching input points to model execution or synchronization.
4. Matching OUT0 with different detections points to decoder/NMS behavior.

The AHB RGB snapshot is read back only in this diagnostic mode. Do not enable the old `debug.yolo11.tensor_tap` property during this first comparison; that adds separate GPU clone work and can perturb timing.

## Model and input-size selector

The selector on this branch supports YOLO11n and YOLO11s detection when the matching assets are present. Input sizes are square values from 320 to 480 in 32-pixel steps:

`320, 352, 384, 416, 448, 480`

Camera stream dimensions are selected independently by Camera2. The model input slider changes NCNN target size and the AHB preprocessing destination.

The asynchronous N/N+1 split remains a separate mode and retains its existing graph-cut validation. It is not used by the new AHB comparison mode.
