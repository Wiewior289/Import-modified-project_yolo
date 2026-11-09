# YOLO11 model and input-size selection

## Scope of this branch

The runtime selector is limited to:

- YOLO11n detection
- YOLO11s detection
- square model input from 320 to 480 pixels, in 32-pixel steps

The camera stream dimensions are not changed by the slider. It controls the NCNN model input and the AHB preprocessing destination.

The selector exposes YOLO11s only when both `yolo11s.param` and `yolo11s.bin` are present in APK assets. This avoids offering a model that cannot be loaded.

## YOLO11s asset

The upstream Nihui repository provides:

- `yolo11s.ncnn.param` (21,506 bytes)
- `yolo11s.ncnn.bin` (18,918,684 bytes)

This branch includes the parameter file renamed to the asset name expected by this project's native loader:

`app/src/main/assets/yolo11s.param`

The binary weight file is not committed by this change. Copy the upstream `yolo11s.ncnn.bin` to:

`app/src/main/assets/yolo11s.bin`

The loader deliberately expects `yolo11s.bin`, because `Yolo11::load()` builds asset paths as `yolo11%s.param` and `yolo11%s.bin`.

## Why the current N/N+1 cut is allowed for S

A structural comparison of the official Nihui YOLO11n and YOLO11s detection parameter files found:

- 257 layers and 307 blobs in both graphs
- identical layer order, types, names, bottom IDs and top IDs across all 257 layers
- the same two attention MatMul layers and the same blob IDs
- no graph-edge differences in the parameter files

The existing split remains:

- GPU prefix: layers 1..109
- CPU suffix: layers 110..256
- boundary blobs: 37, 72, 123, 125, 128, 129, 133, 135, 136, 137

The native loader now permits only model types n and s for split modes. It still executes the existing runtime checks for layer/blob counts, boundary producers, Vulkan support in the prefix, and cut completeness. A graph with merely matching layer counts is not enough to pass those checks.

YOLO11m is intentionally not exposed. Its detection graph has 327 layers and 387 blobs, so the current cut is not valid for it.

## Why classification is not a drop-in model choice

The repository's `yolo11n_cls.ncnn.param` is a distinct classification graph (121 layers, 143 blobs). Its tail includes global average pooling, flatten, a 1000-output InnerProduct and Softmax. The current native postprocessor is a detection decoder expecting DFL features and 80 class logits. The classification asset therefore cannot be selected through this detector UI.

## Resolution range

Valid values are:

`320, 352, 384, 416, 448, 480`

The resolution selector applies only after the user releases the slider. It does not repeatedly destroy/reinitialize NCNN while the thumb is moving.

Nihui's original UI combines model and size in one model index and exposes the presets 320, 480 and 640. This branch separates model choice from input size and deliberately stops at 480. The additional intermediate values (352, 384, 416 and 448) are this branch's extension, based on the detector's stride-32 grid and the dynamic-shape graph. They still require device-side numerical and performance validation.

The native loader validates the same range and 32-pixel step. NCNN grid/stride generation and the AHB import/preprocess target use the configured dimensions.

## Reinitialization and async safety

Changing model, resolution or compute mode reinitializes the native engine. When the previous mode is N/N+1, camera callbacks are detached and the GPU and CPU executor queues are drained before the engine is destroyed.

A pending-drain flag preserves this ordering if the user changes configuration again while the first asynchronous drain is still in progress.

## AHB normalized YCbCr patch: applying it to the library actually linked by the APK

The repository already contains:

- `patches/ncnn/0001-ahb-normalized-ycbcr-sampling.patch`
- `scripts/apply_ncnn_ahb_normalized_sampling.sh`

The patch changes the Vulkan external-format sampler setup and YCbCr coordinate handling. It does not patch a prebuilt NCNN package automatically. CMake links the package found via `ncnn_DIR`, so patching an unrelated NCNN source checkout does not change the APK.

Build and install the patched library into the exact package path used by this app:

```bash
export ANDROID_NDK_HOME=/path/to/android-ndk
./scripts/build_patched_ncnn_android.sh /path/to/ncnn
```

The script requires the pinned NCNN HEAD:

`6a1bf000f363714839a36793addc8c879d3d899e`

It applies the AHB patch, configures an arm64 Android Vulkan build with `NCNN_BENCHMARK=OFF`, builds it and installs the SDK package under:

`app/src/main/cpp/ncnn-android-vulkan/arm64-v8a`

Then rebuild the app without stale native build output:

```bash
rm -rf app/.cxx app/build
./gradlew clean assembleDebug --no-configuration-cache
```

## Runtime validation after rebuilding

For the AHB path, verify that runtime diagnostics no longer report:

- `unnormalized_coordinates=1`
- the invalid external-format sampler pNext chain

Compare the same camera frame against CPU YUV conversion and inspect RGB alignment at the four rotations (1, 3, 6, 8), including resize and no-resize cases. A successful APK build alone does not prove that the patched NCNN package was linked or that YCbCr sampling is numerically correct.
