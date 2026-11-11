# YOLO11 model selection and backend diagnostics

## NCNN base and repository patchset

The project pins the Tencent/ncnn upstream base at:

`6a1bf000f363714839a36793addc8c879d3d899e`

The pin is stored in the repository-root file `ncnn.commit`. This branch adds a repository-owned Vulkan implementation for the NCNN `MatMul` layer:

- `patches/ncnn/matmul_vulkan.h`
- `patches/ncnn/matmul_vulkan.cpp`
- `patches/ncnn/matmul.comp`

The implementation is deliberately conservative. It uses a conventional storage-buffer compute shader, one invocation per output element, and does not require cooperative matrices or subgroup shuffle. It preserves the surrounding graph's FP16 or FP32 storage representation and uses FP32 accumulation. The layer currently supports pack1 inputs with equal FP16/FP32 element sizes, tensor ranks 1 through 4, `transB`, vector/matrix cases, and the batch broadcasting forms handled by the pinned CPU `MatMul` layer. Unsupported shape/type combinations return an error rather than silently calculating with invalid addressing.

The NCNN build script applies these three files after resetting the managed `.deps/ncnn-<SHA>` checkout to the exact upstream commit. NCNN's existing `ncnn_add_layer(MatMul)` CMake macro then discovers the Vulkan source and shader during configure. No upstream tracked file is edited by the builder.

The builder calculates a SHA-256 over the concatenated patch files in this order: header, C++ source, shader. It installs that value as `ncnn-patchset-sha.txt` beside `ncnnConfig.cmake`. App CMake recalculates the same value from the repository files and refuses an installed NCNN package whose source SHA or patchset SHA differs. JNI logs both identities:

- `NCNN_SOURCE_SHA`
- `NCNN_PATCHSET_SHA`

Build the pinned and patched NCNN library:

```bash
export ANDROID_NDK_HOME=/path/to/android-ndk
bash scripts/build_pinned_ncnn_android.sh
rm -rf app/.cxx app/build
./gradlew clean assembleDebug --no-configuration-cache
```

The Android build and Mali-G72 execution have not been verified by this branch automation. A successful GitHub source update is not a successful NDK build, and no performance or correctness result is claimed until the device run completes.

## MatMul Vulkan versus CPU diagnostic

Select this entry in the compute selector:

`DIAG: MatMul Vulkan vs CPU (same frame, 10 samples)`

The mode is intended to verify the new `MatMul_vulkan` backend against the already-loaded CPU reference Net. During initialization it:

1. Loads the normal Vulkan Net and a separate CPU-only reference Net.
2. Walks the loaded model graph and logs each `MatMul` layer as `MATMUL_VK_LAYER`.
3. Refuses to start diagnostic mode if a MatMul layer still reports `support_vulkan=0`.
4. Logs `MATMUL_VK_GRAPH_READY` only after every MatMul layer is assigned a Vulkan implementation.

At runtime the diagnostic reuses the existing synchronous `DetectFrame` path. Kotlin passes one Camera2 `HardwareBuffer` and its image timestamp to JNI; that buffer remains alive for the full native call. For each selected sample, native code executes three paths sequentially against the same source frame:

1. `AHB_VULKAN`: imports the AHardwareBuffer, runs the existing Vulkan YCbCr/RGB preprocessing, then runs the Vulkan model with the new MatMul backend.
2. `CPU_YUV_CPU`: locks the YUV planes of the same AHardwareBuffer, converts YUV to RGB on CPU, then runs the CPU-only reference Net.
3. `CPU_YUV_VULKAN`: repeats the same CPU YUV-to-RGB preprocessing, then runs the Vulkan Net. This path isolates inference backend differences from AHB preprocessing differences.

The normal app result remains the detections from `AHB_VULKAN`; CPU and CPU-YUV/Vulkan paths are diagnostic references. A failure of a reference path is logged and does not replace the live AHB result. Tensor comparison logs include RGB/input/out0 statistics and differences, plus decoded detection summaries and per-path timings.

For log volume control, samples run on every 30th camera frame, up to 10 samples per model initialization. Other frames continue through the regular AHB inference path. Because a sample runs three full passes, its total wall time is diagnostic overhead, not production FPS.

The new mode emits summary records prefixed with `MATMULCMP_`:

- `MATMULCMP_BEGIN`: sample, frame ID, image timestamp, source size, AHB format/stride/usage, rotation.
- `MATMULCMP_PATH_STATUS`: return status for AHB Vulkan, CPU inference, and Vulkan inference.
- `MATMULCMP_TIMING`: preprocessing, inference/extract, and postprocessing wall times for the three passes.
- `MATMULCMP_END`: `OK`, `PARTIAL`, or AHB failure.

Existing detailed comparison records (`AHBCMP_TENSOR`, `AHBCMP_DIFF`, `AHBCMP_DET`, and related records) are retained so the new diagnostic can be compared directly with previous AHB/CPU logs.

## Reading the comparison

A useful sample needs a matching `MATMULCMP_BEGIN` and `MATMULCMP_END status=OK` for the same frame and timestamp.

Compare these in order:

1. `CPU_YUV_CPU` versus `CPU_YUV_VULKAN`: same CPU-preprocessed input, CPU versus Vulkan inference. This is the key backend comparison for MatMul.
2. `OUT0_CPU_vs_GPU`: compares model output tensors before decode/NMS.
3. Detection count, classes, boxes and scores: checks downstream decode/NMS agreement.
4. `AHB_VULKAN` versus `CPU_YUV_VULKAN`: identifies input-preprocessing divergence, separate from backend differences.

Do not interpret `infer_extract_wall` or the three-pass sample wall time as raw shader execution time. Extract wall time includes queue synchronization, extraction and readback.

## Existing AHB comparison mode

The previous selector remains available:

`DIAG: AHB vs CPU YUV + Vulkan (10 samples)`

It uses the same three-path mechanism described above but does not require the MatMul-specific graph validation. Existing AHB cache, YCbCr and tensor diagnostic records are retained.

## Model and input-size selector

The selector supports YOLO11n and YOLO11s detection when matching assets are present. Input sizes are square values from 320 to 480 in 32-pixel steps:

`320, 352, 384, 416, 448, 480`

Camera stream dimensions are selected independently by Camera2. The asynchronous N/N+1 split remains a separate mode and retains its existing graph-cut validation.
