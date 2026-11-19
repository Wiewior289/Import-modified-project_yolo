# MatMul runtime diagnostics

## Purpose

This branch adds runtime placement evidence for the two YOLO11n MatMul layers and restores NCNN Vulkan timestamp instrumentation without reading timestamp queries during intermediate command-buffer flushes.

The instrumentation is diagnostic. It is not a claim that the Mali-G72 driver issue is fully resolved until the APK is built and tested on the device.

## Build

The pinned NCNN source remains at the SHA stored in `ncnn.commit`.

```bash
export ANDROID_NDK_HOME="$HOME/Android/Sdk/ndk/<your-ndk-version>"
./scripts/build_pinned_ncnn_android.sh
./gradlew clean assembleDebug --no-configuration-cache
```

The pinned builder applies the MatMul backend files and the NCNN runtime patches to a pristine checkout, then regenerates NCNN. The patchset SHA includes every applied patch and backend file, and app CMake verifies that hash before linking. Stage 2 adds `patches/ncnn/featmask_option_normalization.patch`, which keeps per-layer FP16 option masking consistent with NCNN's global option sanitization.

Build flags:
- `NCNN_VULKAN=ON`
- `NCNN_BENCHMARK=ON` enables layer timing instrumentation in the NCNN library.
- `NCNN_BUILD_BENCHMARK=OFF` deliberately leaves out the separate command-line benchmark program. It does not disable runtime timestamps.

The builder deletes `app/.cxx` and `app/build` to avoid stale CMake/Gradle outputs.

## GUI compute modes

| GUI label | Native mode | Meaning |
|---|---:|---|
| CPU (pełny baseline) | 0 | Full CPU inference baseline |
| GPU Vulkan + AHB / MatMul CPU (kontrola) | 7 | Vulkan + direct AHB preprocessing; the two MatMul wrappers are switched to CPU |
| GPU_MATMUL Vulkan + AHB | 8 | Strict MatMul Vulkan mode; initialization fails if a MatMul layer has no Vulkan implementation |
| DIAG: porównanie MatMul Vulkan vs CPU | 6 | Existing three-path comparison; it is deliberately expensive and sample-limited |

Modes 3, 4 and 5 remain in native code but have been removed from the main GUI selector. Mode 8 is the initial selection.

## Evidence markers

### During initialization

- `MATMUL_CONFIG ... requested=VULKAN ... support_before=1 support_after=1`: the Layer_final wrapper exposes a Vulkan implementation after model loading.
- `MATMUL_CONFIG ... requested=CPU ... support_after=0`: the CPU-control mode destroyed the custom Vulkan pipeline and disabled the wrapper's Vulkan route.
- `MATMUL_GRAPH_READY ... runtime_route=WAITING_FOR_FRAME`: initialization checks passed; it is explicitly not a runtime execution claim.

### During an inference pass

- `[MATMUL_ROUTE] ... route=VULKAN_FORWARD_VKMAT`: NCNN selected the Vulkan branch for that MatMul and records whether the inputs are present as VkMat or Mat.
- `[MATMUL_VK_FORWARD]`: execution entered the custom `MatMul_vulkan::forward(VkMat)`; dimensions and transB are logged.
- `[MATMUL_VK_DISPATCH_RECORDED]`: the MatMul pipeline was added to the Vulkan command buffer. This alone does not mean the queue has completed it.
- `[MATMUL_ROUTE] ... route=CPU_FORWARD_MAT`: NCNN selected the CPU implementation.
- `[MATMUL_ROUTE_DONE] ... ret=0`: the selected layer forward returned successfully.
- `[NCNN_BENCHMARK_GPU] ... duration=...us`: GPU timestamp duration for the MatMul layer.
- CPU MatMul timing lines come from NCNN's CPU benchmark helper.

For mode 8, the expected sequence is a Vulkan route, `MATMUL_VK_FORWARD`, `MATMUL_VK_DISPATCH_RECORDED`, a successful route completion, and a MatMul GPU timestamp. If the Vulkan forward marker is absent, inspect the route marker and its input storage flags first.

For mode 7, the expected sequence is `CPU_FORWARD_MAT`, followed by a CPU MatMul timing. This is the control case that isolates the MatMul placement while retaining the surrounding Vulkan graph.

## Timestamp safety change

The old NCNN code attempted to retrieve timestamps after every partial `submit_and_wait()`. The same query pool was reset as command buffers were recycled, so those reads covered an incomplete query set and repeatedly called the Mali driver's `vkGetQueryPoolResults()` during inference.

The patched flow:
1. does not read query results during partial flushes;
2. resets the query pool once per VkCompute recording sequence, not on every command-buffer reset;
3. checks query-pool creation and the final queue submit result;
4. validates the pool handle, query range and destination size;
5. retrieves timestamp plus availability for each query, without `WAIT_BIT` and without a busy-poll loop;
6. reads the result once after the final submit-and-wait.

The dispatch-threshold `submit_and_wait()` remains. It is NCNN's intentional queue flush to avoid excessive pending work / driver timeouts, not polling.

A residual Mali driver fault at the final `vkGetQueryPoolResults()` call is still possible. The change removes the repeated partial-flush read pattern seen immediately before the uploaded SIGSEGVs, but only an on-device run can confirm whether the driver fault is gone.

## What this test can and cannot prove

- `support_vulkan=1` means the Layer_final wrapper retained a Vulkan implementation after initialization. It is not a trace of a particular frame.
- `MATMUL_ROUTE` plus `MATMUL_VK_FORWARD` proves that NCNN entered the Vulkan MatMul forward.
- `MATMUL_VK_DISPATCH_RECORDED` proves the shader dispatch was recorded, not that GPU execution completed.
- A successful final submit and valid timestamp pair provide the remaining evidence that the command buffer completed and the GPU timestamp range was available.
- GPU timestamps measure the recorded layer interval. They are not end-to-end camera-to-detection latency.
- Mode 6 performs multiple full inference passes per diagnostic sample. Its throughput is not normal live FPS.

Keep the first validation short. Compare mode 7 and mode 8 at the same model resolution, collect the MatMul route and timestamp lines, then check for native crashes and output agreement before drawing performance conclusions.


## Stage 2: per-layer option normalization

The new `featmask_option_normalization.patch` changes only `get_masked_option()` in the pinned NCNN `src/net.cpp`.

- Vulkan path: if both FP16 packed and FP16 storage are disabled, FP16 arithmetic is disabled too.
- Any path: if FP16 arithmetic is disabled, FP16 uniform is disabled.
- CPU path: if FP16 storage is disabled, FP16 arithmetic is disabled.

The change does not alter the global FP16 defaults, MatMul kernels, queue submission policy, or the model's layer masks. It only makes per-layer options obey the same dependencies as the global configuration.

The builder applies this patch after the existing async/runtime patches and includes it in `NCNN_PATCHSET_SHA`. App CMake includes the same file in its patchset hash calculation, so an APK cannot silently link an NCNN library built without the new patch.

This patch is implemented but is not yet validated by an Android NDK build or a device run. See [Stage 2 implementation plan](STAGE2_IMPLEMENTATION_PLAN.md).
