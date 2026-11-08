# YOLO11 asynchronous frame-overlap implementation

## Scope

This branch implements **frame-level overlap**, not asynchronous layer scheduling:

- frame N: CPU suffix (boundary `Mat` tensors -> `out0` -> DFL/decode/NMS)
- frame N+1: GPU prefix (Camera2 AHB import -> preprocess -> NCNN layers 0..109)

The current synchronous AHB inference path remains available as a baseline until the split path is validated on-device. No `.param` or `.bin` file is physically split.

## Verified model cut

The checked-in `app/src/main/assets/yolo11n.param` declares 257 layers and 307 blobs. Reproducing the blob-index assignment performed by NCNN `Net::load_param()` gives ten blobs crossing the cut after layer 109:

| blob ID | Producer | First consumer in CPU suffix |
|---:|---|---|
| 37 | layer 31 `splitncnn_2` | layer 145 `cat_10` |
| 72 | layer 60 `splitncnn_6` | layer 130 `cat_8` |
| 123 | layer 102 `split_4` | layer 125 `cat_7` |
| 125 | layer 103 `splitncnn_13` | layer 119 `add_8` |
| 128 | layer 105 `splitncnn_14` | layer 114 `Tensor.view_6` |
| 129 | layer 105 `splitncnn_14` | layer 115 `Tensor.reshape_4` |
| 133 | layer 107 `split_5` | layer 110 `matmul_167` |
| 135 | layer 108 `splitncnn_15` | layer 115 `Tensor.reshape_4` |
| 136 | layer 108 `splitncnn_15` | layer 113 `matmultransb_0` |
| 137 | layer 109 `transpose_169` | layer 110 `matmul_167` |

Layers 110 onward belong to the CPU suffix, including Vulkan-capable operators in that range. The CPU `Net` must be loaded with Vulkan disabled; mutating `Layer::support_vulkan` after model load is not the design.

Run the topology guard from the repository root:

```sh
python3 tools/verify_async_graph_cut.py app/src/main/assets/yolo11n.param
```

The guard intentionally fails if the model shape, producer/consumer topology, or expected cut changes.

## NCNN execution contract

The pinned NCNN source inspected for this branch is `6a1bf000f363714839a36793addc8c879d3d899e`.

- `Extractor::input(int, const VkMat&)`, `Extractor::extract(int, VkMat&, VkCompute&)`, `Extractor::input(int, const Mat&)`, and `Extractor::extract(int, Mat&)` are available.
- `Extractor::set_light_mode(false)` is required on the GPU-prefix extractor so intermediates needed at the ten cut edges are not recycled before download.
- `VkCompute::record_download()` records each device-to-host transfer into one command context. `submit_and_wait()` waits for that context's fence and performs deferred host copies/casts before returning.
- `VulkanDevice::acquire_queue()` / `reclaim_queue()` protect the compute-queue pool with a mutex and condition variable. `VkCompute::submit_and_wait()` returns the queue to the pool after `vkQueueSubmit()` and waits on its own fence afterwards. Independent `VkCompute` objects therefore do not share command buffers or fences. This permits other host work, including a CPU suffix, while another frame's GPU submit call is waiting; it does **not** make a single `VkCompute` call itself asynchronous.
- NCNN allocator choices must remain frame-safe. Host boundary `Mat`s must outlive the GPU-prefix context and remain valid until the CPU suffix has consumed them. Vulkan allocators must not be reclaimed before the GPU fence has completed and all frame `VkMat` references have been released.

## Frame and resource ownership

Each in-flight frame owns its boundary tensors, output buffer, identifiers, and timing metadata. At most two frame slots are admitted initially. The camera `Image` and Java `HardwareBuffer` wrapper remain alive until the GPU-prefix fence has completed; only then may the camera image be closed. The CPU suffix must not read the AHB.

A GPU-prefix worker serializes AHB cache operations. A single CPU-suffix worker serializes CPU `Extractor` execution and the current post-processing scratchpads (`proposal_scratchpad`, `candidates_scratchpad`, `last_top_score`, `last_top_class`). This avoids pretending those existing shared scratchpads are thread-safe. The two workers may overlap with each other because they operate on separate model instances and frame-owned tensors.

Model replacement, cache destruction, and Vulkan-device teardown must occur only after both frame workers have drained. A short-lived engine/lifecycle lock may protect the engine handle, but must not be held across inference.

## Validation gates

1. The graph-cut guard passes against the exact model asset.
2. The GPU prefix returns all ten non-empty boundary tensors, with dimensions and element packing recorded for each frame.
3. A single frame through GPU-prefix + CPU-suffix produces the same detection count/classes/boxes as the unmodified full-GPU path. Compare with a documented numeric tolerance; do not silently tune FP16 settings in this branch.
4. A stress run with frame IDs and timestamps proves that frame N CPU suffix overlaps frame N+1 GPU prefix without cross-frame tensor/result mixing.
5. Verify AHB lifetime: the `Image` closes only after the prefix fence, and no camera buffer is referenced by CPU suffix.
6. Compare throughput and per-frame latency separately. A faster pipeline throughput does not imply lower single-frame latency.

AHB YCbCr sampler/UV correctness is a separate work item and must not be mixed into the frame-overlap patch.
