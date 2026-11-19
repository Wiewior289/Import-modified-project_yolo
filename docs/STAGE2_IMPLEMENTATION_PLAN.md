# Stage 2 implementation plan: Mali-G72 convolution routing

## Branch and baseline

- Source branch: `yolo_diagnostic_compute`
- Work branch: `yolo_stage2_kernel_routing`
- Pinned upstream NCNN: `6a1bf000f363714839a36793addc8c879d3d899e`
- Scope: NCNN per-layer option correctness, convolution algorithm evidence, and controlled YOLO11s A/B tests.
- The source branch remains unchanged. No changes are made to the existing MatMul implementation or asynchronous queue policy in this stage.

## Milestones

### S2.0 - Isolated branch and reproducible baseline
- [x] Create a new branch from the exact source branch head.
- [x] Preserve the pinned NCNN SHA and existing patchset.
- [ ] Record baseline build/test status on the target device.

### S2.1 - Correct per-layer feature-mask normalization
- [x] Implement `get_masked_option()` normalization consistent with NCNN's global option sanitization.
- [x] When FP16 arithmetic is disabled, also disable FP16 uniform.
- [x] When Vulkan FP16 packed and storage are both disabled, disable FP16 arithmetic.
- [x] Preserve CPU-side rule: FP16 arithmetic is disabled when FP16 storage is disabled.
- [x] Include the new patch in the pinned build script, patchset SHA, and CMake verification.
- [ ] Acceptance pending: pristine pinned NCNN must accept the complete patch stack; option diagnostics must show normalized values.
- Implementation commit: `featmask_option_normalization.patch` plus builder/CMake patchset verification. Android build and on-device validation have not been run.

### S2.2 - Convolution route and pipeline diagnostics
- [ ] Log per-layer convolution parameters at pipeline creation: kernel, stride, dilation, input/output channels, input/output shape, FP16 flags, SGEMM/Winograd flags, cooperative-matrix decision.
- [ ] Log the selected shader family/type, specialization constants, subgroup size, and local workgroup size.
- [ ] Log the actual pipeline family selected during forward/dispatch, not merely the pipeline candidates created.
- [ ] Keep logging opt-in and avoid extra queue submits, fences, or per-layer waits.
- Acceptance: one frame can be mapped from NCNN layer name to selected algorithm and dispatch geometry without changing scheduling.

### S2.3 - Controlled per-layer A/B
- [ ] Baseline: current FP16 arithmetic + SGEMM configuration.
- [ ] FP32 arithmetic only on the measured stride-2 candidates (`conv_2`, `conv_7`, `conv_12`, `conv_22`), after verifying names and layer indices in the exact model param.
- [ ] Disable SGEMM only on the same candidates, independently from the FP16 test.
- [ ] Test combined masks only after the independent tests.
- Acceptance: repeated paired runs on identical frames, stable thermal/frequency conditions, output agreement, and GPU timestamps.

### S2.4 - Decision and optimization
- [ ] Compare per-layer GPU time, whole-inference GPU time, end-to-end frame latency, and detection agreement separately.
- [ ] Keep a change only if the gain repeats and output correctness is preserved.
- [ ] Do not replace NCNN's default route globally based on a four-layer sample.
- Acceptance: documented result table and a reproducible model/param configuration.

## Measurement rules

- GPU timestamp duration is not camera-to-detection latency.
- Queue wait and readback time are not per-layer shader execution time.
- Do not compare unmatched warm-up frames as a clean A/B.
- No additional command-buffer slots or queue-priority changes are part of this plan.
- No forced per-layer synchronization is allowed in the diagnostic path.

## Current implementation status

The branch was created. The first code change is the feature-mask option-normalization patch. It has not yet been built with Android NDK or validated on the Mali-G72 device. All performance conclusions remain provisional until S2.3.
