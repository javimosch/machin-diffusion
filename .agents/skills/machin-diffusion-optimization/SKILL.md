---
name: machin-diffusion-optimization
description: Optimization history (434s → 19.8s), patterns that worked, and the roadmap for further gains. Read this before planning optimizations.
---

# Optimization History and Roadmap

## Current state: 2.9s on RTX 4090 / 13.0s on RX 6600 (was 434s — 149x speedup)

| Stage | RTX 4090 (RunPod) | RX 6600 (Windows) |
|-------|-------------------|-------------------|
| CLIP | 313ms | 1316ms |
| UNet | 1228ms | 8060ms |
| VAE | 1416ms | 3613ms |
| **Total** | **2.96s** | **13.0s** |

Same binary, same OpenCL kernels — 4.4x faster on RTX 4090. Linux + Windows, AMD + NVIDIA.

## What worked (8 passes, all advised by Claude Fable 5.1)

| # | Optimization | Time saved | Where |
|---|-------------|-----------|-------|
| 1 | Batch CLIP linears (462→6 dispatches/layer) | 103s→9s | `clip.src` |
| 2 | Fused flash-attention-lite kernel | 109s→31s | `codegen.go` + `unet_blocks.src` |
| 3 | VAE attention via 2x matmul_f32 | 37s→30s | `vae.src` |
| 4 | Parallel group_norm (256-lane workgroups) | 30s→24s | `codegen.go` |
| 5 | Tiled matmul (16x16 local memory) | 61s→37s | `codegen.go` |
| 6 | Fused group_norm+SiLU | 30s→22s | `codegen.go` + `vae.src` + `unet_blocks.src` |
| 7 | Register-tiled 3x3 conv2d (4ch×4px per work-item) | 22s→22s | `codegen.go` |
| 8 | GPU broadcast add for time-embedding | 23s→20s | `codegen.go` + `unet_blocks.src` |

## Key patterns that worked

1. **Measure before changing** — add `now_ms()` timers around each stage before optimizing. The CLIP 103s was dispatch overhead nobody expected.
2. **Preserve validated output** — after every optimization, compare pixel diff against the previous validated output. Target: <0.05 mean, max≤1.
3. **Prefer high-return low-risk** — batching and fusion before rewriting kernels.
4. **Fuse CPU passes into GPU kernels** — the CPU `expf` loop (silu_f32) was 4-8s because expf doesn't auto-vectorize on Windows. Fusing into the GPU kernel eliminated both the compute AND the memory pass.
5. **Use existing builtins where possible** — VAE attention with head_dim=512 doesn't fit registers, so decompose into 2x `matmul_f32` instead of writing a new kernel.
6. **Register tile for memory-bound kernels** — naive conv/matmul were memory-bound (uncoalesced access). Register tiling improved FMA/VMEM ratio 16x.

## Roadmap (not yet done, from Fable 5.1)

### Done: Device-resident activation chaining (pass 9)

**What:** Single-buffer device-resident tracking (`mfl_dev_active`/`mfl_dev_host`/`mfl_dev_sz`). GPU wrappers try `mfl_dev_take(input)` before uploading; output stays on device via `mfl_dev_set()`. `mfl_dev_take` downloads the previous device buffer before releasing when the chain breaks. `ocl_sync()` builtin forces download before CPU reads.

**Result:** UNet 17.5s→12.0s (31%), VAE 7.9s→4.1s (48%), total GPU compute ~25.4s→~16.1s (37%).

**Critical correctness rules:**
- `clFinish` MUST be called after every kernel dispatch (before releasing any input/weight buffers). Without it, the AMD OpenCL driver crashes with ACCESS_VIOLATION in `amdocl64.dll` because buffers are released while the kernel is still running.
- `ocl_sync()` MUST be called before any CPU operation that reads GPU output (copy_buf, concat_chw_f32, upsample_nearest, peek_f32, CPU layer_norm loops).
- `ocl_sync()` MUST be called at the start of resnet blocks and attention blocks — the input buffer is used both as kernel input (consumed from device) and as residual (needs host data). Without the sync, the residual reads stale host data.
- `mfl_dev_take` MUST download the current device buffer before releasing it when the chain breaks. Without this, data is lost when a new GPU op needs a different input.
- CLIP linear functions sync after every `matmul_f32` (CLIP has CPU attention/layer_norm that needs host data — chaining benefit is minimal for CLIP's small matmuls).

**`clFinish` overhead:** The `clFinish` after every kernel prevents pipelining (kernels can't overlap). The benefit is still significant because downloads (PCIe transfers) are eliminated — only kernel execution time remains. Future optimization: use events for deferred buffer releases to enable pipelining.

### Done: Windows mmap (pass 10)

**What:** Implemented `CreateFileMappingA` + `MapViewOfFile` in `mfl_mmap_file` for Windows (was stubbed, returning 0/0). This enables lazy page-mapped file access instead of reading the entire safetensors file into memory.

**Result:** UNet 12.0s→8.0s (33%), VAE 4.1s→3.6s (12%). Eliminated ~5GB of memcpy (3.5GB UNet + 334MB VAE + 1.3GB CLIP). File I/O dropped from ~110s to ~0s (pages loaded on demand during computation).

**Caveat:** `#include <windows.h>` must appear before the mmap function in the generated C (the OpenCL section includes it later). Added a local include in the `#elif _WIN32` block.

### Done: axpy_f32 GPU dispatch threshold (pass 11)

**What:** Added `n >= 4096` threshold to `mfl_axpy_dispatch_f32`. Small vectors use CPU; large vectors use GPU.

**Result:** CLIP 112s→1.3s (86x). The CLIP attention calls `axpy_f32` in a tight loop with 85-float vectors (head_dim=1024/12). Each GPU dispatch had ~0.26ms overhead (buffer create, upload, kernel, clFinish, release). 432K calls × 0.26ms = 112s. With CPU fallback for small vectors, the loop runs at full CPU speed.

**Lesson:** GPU dispatch has fixed overhead (~0.1-0.3ms per call). For small operations (< 4096 floats = 16KB), CPU is always faster. Always add a size threshold when making a CPU builtin dispatch to GPU.

### Done: Linux OpenCL port (pass 12)

**What:** Ported the OpenCL backend from Windows-only (`#ifdef _WIN32`) to cross-platform (`#if defined(_WIN32) || defined(__linux__)`). On Linux, uses `dlopen`/`dlsym` instead of `LoadLibraryA`/`GetProcAddress`. Added `-ldl` linking when OpenCL is used on Linux. Requires ICD vendor file in `/etc/OpenCL/vendors/` (e.g. `nvidia.icd` containing path to `libnvidia-opencl.so.1`).

**Result:** Same binary and kernels run on NVIDIA RTX 4090 via RunPod. RTX 4090: 2.9s total (CLIP 313ms, UNet 1228ms, VAE 1416ms) — 4.4x faster than RX 6600's 13.0s. Image output numerically identical across both GPUs.

**Key changes:** `codegen.go` — platform abstraction macros (`MFL_OCL_LOAD`/`MFL_OCL_SYM`/`MFL_OCL_HANDLE`/`MFL_OCL_LIBNAME`), all `#ifdef _WIN32` guards in OpenCL section changed to `#if defined(_WIN32) || defined(__linux__)`. `build.go` — added `-ldl` for Linux when `mfl_ocl_init` is in the C source.

**Caveat:** On RunPod pods, the NVIDIA OpenCL ICD file may not be pre-installed. Create `/etc/OpenCL/vendors/nvidia.icd` with the path to `libnvidia-opencl.so.1` (found via `ldconfig -p | grep libnvidia-opencl`).

### High-value next steps
- **Event-based deferred releases** — replace `clFinish` after every kernel with OpenCL events. Buffers are released only after their kernel's event completes. Enables kernel pipelining (multiple kernels queued without blocking). Could save 2-4s on UNet.
- **Buffer pool** keyed by size — saves `clCreateBuffer`/`Release` churn on 134MB buffers every call.
- **fp16 weights** — after correctness infrastructure is stable. ~2x throughput.

### Alternative paths
- **TAESD fast VAE** — ~19s saved but different image quality (softer detail, color drift). Opt-in `--fast-vae`. Weights: `taesd_decoder.safetensors` from madebyollin/taesd (use SD1/2 version, not taesdxl). Feed raw UNet latents (do NOT divide by 0.18215), input through `tanh(x/3)*3`, output is [0,1] not [-1,1].
- **Windows CreateFileMapping mmap** — currently reads full safetensors into memory (Windows mmap is stubbed).
- **Multi-step Euler scheduler** — after one-step path is fully optimized.
- **CLIP embedding cache** — for repeated prompts.

## Advisor

**Claude Fable 5.1** is the optimization advisor. Invoke via:
```bash
devin --model claude-fable-5-1-high --permission-mode dangerous -p "<question>"
```
Fable provided all 8 optimization designs. The pattern: measure first, preserve output, prefer high-return low-risk optimizations, compare against reference after each change.

## What NOT to do

- **Don't do im2col + matmul for conv** — at 512² with 256 channels the im2col matrix is 2.4GB and round-trips through host. Use register-tiled direct kernel instead.
- **Don't use `-cl-fast-relaxed-math`** — changes exp precision in group_norm/attention, breaks validation. Use `-cl-mad-enable` only.
- **Don't claim "byte-identical"** after fused kernels — float reordering gives ~0.02 mean, max=1. Use measured tolerances.
- **Don't add CFG for SD-Turbo** — guidance_scale=0.0 is correct. CFG at 1 step produces high-contrast garbage.
