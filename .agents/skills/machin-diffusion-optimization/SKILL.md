---
name: machin-diffusion-optimization
description: Optimization history (434s → 19.8s), patterns that worked, and the roadmap for further gains. Read this before planning optimizations.
---

# Optimization History and Roadmap

## Current state: 19.8 seconds (was 434s, 22x speedup)

| Stage | Time |
|-------|------|
| CLIP | 1.6s |
| UNet | 12.2s |
| VAE | 6.0s |
| **Total** | **19.8s** |

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

### High-value next steps
- **Device-resident activations** (`gpu_alloc`, builtins accept device handles) — eliminates ~8GB PCIe traffic, ~2s win. Biggest remaining architectural change.
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
