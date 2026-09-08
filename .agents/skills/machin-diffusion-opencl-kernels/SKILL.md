---
name: machin-diffusion-opencl-kernels
description: OpenCL GPU kernel design patterns, caveats, and how to add new builtins to the machin runtime. Read this before optimizing GPU kernels or adding new builtins.
---

# OpenCL Kernel Design and Caveats

All GPU kernels live in **`machin/codegen.go`** as string literals (`mfl_ocl_source`). The machin-diffusion repo uses these builtins but the kernel source is in the machin compiler.

## Kernel inventory

| Kernel | Pattern | When used |
|--------|---------|-----------|
| `conv2d` | Naive: 1 work-item per (output_channel, output_position) | Fallback for non-3x3, stride≠1, c_out%4≠0 |
| `conv3x3_t4x4` | Register-tiled: 4 output channels × 4 x-pixels per work-item | 3x3/stride1/pad1, c_out%4==0, w%4==0 |
| `matmul` | Naive: 1 work-item per (output, batch) | Small dims (<16) |
| `matmul_tiled` | 16x16 local memory tiles, bank-conflict-free padding (17 stride) | All dims >= 16 |
| `group_norm` | 256-lane workgroups, local memory tree reduction | All group_norm calls |
| `group_norm_silu` | Fused GroupNorm+SiLU (same as group_norm + SiLU in second pass) | When SiLU follows group_norm |
| `attention_f32` | Flash-attention-lite: 1 work-item per (head, query), online softmax | UNet cross/self-attention (head_dim=64) |
| `add_vec_spatial` | Broadcast add vector to spatial tensor | Time-embedding addition |

## Key design decisions (the "whys")

### Why flash-attention-lite for UNet but matmul decomposition for VAE?
- UNet attention: head_dim=64, fits in 128-float register array → fused kernel with online softmax
- VAE attention: head_dim=512, **doesn't fit in registers** → decompose into 2x `matmul_f32` (scores=Q@K^T, softmax, out=scores@V^T) + one cheap transpose of V

### Why fused group_norm+SiLU?
- `silu_f32` on CPU uses `expf` which **does not auto-vectorize** on Windows mingw/MSVC
- This was 4-8 seconds of pure CPU time in the VAE
- Fusing SiLU into the group_norm GPU kernel eliminates both the expf loop AND its memory pass

### Why register-tiled conv2d?
- Naive conv2d: every FMA needs 2 VMEM loads, weight index depends on `co=get_global_id(0)` which isn't provably uniform → uncoalesced → ~150 GFLOPS (1/64 of peak)
- Tiled: 6 VMEM loads → 48 FMAs (8 FMA/VMEM vs 0.5) → ~16x improvement in the memory bound

### Why tiled matmul?
- Naive matmul: uncoalesced `w[o*n_in+k]` with `o` varying per work-item
- Tiled: 16x16 local memory tiles with 17-stride padding (bank-conflict-free)

## Caveats

- **Every builtin does host↔device round trip per call** (clCreateBuffer + Write + Read + Release). This is ~8GB traffic for VAE. Future optimization: device-resident activations.
- **Bias buffer is always created** (OpenCL can't dereference null). If bias=0, zero-fill with blocking write.
- **Build with `-cl-mad-enable`** (NOT `-cl-fast-relaxed-math` — that changes `exp` precision in group_norm/attention and breaks validation).
- **Accumulation order changes** slightly with fused kernels. Expect pixel diff ~0.02 mean, max=1. Do NOT claim "byte-identical" — use measured tolerances.
- **Windows only**: OpenCL path is `#ifdef _WIN32`. CPU fallback for other platforms.
- **machin main branch is protected** — runtime changes need a PR. machin-diffusion repo is not protected.

## How to add a new builtin to machin

All steps in `machin/` (the compiler repo):

1. **Add OpenCL kernel source** to `mfl_ocl_source` string in `codegen.go`
2. **Add kernel handle**: `static cl_kernel mfl_ocl_k_<name> = NULL;`
3. **Add `p_clCreateKernel`** call in `mfl_ocl_init`
4. **Add to the null-check guard**
5. **Add GPU dispatch function** (`mfl_<name>_gpu`)
6. **Add CPU fallback + dispatch wrapper** (`mfl_<name>_f32` with `#ifdef _WIN32` GPU dispatch)
7. **Add codegen case** in `emitBuiltinExpr` (search for `case "matmul_f32"`)
8. **Add type checker entry** in `types.go` (`checkBuiltin` — search for `case "matmul_f32"`)
9. **Add guide entry** in `guide.go`
10. **Rebuild**: `cd ~/ai/machin && go build -o $(command -v machin) .`
11. **If touching selfhost**: run `python3 selfhost/gen-prelude.py` + `selfhost/verify-cgen.sh` + `verify-fixpoint.sh` (CI requires Go and self-hosted compilers to emit identical C)

## Validation after kernel changes

```bash
# Build, deploy, run
cd ~/ai/machin-diffusion
machin encode safetensors.src clip.src spatial.src unet_blocks.src unet.src vae.src image.src main.src > machin-diffusion.mfl
machin build machin-diffusion.mfl --target windows -o machin-diffusion.exe
rcc ordi-jla ~/ai/machin-diffusion/machin-diffusion.exe M:/machin-diffusion/machin-diffusion.exe 30
rcx ordi-jla "cmd /c M:\machin-diffusion\machin-diffusion.exe M:\machin-diffusion\models\sd-turbo M:\machin-diffusion\token_ids.txt M:\machin-diffusion\output.ppm" 600

# Check timing breakdown in log
# Verify output: pixel diff should be ~0.02 mean, max=1 vs previous validated output
```
