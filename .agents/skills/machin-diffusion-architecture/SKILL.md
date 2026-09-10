---
name: machin-diffusion-architecture
description: Architecture and pipeline overview for machin-diffusion (SD-Turbo in pure MFL). Read this first to understand the codebase before making changes.
---

# machin-diffusion Architecture

## What this is

Stable Diffusion Turbo inference in **pure machin (MFL)** — no Python, no PyTorch, no libtorch. One static binary. GPU-accelerated via OpenCL on AMD RX 6600. Runs in **19.8 seconds** for a 512×512 image (was 7 minutes before optimization).

## Model

- **SD-Turbo** (distilled from SD 2.1 for single-step generation)
- HuggingFace: https://huggingface.co/stabilityai/sd-turbo
- Stored at: `models/sd-turbo/` (safetensors format)

## Pipeline

```
text → CLIP text encoder → text_embeddings
noise → UNet(text_embeddings, timestep=999) → noise_pred
latent = noise * sigma - sigma * noise_pred
latent /= 0.18215
latent → VAE decoder → image [3, 512, 512]
```

### CLIP ViT-H/14 text encoder
- Hidden size: 1024, 16 heads, head_dim: 64
- 23 transformer layers, intermediate size: 4096
- Vocabulary: 49408, max sequence length: 77
- Causal attention (token 0 = BOS/SOS, token 1 = prompt start, last non-pad = EOS)
- Output: `text_embeddings` [77, 1024] — take rows 0..last_non_pad (inclusive)

### UNet
- Conv/residual/transformer blocks with time embeddings
- Cross-attention to text embeddings (multi-head, head_dim=64)
- Self-attention in transformer blocks
- ResNet blocks: norm→SiLU→conv→(add time_emb)→norm→SiLU→conv→residual

### VAE decoder
- Latent [4, 64, 64] → image [3, 512, 512]
- `post_quant_conv` → `conv_in` → `mid_block` (resnet + attention + resnet) → 4 `up_blocks` → `conv_norm_out` → `conv_out`
- Up blocks: 3 resnets each, first 3 have upsamplers (nearest 2x + conv)
- Single-head attention in mid_block (head_dim=512, seq=4096)

## File layout

| File | Role |
|------|------|
| `main.src` | Entry point, timing instrumentation, scheduler, one-step Euler |
| `clip.src` | CLIP text encoder (batched linears) |
| `unet.src` | UNet forward pass |
| `unet_blocks.src` | ResNet blocks, transformer blocks, spatial attention |
| `vae.src` | VAE decoder |
| `spatial.src` | Helper ops (upsample, add_buf, group_norm, silu_buf) |
| `image.src` | PPM image output |
| `safetensors.src` | Safetensors loader (st_tensor, st_header) |
| `build.sh` | Build script |
| `scripts/tokenize.py` | CLI tokenizer (vocab + merges → token IDs) |
| `scripts/extract_header.py` | Extract safetensors header |
| `scripts/extract_vocab.py` | Extract CLIP vocab/merges from tokenizer |

## Machin builtins used

| Builtin | Role |
|---------|------|
| `matmul_f32` | Batched matmul (QKV, FFN, projections). GPU: tiled 16x16 local memory. |
| `conv2d_f32` | 2D convolution. GPU: register-tiled 3x3 kernel (4ch×4px per work-item). |
| `group_norm_f32` | GroupNorm. GPU: 256-lane workgroups with local memory reduction. |
| `group_norm_silu_f32` | Fused GroupNorm+SiLU. Eliminates CPU expf loop. |
| `attention_f32` | Fused flash-attention-lite (online softmax). For UNet cross/self-attention. |
| `add_vec_spatial_f32` | Broadcast add vector to spatial tensor. For time-embedding addition. |
| `silu_f32` | SiLU in-place (CPU, vectorized). |
| `dot_f32` | Float dot product (CPU). |
| `axpy_f32` | AXPY (CPU, vectorized). For residual additions. |
| `now_ms` | Timing. |

See `machin-diffusion-opencl-kernels` skill for kernel internals and caveats.

## Build and deploy

```bash
# Build (local, targets Windows)
cd ~/ai/machin-diffusion
machin encode safetensors.src clip.src spatial.src unet_blocks.src unet.src vae.src image.src main.src > machin-diffusion.mfl
machin build machin-diffusion.mfl --target windows -o machin-diffusion.exe

# Deploy to Windows GPU machine
rcc ordi-jla ~/ai/machin-diffusion/machin-diffusion.exe M:/machin-diffusion/machin-diffusion.exe 30

# Run
rcx ordi-jla "cmd /c M:\machin-diffusion\machin-diffusion.exe M:\machin-diffusion\models\sd-turbo M:\machin-diffusion\token_ids.txt M:\machin-diffusion\output.ppm" 600

# Tokenize a prompt
python3 scripts/tokenize.py models/sd-turbo/tokenizer/vocab.json models/sd-turbo/tokenizer/merges.txt "prompt text" /tmp/tokens.txt
```

## Hardware

- GPU: AMD RX 6600 (OpenCL)
- CPU baseline: Intel i7-2700K
- OpenCL path is `#ifdef _WIN32` — Windows only. CPU fallback for other platforms.

## SDXL-Lightning architecture (separate pipeline)

A separate SDXL-Lightning 4-step pipeline exists (`clip_sdxl.src`, `unet_sdxl.src`,
`main_sdxl.src`, `build_sdxl.sh`). Key differences from SD-Turbo:

### Dual text encoders
- **TE1**: CLIP ViT-L/14 — hidden 768, 12 heads, 12 layers, intermediate 3072, quick GELU
- **TE2**: OpenCLIP ViT-bigG — hidden 1280, 20 heads, 32 layers, intermediate 5120, GELU tanh
- Context output: `[77, 2048]` = concat(TE1 `[77, 768]`, TE2 `[77, 1280]`)
- Pooled output: `[1280]` from TE2 — hidden state at EOS position after `final_layer_norm`,
  then projected through `text_projection.weight` (1280×1280)
- **EOS position**: found by scanning token IDs for the highest ID (49407), NOT `seq-1`
- **CAUSAL attention required**: both TE1 and TE2 use causal attention (`attention_causal_f32`).
  Using bidirectional `attention_f32` produces 0.18 correlation → abstract noise output.

### SDXL UNet
- 2.6B params, [1,2,10] transformer blocks (1 down, 2 mid, 10 up)
- ADM/additional conditioning: `add_time_ids` [1024,1024,0,0,1024,1024] sinusoidally embedded,
  concatenated with 1280d pooled text → 2816d → projected to 1280d
- 4-step denoising with EulerDiscreteScheduler (trailing spacing)
- Sigmas: [14.615, 4.082, 1.613, 0.693, 0.0], timesteps: [999, 749, 499, 249]

### SDXL VAE
- Latent [4, 128, 128] → image [3, 1024, 1024]
- Scaling factor: 0.13025 (latents divided by this before decode)

### `attention_causal_f32` builtin

Added to the machin compiler for SDXL CLIP. Same signature as `attention_f32` but
position i only attends to positions j ≤ i (causal mask). Has both OpenCL GPU kernel
and CPU fallback. The SD-Turbo `clip.src` already had a scalar causal attention loop;
the SDXL port needed the fused GPU version.
