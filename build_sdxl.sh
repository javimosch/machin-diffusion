#!/bin/bash
# build_sdxl.sh — build SDXL-Lightning machin-diffusion
set -e

cd "$(dirname "$0")"

# Encode loose source → canonical .mfl
# Includes clip.src for shared math functions (gelu, layer_norm, linear, etc.)
machin encode \
  safetensors.src \
  clip.src \
  clip_sdxl.src \
  spatial.src \
  unet_blocks.src \
  unet_sdxl.src \
  vae.src \
  image.src \
  main_sdxl.src \
  > machin-diffusion-sdxl.mfl

# Build → native binary
machin build machin-diffusion-sdxl.mfl -o machin-diffusion-sdxl

echo "Built machin-diffusion-sdxl"
