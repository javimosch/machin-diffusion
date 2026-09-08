#!/bin/bash
# build.sh — build machin-diffusion
set -e

cd "$(dirname "$0")"

# Encode loose source → canonical .mfl
machin encode safetensors.src clip.src spatial.src unet_blocks.src unet.src vae.src image.src main.src > machin-diffusion.mfl

# Build → native binary
machin build machin-diffusion.mfl -o machin-diffusion

echo "Built machin-diffusion"
