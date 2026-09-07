#!/bin/bash
# build.sh — build machin-diffusion
set -e

cd "$(dirname "$0")"

# Encode loose source → canonical .mfl
machin encode safetensors.src clip.src main.src > machin-diffusion.mfl

# Build → native binary
machin build machin-diffusion.mfl -o machin-diffusion

echo "Built machin-diffusion"
./machin-diffusion 2>&1 | head -5 || true
