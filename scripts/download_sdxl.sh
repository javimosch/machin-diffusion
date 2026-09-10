#!/bin/bash
# Download SDXL base + SDXL-Lightning 4-step UNet
set -e
mkdir -p /workspace/models/sdxl
cd /workspace/models/sdxl

echo "=== Downloading SDXL base model components ==="
# Text encoders, tokenizer, VAE from stabilityai/sdxl-base
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'stabilityai/stable-diffusion-xl-base-1.0',
    allow_patterns=[
        'text_encoder/*',
        'text_encoder_2/*',
        'tokenizer/*',
        'tokenizer_2/*',
        'vae/*',
    ],
    local_dir='/workspace/models/sdxl'
)
print('Base components downloaded')
"

echo "=== Downloading SDXL-Lightning 4-step UNet ==="
python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    'ByteDance/SDXL-Lightning',
    'sdxl_lightning_4step_unet.safetensors',
    local_dir='/workspace/models/sdxl'
)
print(f'Lightning UNet: {path}')
"

# Rename Lightning UNet to expected name
if [ -f /workspace/models/sdxl/sdxl_lightning_4step_unet.safetensors ]; then
    mv /workspace/models/sdxl/sdxl_lightning_4step_unet.safetensors /workspace/models/sdxl/unet_4step_fp32.safetensors
fi

echo "=== Download complete ==="
ls -la /workspace/models/sdxl/
echo "---"
du -sh /workspace/models/sdxl/
