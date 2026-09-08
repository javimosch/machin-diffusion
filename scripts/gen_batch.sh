#!/bin/bash
# Generate images with diverse SD-Turbo prompts
set -e

VOCAB=~/ai/machin-diffusion/models/sd-turbo/tokenizer/vocab.json
MERGES=~/ai/machin-diffusion/models/sd-turbo/tokenizer/merges.txt
REMOTE_DIR=M:/machin-diffusion

PROMPTS=(
  "a girl photo, close take, warm lighting"
  "planet earth seen from the moon, detailed, cinematic"
  "a spaceship landing on a launch pad, top view, cinematic"
  "a cyberpunk city street at night, neon lights, rain, cinematic"
  "a mountain landscape at golden hour, dramatic clouds, wide shot"
  "a portrait of an old fisherman, weathered face, dramatic lighting"
  "a futuristic sports car, side view, studio lighting, reflective surface"
  "a cozy cabin in a snowy forest at dusk, warm window light"
  "an astronaut floating in space, earth reflection in visor, cinematic"
  "a japanese garden with cherry blossoms, peaceful, soft morning light"
)

for i in "${!PROMPTS[@]}"; do
  IDX=$((i+1))
  PROMPT="${PROMPTS[$i]}"
  echo "=== Image $IDX/10: $PROMPT ==="

  # Tokenize
  python3 ~/ai/machin-diffusion/scripts/tokenize.py "$VOCAB" "$MERGES" "$PROMPT" /tmp/tokens_$IDX.txt

  # Deploy tokens
  rcc ordi-jla /tmp/tokens_$IDX.txt ${REMOTE_DIR}/tokens_$IDX.txt 30

  # Run generation
  rcx ordi-jla "cmd /c ${REMOTE_DIR}\machin-diffusion.exe ${REMOTE_DIR}\models\sd-turbo ${REMOTE_DIR}\tokens_$IDX.txt ${REMOTE_DIR}\gallery_$IDX.ppm 2>${REMOTE_DIR}\gerr_$IDX.txt >${REMOTE_DIR}\glog_$IDX.txt" 120

  echo "  done"
done

echo "=== All 10 images generated ==="
