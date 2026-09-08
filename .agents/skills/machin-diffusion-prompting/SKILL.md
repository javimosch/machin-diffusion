---
name: machin-diffusion-prompting
description: SD-Turbo prompting guide. How to write prompts that work with 1-step distilled generation (no CFG, no negative prompts). Read this before generating images.
---

# SD-Turbo Prompting Guide

## What SD-Turbo is

SD-Turbo is a **distilled** version of Stable Diffusion 2.1, built for **single-step, real-time** text-to-image generation instead of the usual 20-50 step diffusion process. Stability AI labels it a "research artifact" rather than a production model — SDXL-Turbo is recommended for better prompt understanding and image quality.

Think of it as a **latency optimization**: the tradeoff is speed for fidelity and prompt adherence.

## Key differences from standard SD

| Aspect | SD-Turbo | Standard SD/SDXL |
|--------|----------|-------------------|
| Steps needed | 1 (single network eval) | 20–50 |
| Guidance scale | Not used — `guidance_scale=0.0` | 7–12 typical |
| Negative prompts | Not effective at 1 step | Fully supported |
| Resolution | 512×512 preferred | Up to 1024×1024 |
| Best for | Real-time previews, rapid prototyping, research | Final, detailed outputs |

## Prompting rules

### 1. Write dense, front-loaded prompts
Since there's no CFG steering, the model relies **purely on how well the prompt tokens map to its distilled understanding**. Put the most important descriptors first.

**Good:** `a girl photo, close take`  
**Good:** `planet earth seen from the moon, detailed, cinematic`  
**Bad:** `a photo of a girl who is standing close to the camera and looking at the viewer` (too verbose, key info buried)

### 2. Skip negative prompts entirely
They have **no effect** at one step. Don't waste tokens on them.

### 3. Use concrete visual descriptors
- Subject first, then style/mood
- `close take`, `top view`, `wide shot` for framing
- `cinematic`, `photo`, `detailed` for style
- Color/lighting descriptors work well: `warm lighting`, `golden hour`

### 4. Resolution
512×512 is the preferred resolution. Higher works but quality degrades.

## Best use cases

- **Live preview loops** — iterating on a prompt with instant visual feedback
- **Rapid prototyping** — quick concept exploration before a slower high-quality pass
- **Research** — real-time generative model experiments

## What SD-Turbo is NOT good at

- Fine detail (hair, text, texture) — softer than full SD
- Complex prompt adherence — distilled model loses some nuance
- Production-quality final outputs — use SDXL for that

## Future direction

Newer turbo variants (2025-2026) use Adversarial Diffusion Distillation (ADD) for 4-step generation with better quality retention. SD3.5 Large Turbo is an example. These may be better fits if you want speed without sacrificing as much prompt fidelity.

## Model

- HuggingFace: https://huggingface.co/stabilityai/sd-turbo
- Local path: `models/sd-turbo/`

## Generating an image

```bash
# 1. Tokenize
python3 scripts/tokenize.py models/sd-turbo/tokenizer/vocab.json models/sd-turbo/tokenizer/merges.txt "your prompt here" /tmp/tokens.txt

# 2. Deploy tokens
rcc ordi-jla /tmp/tokens.txt M:/machin-diffusion/tokens.txt 30

# 3. Run (uses seed 12345 by default)
rcx ordi-jla "cmd /c M:\machin-diffusion\machin-diffusion.exe M:\machin-diffusion\models\sd-turbo M:\machin-diffusion\tokens.txt M:\machin-diffusion\output.ppm" 600
```
