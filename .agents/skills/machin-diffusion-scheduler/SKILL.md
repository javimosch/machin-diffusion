---
name: machin-diffusion-scheduler
description: CRITICAL — SD-Turbo scheduler correctness. The #1 bug was using wrong timestep spacing. Read this before touching the scheduler or if outputs look like abstract noise.
---

# SD-Turbo Scheduler Correctness

## The #1 bug we hit

Using `timestep_spacing: "leading"` (the diffusers default for some schedulers) produced **abstract noise** instead of images. The UNet was told the input was nearly clean (timestep=1, sigma~0.04) while supplying high-noise latents, so it left the noise mostly intact.

**Fix:** SD-Turbo requires `timestep_spacing: "trailing"`.

## Correct SD-Turbo scheduler config

```
timestep_spacing: "trailing"
steps_offset: 1
num_train_timesteps: 1000
beta_schedule: "scaled_linear"
beta_start: 0.00085
beta_end: 0.012
prediction_type: "epsilon"
use_karras_sigmas: false
```

### One-step values
- Timestep: `999`
- Sigma: `~14.6146`
- `init_noise_sigma`: `~14.6146`

### Four-step values (for reference)
- Timesteps: `[999, 749, 499, 249]`
- Sigmas: `[14.6146, 4.08173, 1.61289, 0.693205, 0.0]`

## One-step Euler operation

```
latent = noise * 14.6146
unet_input = latent / sqrt(14.6146² + 1)
noise_pred = UNet(unet_input, timestep=999, text_embeddings)
latent = latent - 14.6146 * noise_pred
latent = latent / 0.18215          # VAE scaling factor
VAE.decode(latent)
```

## General multi-step epsilon prediction

```
latent = randn * init_noise_sigma

for each step:
    model_input = latent / sqrt(sigma² + 1)
    eps = UNet(model_input, timestep, embeddings)
    latent = latent + eps * (sigma_next - sigma)

latent = latent / 0.18215
VAE.decode(latent)
```

**The carried state is the UNSCALED latent**, not the model input and not `x0_hat`.

## Guidance

- `guidance_scale=0.0` is the **official recommended** setting for SD-Turbo.
- SD-Turbo's ADD (Adversarial Diffusion Distillation) incorporates guidance behavior.
- **Do NOT add CFG** for the one-step path.
- Test with `guidance_scale=7.5` produced high-contrast, poor results.
- Negative prompts have **no effect** at 1 step.

## How to verify the scheduler

If you're debugging noisy output, check:
1. `timestep_spacing` is `"trailing"` (not `"leading"`)
2. One-step timestep is `999` (not `1`)
3. Sigma is `~14.6146` (not `~0.04`)
4. `guidance_scale` is `0.0`
5. The latent is divided by `0.18215` before VAE decode

## Reference

The reference pipeline is at `/tmp/ref_pipeline.py` (diffusers/PyTorch with the correct trailing scheduler). Validate against it stage-by-stage.
