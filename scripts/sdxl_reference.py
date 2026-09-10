#!/usr/bin/env python3
"""SDXL-Lightning 4-step reference using direct model loading (no diffusers pipeline).
Saves intermediate outputs for stage-by-stage validation.

Usage: python3 sdxl_reference.py <model_dir> <prompt> <output_dir>
"""
import sys, os, numpy as np, torch

model_dir = sys.argv[1]
prompt = sys.argv[2]
out_dir = sys.argv[3] if len(sys.argv) > 3 else "/workspace/ref"
os.makedirs(out_dir, exist_ok=True)

from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from diffusers import AutoencoderKL, EulerDiscreteScheduler
from safetensors.torch import load_file

device = "cuda"
dtype = torch.float32

# 1. Tokenizer
tokenizer = CLIPTokenizer.from_pretrained(f"{model_dir}/tokenizer")
tokens = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
input_ids = tokens.input_ids  # [1, 77] — keep batch dim
print(f"Token IDs: {input_ids[0].tolist()}")
np.save(f"{out_dir}/input_ids.npy", input_ids[0].numpy())

# 2. Text encoders
print("Loading TE1...")
te1 = CLIPTextModel.from_pretrained(f"{model_dir}/text_encoder", torch_dtype=dtype, attn_implementation="eager").to(device)
print("Loading TE2...")
te2 = CLIPTextModelWithProjection.from_pretrained(f"{model_dir}/text_encoder_2", torch_dtype=dtype, attn_implementation="eager").to(device)

with torch.no_grad():
    te1_out = te1(input_ids.to(device), output_hidden_states=True)
    te1_hidden = te1_out.last_hidden_state  # [1, 77, 768]

    te2_out = te2(input_ids.to(device), output_hidden_states=True)
    te2_hidden = te2_out.last_hidden_state  # [1, 77, 1280]
    te2_pooled = te2_out.text_embeds  # [1, 1280] (after text_projection)

    context = torch.cat([te1_hidden, te2_hidden], dim=-1)  # [1, 77, 2048]
    pooled = te2_pooled  # [1, 1280]

    np.save(f"{out_dir}/te1_hidden.npy", te1_hidden.cpu().numpy())
    np.save(f"{out_dir}/te2_hidden.npy", te2_hidden.cpu().numpy())
    np.save(f"{out_dir}/te2_pooled.npy", te2_pooled.cpu().numpy())
    np.save(f"{out_dir}/context.npy", context.cpu().numpy())
    np.save(f"{out_dir}/pooled.npy", pooled.cpu().numpy())

    print(f"TE1 hidden: {te1_hidden.shape}, mean={te1_hidden.mean():.6f}, std={te1_hidden.std():.6f}")
    print(f"TE2 hidden: {te2_hidden.shape}, mean={te2_hidden.mean():.6f}, std={te2_hidden.std():.6f}")
    print(f"TE2 pooled: {te2_pooled.shape}, mean={te2_pooled.mean():.6f}, std={te2_pooled.std():.6f}")
    print(f"Context: {context.shape}, mean={context.mean():.6f}, std={context.std():.6f}")

# Free text encoders
del te1, te2
torch.cuda.empty_cache()

# 3. Scheduler - SDXL-Lightning 4-step uses EulerDiscrete with trailing spacing
print("Setting up scheduler...")
scheduler = EulerDiscreteScheduler.from_pretrained(f"{model_dir}/scheduler", timestep_spacing="trailing")
scheduler.set_timesteps(4, device=device)
print(f"Scheduler sigmas: {scheduler.sigmas.cpu().numpy()}")
print(f"Scheduler timesteps: {scheduler.timesteps.cpu().numpy()}")
np.save(f"{out_dir}/sigmas.npy", scheduler.sigmas.cpu().numpy())
np.save(f"{out_dir}/timesteps.npy", scheduler.timesteps.cpu().numpy())

# 4. UNet - load from safetensors
print("Loading UNet...")
unet_weights = load_file(f"{model_dir}/unet_4step_fp32.safetensors")

# Build UNet using diffusers UNet2DConditionModel
from diffusers import UNet2DConditionModel
import json

unet_config = json.load(open(f"{model_dir}/unet/config.json"))
unet = UNet2DConditionModel(**unet_config).to(device, dtype=dtype)
unet.load_state_dict(unet_weights, strict=False)
del unet_weights
print("UNet loaded")

# 5. Generate latents with fixed seed
generator = torch.Generator(device).manual_seed(42)
latents = torch.randn((1, 4, 128, 128), generator=generator, device=device, dtype=dtype)
latents = latents * scheduler.init_noise_sigma
np.save(f"{out_dir}/init_latent.npy", latents.cpu().numpy())
print(f"Init latent: {latents.shape}, mean={latents.mean():.6f}, std={latents.std():.6f}")

# 6. UNet denoising loop
add_time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], dtype=dtype, device=device)
np.save(f"{out_dir}/add_time_ids.npy", add_time_ids.cpu().numpy())

with torch.no_grad():
    for i, t in enumerate(scheduler.timesteps):
        latent_model_input = scheduler.scale_model_input(latents, t)
        if i == 0:
            np.save(f"{out_dir}/unet_input_step0.npy", latent_model_input.cpu().numpy())

        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=context,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids},
        ).sample

        np.save(f"{out_dir}/noise_pred_step{i}.npy", noise_pred.cpu().numpy())
        print(f"Step {i} (t={t}): noise_pred mean={noise_pred.mean():.6f}, std={noise_pred.std():.6f}")

        latents = scheduler.step(noise_pred, t, latents).prev_sample
        np.save(f"{out_dir}/latent_after_step{i}.npy", latents.cpu().numpy())

# 7. VAE decode
print("Loading VAE...")
vae = AutoencoderKL.from_pretrained(f"{model_dir}/vae", torch_dtype=dtype).to(device)

with torch.no_grad():
    latents_scaled = latents / vae.config.scaling_factor
    image = vae.decode(latents_scaled).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    np.save(f"{out_dir}/vae_output.npy", image.cpu().numpy())
    print(f"VAE output: {image.shape}, mean={image.mean():.6f}")
    print(f"VAE scaling_factor: {vae.config.scaling_factor}")

    from PIL import Image
    img = (image[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(img).save(f"{out_dir}/reference.png")
    print(f"Saved reference.png ({img.shape})")

print(f"\nAll intermediates saved to {out_dir}/")
