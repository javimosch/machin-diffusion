#!/usr/bin/env python3
"""Compare machin-diffusion SDXL intermediates against diffusers reference.
Usage: python3 sdxl_compare.py <ref_dir> <our_dir>
"""
import sys, os, numpy as np

ref_dir = sys.argv[1]
our_dir = sys.argv[2]

def load_npy(path):
    return np.load(path)

def load_bin(path, shape, dtype=np.float32):
    data = np.fromfile(path, dtype=dtype)
    return data.reshape(shape)

def compare(name, ref, ours):
    if ref.shape != ours.shape:
        print(f"  {name}: SHAPE MISMATCH ref={ref.shape} ours={ours.shape}")
        return False
    diff = np.abs(ref.flatten() - ours.flatten())
    corr = np.corrcoef(ref.flatten(), ours.flatten())[0,1]
    print(f"  {name}: shape={ref.shape} corr={corr:.6f} max_diff={diff.max():.6f} mean_diff={diff.mean():.6f}")
    if corr < 0.99 or diff.max() > 1.0:
        print(f"    *** MISMATCH ***")
        return False
    return True

print("=== SDXL intermediate comparison ===")
print(f"Reference: {ref_dir}")
print(f"Ours:      {our_dir}")
print()

all_ok = True

# TE1 hidden [77, 768]
if os.path.exists(f"{ref_dir}/te1_hidden.npy") and os.path.exists(f"{our_dir}/te1_hidden.bin"):
    ref = load_npy(f"{ref_dir}/te1_hidden.npy")[0]  # [77, 768]
    ours = load_bin(f"{our_dir}/te1_hidden.bin", (77, 768))
    all_ok &= compare("TE1 hidden", ref, ours)

# TE2 hidden [77, 1280]
if os.path.exists(f"{ref_dir}/te2_hidden.npy") and os.path.exists(f"{our_dir}/te2_hidden.bin"):
    ref = load_npy(f"{ref_dir}/te2_hidden.npy")[0]  # [77, 1280]
    ours = load_bin(f"{our_dir}/te2_hidden.bin", (77, 1280))
    all_ok &= compare("TE2 hidden", ref, ours)

# TE2 pooled (after text_projection) [1280]
if os.path.exists(f"{ref_dir}/pooled.npy") and os.path.exists(f"{our_dir}/pooled.bin"):
    ref = load_npy(f"{ref_dir}/pooled.npy")[0]  # [1280]
    ours = load_bin(f"{our_dir}/pooled.bin", (1280,))
    all_ok &= compare("TE2 pooled (projected)", ref, ours)

# Context [77, 2048]
if os.path.exists(f"{ref_dir}/context.npy") and os.path.exists(f"{our_dir}/context.bin"):
    ref = load_npy(f"{ref_dir}/context.npy")[0]  # [77, 2048]
    ours = load_bin(f"{our_dir}/context.bin", (77, 2048))
    all_ok &= compare("Context (concatenated)", ref, ours)

# Noise pred step 0 [4, 128, 128]
if os.path.exists(f"{ref_dir}/noise_pred_step0.npy") and os.path.exists(f"{our_dir}/noise_pred_step0.bin"):
    ref = load_npy(f"{ref_dir}/noise_pred_step0.npy")[0]  # [4, 128, 128]
    ours = load_bin(f"{our_dir}/noise_pred_step0.bin", (4, 128, 128))
    all_ok &= compare("UNet noise_pred step 0", ref, ours)

# Final latent [4, 128, 128]
if os.path.exists(f"{ref_dir}/latent_after_step3.npy") and os.path.exists(f"{our_dir}/final_latent.bin"):
    ref = load_npy(f"{ref_dir}/latent_after_step3.npy")[0]  # [4, 128, 128]
    ours = load_bin(f"{our_dir}/final_latent.bin", (4, 128, 128))
    all_ok &= compare("Final latent", ref, ours)

# VAE output [3, 1024, 1024]
if os.path.exists(f"{ref_dir}/vae_output.npy") and os.path.exists(f"{our_dir}/vae_output.bin"):
    ref = load_npy(f"{ref_dir}/vae_output.npy")[0]  # [3, 1024, 1024]
    ours = load_bin(f"{our_dir}/vae_output.bin", (3, 1024, 1024))
    all_ok &= compare("VAE output", ref, ours)

print()
if all_ok:
    print("=== ALL STAGES MATCH ===")
else:
    print("=== MISMATCHES FOUND — see above ===")
