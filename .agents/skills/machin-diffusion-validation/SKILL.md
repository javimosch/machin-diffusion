---
name: machin-diffusion-validation
description: How to validate output correctness after optimizations. Reference comparison, pixel diff methodology, and image quality metrics.
---

# Validation Methodology

## Why validate after every optimization

Fused GPU kernels reorder floating-point operations, giving slightly different results. The goal is to confirm the optimization didn't break correctness — not to achieve bit-exact reproduction.

## Reference pipeline

The reference is at `/tmp/ref_pipeline.py` (diffusers/PyTorch with the **correct trailing scheduler**). It validates stage-by-stage:

| Stage | Metric | Typical value |
|-------|--------|--------------|
| CLIP embeddings | max diff | ~0.000018 |
| UNet noise prediction | max diff | ~0.000307 |
| UNet noise prediction | correlation | 1.000000 |
| Final latent | max diff | ~0.000070 |
| Final latent | correlation | 1.000000 |
| Image | mean pixel diff | 0.01–0.05 |
| Image | max pixel diff | 1–2 |

After fused kernels (group_norm_silu, attention_f32, conv3x3_t4x4, matmul_tiled):
- **pixel diff mean=0.02, max=1** — within tolerance, from float reordering in online softmax and tiled accumulation.

## Standard validation prompt

```
"a girl photo, close take"
seed: 12345
guidance_scale: 0.0
512x512
trailing scheduler
```

Expected output: photo-like image, ~75% skin-tone pixels, edge strength ~5.0, R mean ~151.

## How to compare images

```python
import numpy as np
from PIL import Image

prev = np.array(Image.open('previous.png'), dtype=np.int16)
curr = np.array(Image.open('current.png'), dtype=np.int16)
diff = np.abs(curr - prev)
print(f'pixel diff mean={diff.mean():.2f} max={diff.max()}')
# Acceptable: mean < 0.05, max <= 1
```

## Image quality metrics

```python
arr = np.array(img)
print(f'R: mean={arr[:,:,0].mean():.0f} std={arr[:,:,0].std():.0f}')
gray = arr.mean(axis=2).astype(np.float32)
gx = np.abs(np.diff(gray, axis=1))
print(f'edge strength: {gx.mean():.1f}')
skin = ((arr[:,:,0] > arr[:,:,1]) & (arr[:,:,1] > arr[:,:,2]) &
       (arr[:,:,0] > 80) & (arr[:,:,0] < 220)).sum()
print(f'skin-tone: {100*skin/(w*h):.1f}%')
```

## Fetching output from Windows

```bash
# PPM → base64 → decode locally
rcx ordi-jla "certutil -encode output.ppm out.b64 && type out.b64" 60 > /tmp/raw
# Then: decode base64, parse PPM header (P6\n<w> <h>\n255\n), load with PIL
```

## What "correct" looks like

- **Abstract noise** → scheduler is wrong (check `timestep_spacing: "trailing"`)
- **High-contrast garbage** → CFG was added (check `guidance_scale: 0.0`)
- **Photo-like, warm, smooth** → correct
- **pixel diff > 0.05 or max > 2** → optimization broke something, investigate

## Don't claim byte-identical

After fused kernels, output is numerically equivalent but not bit-identical. Use measured tolerances (mean < 0.05, max ≤ 1) in documentation, not "byte-identical" or "exact".
