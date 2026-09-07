# machin-diffusion

**Stable Diffusion Turbo inference in pure [machin](https://github.com/javimosch/machin) (MFL).** No Python, no PyTorch, no libtorch — one static binary.

## Status

**Phase 1: CLIP text encoder — working.**

| Component | Status |
|-----------|--------|
| Safetensors reader (mmap + Windows fallback) | ✅ |
| CLIP BPE tokenizer (Python pre-processing) | ✅ |
| CLIP text encoder (23-layer transformer, fp32) | ✅ |
| UNet (conv-based, 1-step diffusion) | 🔜 |
| VAE decoder (latent → image) | 🔜 |
| Image output (PPM/PNG) | 🔜 |

## Architecture

```
prompt → tokenize.py → token_ids.txt
                              ↓
         machin-diffusion.exe → CLIP text encoder (pure MFL)
                              ↓
         text embeddings [77 × 1024]  ← CURRENT
                              ↓
         UNet forward pass (1 step)  ← TODO
                              ↓
         VAE decoder → PPM image     ← TODO
```

### Weight loading

Safetensors files are used **as-is** — zero conversion. A small `.idx` sidecar (28KB) is pre-extracted with `scripts/extract_header.py` for tensor metadata. On POSIX the safetensors is mmap'd (zero-copy); on Windows (where mmap is stubbed) it's read into memory with `read_file_raw`.

### CLIP text encoder

SD-Turbo uses CLIP ViT-H/14:
- hidden_size=1024, 16 heads, head_dim=64, 23 layers
- intermediate_size=4096, vocab_size=49408, max_pos=77
- GELU activation, LayerNorm (eps=1e-5)
- **Bidirectional** (non-causal) attention

Implemented in pure MFL using `peek_f32`/`poke_f32` for buffer ops and scalar fp32 matmul loops.

## Usage

```bash
# 1. Download model (text encoder + tokenizer)
mkdir -p models/sd-turbo/{text_encoder,tokenizer}
curl -sL -o models/sd-turbo/text_encoder/model.safetensors \
  "https://huggingface.co/stabilityai/sd-turbo/resolve/main/text_encoder/model.safetensors"
curl -sL -o models/sd-turbo/tokenizer/vocab.json \
  "https://huggingface.co/stabilityai/sd-turbo/resolve/main/tokenizer/vocab.json"
curl -sL -o models/sd-turbo/tokenizer/merges.txt \
  "https://huggingface.co/stabilityai/sd-turbo/resolve/main/tokenizer/merges.txt"

# 2. Extract tensor index sidecar
python3 scripts/extract_header.py models/sd-turbo/text_encoder/model.safetensors

# 3. Tokenize a prompt
python3 scripts/tokenize.py models/sd-turbo/tokenizer/vocab.json \
  models/sd-turbo/tokenizer/merges.txt "a cat sitting on a chair" token_ids.txt

# 4. Build and run
./build.sh
./machin-diffusion models/sd-turbo token_ids.txt
```

### Cross-compile for Windows

```bash
machin encode safetensors.src clip.src main.src > machin-diffusion.mfl
machin build machin-diffusion.mfl --target windows -o machin-diffusion.exe
```

## Performance

CLIP text encoder on Intel i7-2700K (8 threads, scalar fp32, no SIMD):
- **24.8 seconds** for 77 tokens × 23 layers

Optimization path: replace scalar `linear()` loops with `dot_f32` builtin (vectorized fp32 dot product) — should give ~5-10x speedup.

## Files

| File | Role |
|------|------|
| `safetensors.src` | Safetensors mmap/reader + tensor index parser |
| `clip.src` | CLIP text encoder (transformer blocks, attention, MLP) |
| `main.src` | CLI entry point |
| `scripts/extract_header.py` | Extract safetensors tensor index to `.idx` sidecar |
| `scripts/tokenize.py` | CLIP BPE tokenizer (pre-processes prompt to token IDs) |
| `build.sh` | Encode + build |

## GPU path (future)

The hot primitives (`linear`, `attention`) are candidates for builtins. Today they dispatch to CPU scalar loops. A GPU backend (OpenCL/Vulkan/DirectML for AMD RX 6600) would be added at the builtin level — the MFL code stays unchanged.

Part of [**awesome-machin**](https://github.com/javimosch/awesome-machin).
