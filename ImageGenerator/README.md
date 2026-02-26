## ImageGenerator – Benchmarking Suite

This project implements a **structured benchmarking suite** for text‑to‑image models, covering:

- One‑shot inference with general constraints  
- One‑shot inference with general + specific constraints  
- Optional fine‑tuning on a small dataset for supported open models

The suite is driven by a **unified CLI**, config files, and per‑model runners.

### Folder layout (high level)

- `requirements.txt` – shared dependencies for the suite.
- `configs/`
  - `experiment.yaml` – default steps, guidance, resolution per model.
  - `prompts_general.jsonl` – prompts with general constraints.
  - `prompts_general_specific.jsonl` – prompts with general + specific constraints.
  - `finetune_dataset/README.md` – dataset format: single JSON array `[{ "id", "image", "text" }, ...]` and an images root to resolve `image` paths.
- `src/`
  - `common/` – CLI, config, prompt handling, IO, registry, types, seeding.
  - `categories/` – model implementations grouped by category:
    - `latent_diffusion_open/` (`sd15`, `sdxl_base`, `pixart_sigma`)
    - `rectified_flow_open/` (`flux_1_dev`, `flux_1_schnell`)
    - `autoregressive_open/` (`vqgan_transformer` – minimal transformer prior finetune)
    - `closed_multimodal_transformer_api/` (`dalle3`, `gpt_image_1`, `gemini_image`)
    - `closed_diffusion_api/` (`imagen_api`, stubbed)
- `outputs/` – auto‑created; images + JSON metadata written here.

### Unified CLI

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# One‑shot inference
python -m src.common.cli run \
  --model sd15 \
  --prompt_file configs/prompts_general.jsonl \
  --mode general \
  --seed 42

# Fine‑tune (supported open models: sd15, sdxl_base, pixart_sigma, flux_1_dev, vqgan_transformer)
# Dataset: single JSON array of { "id", "image", "text" }; --images_root resolves image paths.
python -m src.common.cli finetune \
  --model sd15 \
  --data /path/to/dataset.json \
  --images_root /path/to/images \
  --out ckpts/sd15_lora \
  --max_steps 500 \
  --lr 1e-4 \
  --batch_size 1 \
  --grad_accum 4 \
  --seed 42

# Run with a fine‑tuned checkpoint
python -m src.common.cli run_finetuned \
  --model sd15 \
  --ckpt ckpts/sd15_lora \
  --prompt_file configs/prompts_general_specific.jsonl \
  --mode general_specific \
  --seed 42
```

Each run writes:

- `outputs/<category>/<model_id>/<mode>/<prompt_hash>__seed<seed>.png`  
- Matching `...json` with metadata (model, category, mode, full prompt, seed, steps, guidance_scale, resolution, dtype, device, scheduler, timestamp).


