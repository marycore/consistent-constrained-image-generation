## ImageGenerator – Benchmarking Suite

This project implements a **structured benchmarking suite** for text‑to‑image models, covering:

- One‑shot inference with general constraints  
- One‑shot inference with general + specific constraints  
- Optional fine‑tuning on a small dataset for supported open models

The suite is driven by a **unified CLI**, config files, and per‑model runners.

### Folder layout (high level)

- `requirements.txt` – shared dependencies for the suite.
- `configs/` – paths here are relative to project root.
  - `experiment.yaml` – default steps, guidance, resolution per model.
  - `prompts_general.jsonl` – prompts with general constraints.
  - `prompts_general_specific.jsonl` – prompts with general + specific constraints.
  - `finetune_dataset/` – dataset format (single JSON array with `id`, `image`, and caption field via `--caption_key`), CLIP‑77 compact captions; see `finetune_dataset/README.md`.
- `src/`
  - `common/` – CLI, config, prompt handling, IO, registry, types, seeding.
  - `categories/` – model implementations grouped by category:
    - `latent_diffusion_open/` (`sd15`, `sdxl_base`, `pixart_sigma`, `sd3_5`, `deepfloyd_if`)
    - `rectified_flow_open/` (`flux_1_dev`, `flux_1_schnell`)
    - `autoregressive_open/` (`vqgan_transformer` – minimal transformer prior finetune)
    - `latent_diffusion_accelerated_distilled/` (`z_image_turbo`, inference only)
    - `closed_multimodal_transformer_api/` (`dalle3`, `gpt_image_1`, `gemini_image`)
    - `closed_diffusion_api/` (`imagen_api`, stubbed)
- `outputs/` – auto‑created; images + JSON metadata written here.
- `scripts/` – `compress_clevr_to_clip77.py` (build dataset_clip77.json), `prompt_to_compact.py` (convert prompts to compact style for inference); see `configs/finetune_dataset/README.md`.

### Unified CLI

From the project root (directory containing `src/`). Run each command as **one line**, or end lines with `\` to continue in the shell:

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

# Fine‑tune stage 1 (supported open models: sd15, sdxl_base, pixart_sigma, sd3_5, flux_1_dev, vqgan_transformer)
# Dataset: single JSON array with { "id", "image", <caption_field> }; use --caption_key (default: text, e.g. pred).
python -m src.common.cli finetune \
  --model sd15 \
  --data /path/to/dataset_stage1.json \
  --images_root /path/to/images \
  --caption_key text \
  --out ckpts/sd15_stage1 \
  --max_steps 500 \
  --lr 1e-4 \
  --batch_size 1 \
  --grad_accum 4 \
  --seed 42

# Fine-tune stage 2 (continue from stage 1 checkpoint)
python -m src.common.cli finetune \
  --model sd15 \
  --data /path/to/dataset_stage2.json \
  --images_root /path/to/images \
  --caption_key text \
  --out ckpts/sd15_stage2 \
  --init_ckpt ckpts/sd15_stage1 \
  --max_steps 300 \
  --lr 5e-5 \
  --batch_size 1 \
  --grad_accum 4 \
  --seed 42

# Fine-tune with longer text context on T5-based models (pixart_sigma, flux_1_dev)
python -m src.common.cli finetune \
  --model pixart_sigma \
  --data /path/to/dataset_stage1.json \
  --images_root /path/to/images \
  --caption_key pred \
  --max_sequence_length 256 \
  --out ckpts/pixart_sigma_stage1_longctx \
  --max_steps 200 \
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

Prompt-length note:
- `sd15` and `sdxl_base` rely on CLIP-style context windows (typically short, ~77 tokens per encoder branch).
- `pixart_sigma` and `flux_1_dev` use T5-based text embeddings and now support `--max_sequence_length` in `finetune`.
- `sd3_5` supports LoRA fine-tuning and `run_finetuned` in this benchmark runner (experimental, high VRAM recommended).
- `deepfloyd_if` remains inference-only in this benchmark integration.

### RunPod workflow (Cursor + persistent jobs)

This setup supports:
- local editing in Cursor + sync to RunPod
- running long training jobs in `tmux` so they continue after disconnect/reboot
- `run_finetuned` inference with stage 1 or stage 2 checkpoints (same CLI as locally)
- reconnecting from Cursor any time

1) Configure RunPod target:

```bash
cp scripts/runpod/runpod.env.example scripts/runpod/runpod.env
# then edit scripts/runpod/runpod.env with your host/key/path
```

2) Sync project from local machine to RunPod:

```bash
# Full mirror sync (includes --delete to remove stale remote files)
./scripts/runpod/sync_to_runpod.sh

# Fast sync for day-to-day iteration (only changed/untracked files)
./scripts/runpod/sync_changed_to_runpod.sh
```

3) Bootstrap remote environment (venv, pip deps, tmux):

```bash
./scripts/runpod/bootstrap_remote.sh
```

4) Start a persistent training session in `tmux`:

```bash
./scripts/runpod/start_tmux.sh sd15_stage1 \
  "source .venv/bin/activate && python -m src.common.cli finetune --model sd15 --data configs/finetune_dataset/dataset.json --images_root configs/finetune_dataset/images --caption_key text --out ckpts/sd15_stage1 --max_steps 500 --lr 1e-4 --batch_size 1 --grad_accum 4 --seed 42"
```

4b) **Fine-tuned inference** on the pod uses `run_finetuned` with `--ckpt` pointing at the stage you want (stage 1 vs stage 2 is just a different checkpoint directory). Copy-paste examples, HF cache env, and tmux variants are in [`RUNPOD_README.md`](RUNPOD_README.md) (section **5) Fine-tuned inference**).

5) Reconnect and monitor:

```bash
./scripts/runpod/list_tmux.sh
./scripts/runpod/attach_tmux.sh sd15_stage1
./scripts/runpod/tail_log.sh sd15_stage1
```

6) Pull outputs/checkpoints back to local:

```bash
./scripts/runpod/sync_from_runpod.sh
```

Notes:
- `scripts/runpod/runpod.env` is git-ignored (keeps host/key private).
- `sync_to_runpod.sh` and `sync_from_runpod.sh` are full mirrors with `--delete` (except local `.git/`, `.venv/`, `.env/`).
- For Cursor remote development, add the same RunPod SSH host in Cursor's SSH targets and open `RUNPOD_REMOTE_DIR` there.


