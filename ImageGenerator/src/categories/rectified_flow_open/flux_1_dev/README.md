# FLUX.1-dev

**Category**: `rectified_flow_open`  
**Model ID**: `flux_1_dev`

## Inference

```bash
python -m src.common.cli run \
  --model flux_1_dev \
  --prompt_file configs/prompts_general.jsonl \
  --mode general \
  --seed 42
```

## Fine-tuning (QLoRA / LoRA)

The runner tries QLoRA (4-bit transformer) first; if unsupported, falls back to full LoRA with gradient checkpointing and bf16. Single-GPU friendly.

```bash
python -m src.common.cli finetune \
  --model flux_1_dev \
  --data /path/to/dataset.json \
  --images_root /path/to/images \
  --out ckpts/flux_lora \
  --max_steps 500 \
  --lr 1e-4 \
  --batch_size 1 \
  --grad_accum 4 \
  --seed 42
```

### Two-stage fine-tuning

```bash
# Stage 1
python -m src.common.cli finetune \
  --model flux_1_dev \
  --data /path/to/dataset_stage1.json \
  --images_root /path/to/images \
  --out ckpts/flux_stage1 \
  --max_steps 500 \
  --lr 1e-4 \
  --batch_size 1 \
  --grad_accum 4 \
  --seed 42

# Stage 2 (continue from stage 1)
python -m src.common.cli finetune \
  --model flux_1_dev \
  --data /path/to/dataset_stage2.json \
  --images_root /path/to/images \
  --out ckpts/flux_stage2 \
  --init_ckpt ckpts/flux_stage1 \
  --max_steps 300 \
  --lr 5e-5 \
  --batch_size 1 \
  --grad_accum 4 \
  --seed 42
```

Optional: `--resolution 512` to reduce VRAM.

**Outputs**: `ckpts/flux_lora/adapters/`, `train_config.json` (includes `use_qlora`), `train_log.jsonl`.

**VRAM**: QLoRA ~10 GB; full LoRA ~16–20 GB. If OOM, use `--resolution 512` and `--grad_accum 8`.

## Inference with fine-tuned adapters

```bash
python -m src.common.cli run_finetuned \
  --model flux_1_dev \
  --ckpt ckpts/flux_lora \
  --prompt_file configs/prompts_general_specific.jsonl \
  --mode general_specific \
  --seed 42
```
