# SDXL Base

**Category**: `latent_diffusion_open`  
**Model ID**: `sdxl_base`

## Inference

```bash
python -m src.common.cli run \
  --model sdxl_base \
  --prompt_file configs/prompts_general.jsonl \
  --mode general \
  --seed 42
```

## Fine-tuning (LoRA)

LoRA is applied to the UNet (text-encoder LoRA optional). Default: 1024×1024; use `--resolution 512` to reduce VRAM.

```bash
python -m src.common.cli finetune \
  --model sdxl_base \
  --data /path/to/dataset.json \
  --images_root /path/to/images \
  --out ckpts/sdxl_lora \
  --max_steps 500 \
  --lr 1e-4 \
  --batch_size 1 \
  --grad_accum 4 \
  --seed 42
```

Optional: `--resolution 512` if you hit OOM.

**Outputs**: `ckpts/sdxl_lora/adapters/`, `train_config.json`, `train_log.jsonl`.

**VRAM**: ~12–16 GB at 1024×1024; ~8–10 GB at 512×512 with gradient checkpointing.

## Inference with fine-tuned adapters

```bash
python -m src.common.cli run_finetuned \
  --model sdxl_base \
  --ckpt ckpts/sdxl_lora \
  --prompt_file configs/prompts_general_specific.jsonl \
  --mode general_specific \
  --seed 42
```
