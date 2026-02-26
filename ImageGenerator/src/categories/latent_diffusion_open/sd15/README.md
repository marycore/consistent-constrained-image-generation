# SD 1.5 (Stable Diffusion v1.5)

**Category**: `latent_diffusion_open`  
**Model ID**: `sd15`

## Inference

```bash
python -m src.common.cli run \
  --model sd15 \
  --prompt_file configs/prompts_general.jsonl \
  --mode general \
  --seed 42
```

## Fine-tuning (LoRA)

LoRA is applied to the UNet (diffusers + PEFT). Default: 512×512, batch_size=1, grad_accum=4.

**Dataset format**: Single JSON file (array of `{ "id", "image", "text" }`) and an `--images_root` to resolve image paths.

```bash
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
```

**Outputs**:
- `ckpts/sd15_lora/adapters/` – LoRA weights (PEFT format)
- `ckpts/sd15_lora/train_config.json` – training config
- `ckpts/sd15_lora/train_log.jsonl` – step-wise log

**VRAM**: ~8–10 GB with fp16 and gradient checkpointing. If OOM, use `--resolution 512` (default) and `--grad_accum 8`.

## Inference with fine-tuned adapters

```bash
python -m src.common.cli run_finetuned \
  --model sd15 \
  --ckpt ckpts/sd15_lora \
  --prompt_file configs/prompts_general_specific.jsonl \
  --mode general_specific \
  --seed 42
```

Images and metadata are written to `outputs/latent_diffusion_open/sd15/<mode>/`.
