# Stable Diffusion 3.5

**Category**: `latent_diffusion_open`  
**Model ID**: `sd3_5`

## Inference

```bash
python -m src.common.cli run \
  --model sd3_5 \
  --prompt_file configs/prompts_general.jsonl \
  --mode general \
  --seed 42
```

This runner uses `stabilityai/stable-diffusion-3.5-large` through `diffusers` (`StableDiffusion3Pipeline`).

## Notes

- Better text conditioning than SD1.5/SDXL for long, structured prompts.
- Default generation in this runner is 1024x1024.
- Fine-tuning uses LoRA adapters on the SD3 transformer.
- High VRAM is strongly recommended.

## Fine-tuning (stage 1)

```bash
python -m src.common.cli finetune \
  --model sd3_5 \
  --data /path/to/dataset_stage1.json \
  --images_root /path/to/images \
  --caption_key text \
  --out ckpts/sd3_5_stage1 \
  --max_steps 200 \
  --lr 1e-4 \
  --batch_size 1 \
  --grad_accum 4 \
  --seed 42
```

## Fine-tuning (stage 2 from stage 1)

```bash
python -m src.common.cli finetune \
  --model sd3_5 \
  --data /path/to/dataset_stage2.json \
  --images_root /path/to/images \
  --caption_key text \
  --out ckpts/sd3_5_stage2 \
  --init_ckpt ckpts/sd3_5_stage1 \
  --max_steps 200 \
  --lr 5e-5 \
  --batch_size 1 \
  --grad_accum 4 \
  --seed 42
```

## Run with fine-tuned checkpoint

```bash
python -m src.common.cli run_finetuned \
  --model sd3_5 \
  --ckpt ckpts/sd3_5_stage1 \
  --prompt_file configs/prompts_general.jsonl \
  --mode general \
  --seed 42
```

## Resource guidance

- Recommended: GPU with high VRAM (RunPod class GPUs).
- If model download fails due to gated access/rate limits, configure Hugging Face authentication on the runtime.
