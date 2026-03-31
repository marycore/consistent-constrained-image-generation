# PixArt-Sigma

**Category**: `latent_diffusion_open`  
**Model ID**: `pixart_sigma`

## Inference

```bash
python -m src.common.cli run \
  --model pixart_sigma \
  --prompt_file configs/prompts_general.jsonl \
  --mode general \
  --seed 42
```

## Fine-tuning (LoRA)

LoRA is applied to the transformer (diffusers + PEFT). Default: 1024×1024; use `--resolution 512` to reduce VRAM.

```bash
python -m src.common.cli finetune \
  --model pixart_sigma \
  --data /path/to/dataset.json \
  --images_root /path/to/images \
<<<<<<< HEAD
  --caption_key pred \
=======
>>>>>>> 944ef832ccc5c8e13f4cb8c0be1cb6304a2ad873
  --out ckpts/pixart_lora \
  --max_steps 500 \
  --lr 1e-4 \
  --batch_size 1 \
  --grad_accum 4 \
  --seed 42
```

**Outputs**: `ckpts/pixart_lora/adapters/`, `train_config.json`, `train_log.jsonl`.

**VRAM**: ~10–14 GB at 1024×1024; use `--resolution 512` and `--grad_accum 8` if OOM.

## Inference with fine-tuned adapters

```bash
python -m src.common.cli run_finetuned \
  --model pixart_sigma \
  --ckpt ckpts/pixart_lora \
  --prompt_file configs/prompts_general_specific.jsonl \
  --mode general_specific \
  --seed 42
```
