# DeepFloyd IF (Stage I)

**Category**: `latent_diffusion_open`  
**Model ID**: `deepfloyd_if`

## Inference

```bash
python -m src.common.cli run \
  --model deepfloyd_if \
  --prompt_file configs/prompts_general.jsonl \
  --mode general \
  --seed 42
```

This runner uses `DeepFloyd/IF-I-XL-v1.0` (stage-I pipeline) for text-aligned generation.

## Notes

- Strong text grounding for complex prompts (T5-based conditioning).
- Current runner executes IF **stage-I only** (lower native resolution output).
- Inference-only in this benchmark integration.

## Not supported (yet)

```bash
python -m src.common.cli finetune --model deepfloyd_if ...
python -m src.common.cli run_finetuned --model deepfloyd_if ...
```

Both commands currently raise `NotImplementedError`.

## Resource guidance

- Heavy model; prefer RunPod/high-VRAM GPUs.
- For higher final image quality, add IF stage-II/III upscalers in a later pipeline extension.
