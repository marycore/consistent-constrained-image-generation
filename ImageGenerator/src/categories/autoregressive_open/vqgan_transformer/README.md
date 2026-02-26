# VQGAN + Transformer (Taming Transformers–style)

**Category**: `autoregressive_open`  
**Model ID**: `vqgan_transformer`

## Inference

`run` is not implemented yet (requires VQGAN decoder + autoregressive sampling). Use other models for inference.

## Fine-tuning (transformer prior)

The suite implements a **minimal** transformer prior training loop: image → dummy tokenization → causal LM over tokens conditioned on text. No external VQGAN is required for the training script to run; it uses a placeholder tokenization. For production use, plug in a real VQGAN encoder and decoder.

```bash
python -m src.common.cli finetune \
  --model vqgan_transformer \
  --data /path/to/dataset.json \
  --images_root /path/to/images \
  --out ckpts/vqgan_transformer \
  --max_steps 500 \
  --lr 1e-4 \
  --batch_size 2 \
  --grad_accum 4 \
  --seed 42
```

**Outputs**:
- `ckpts/vqgan_transformer/transformer.pt` – transformer prior state_dict + config
- `ckpts/vqgan_transformer/train_config.json`
- `ckpts/vqgan_transformer/train_log.jsonl`

**VRAM**: Modest (small transformer). Use `--resolution 256` (or default) and small batch.

## Inference with fine-tuned checkpoint

`run_finetuned` is not implemented yet; full generation requires a VQGAN decoder and autoregressive sampling over the learned prior.
