## Fine-tuning not supported: `flux_1_schnell`

This model is **inference-only** in this benchmarking suite.

If you run:

```bash
python -m src.common.cli finetune --model flux_1_schnell --data ... --images_root ... --out ... --seed 42
```

the CLI will call the runner’s `finetune()` and the suite will raise:

**`NotImplementedError: Fine-tuning is not supported for flux_1_schnell (inference-only). Use flux_1_dev for fine-tuning.`**

Use `flux_1_dev` for fine-tuning (QLoRA/LoRA).

