## Fine-tuning dataset format (unified CLI)

The `finetune` command expects:

1. **`--data`**: path to a **single JSON file** that is an array of records.
2. **`--images_root`**: root directory used to resolve relative `image` paths in that JSON.

Each record must have:

- `id` – string identifier.
- `image` – path relative to `--images_root` (e.g. `images/000001.png`).
- `text` – caption / prompt for the image (with constraints if desired).

Example `dataset.json`:

```json
[
  {"id": "000001", "image": "images/000001.png", "text": "a photo of a cat on a sofa, photorealistic"},
  {"id": "000002", "image": "images/000002.png", "text": "a watercolor landscape with mountains and a lake"}
]
```

Example layout:

- `dataset.json` (the file passed to `--data`)
- `images/000001.png`, `images/000002.png`, … (paths relative to `--images_root`)

Example command (from project root):

```bash
python -m src.common.cli finetune \
  --model sd15 \
  --data configs/finetune_dataset/dataset.json \
  --images_root configs/finetune_dataset \
  --out ckpts/sd15_lora \
  --seed 42
```

Train/val split is done inside the loader using `--val_ratio` (default 0.05).
