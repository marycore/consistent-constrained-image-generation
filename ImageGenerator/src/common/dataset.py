"""Dataset loader for fine-tuning: single JSON file with (image path, text) records."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


def load_dataset_json(path: str | Path) -> List[dict]:
    """
    Load dataset from a single JSON file.

    Expected format: array of objects with "id", "image", "text":
    [{"id": "000001", "image": "images/000001.png", "text": "caption with constraints"}, ...]

    Returns list of dicts (unchanged). Caller must resolve image paths with images_root.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Dataset JSON must be a list of records, got {type(data)}")
    for i, rec in enumerate(data):
        if not isinstance(rec, dict) or "image" not in rec or "text" not in rec:
            raise ValueError(
                f"Record at index {i} must be a dict with 'image' and 'text' keys, got {type(rec)}"
            )
    return data


def resolve_image_paths(records: List[dict], images_root: str | Path) -> List[dict]:
    """Resolve relative image paths in records against images_root. Mutates records."""
    root = Path(images_root)
    if not root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")
    out = []
    for rec in records:
        rel = rec["image"]
        full = root / rel
        if not full.exists():
            raise FileNotFoundError(f"Image file not found: {full} (record id={rec.get('id', '?')})")
        out.append({**rec, "image": str(full)})
    return out


def train_val_split(
    records: List[dict],
    val_ratio: float,
    seed: int,
) -> Tuple[List[dict], List[dict]]:
    """Split records into train and validation by val_ratio (e.g. 0.05 = 5% val)."""
    if val_ratio < 0 or val_ratio >= 1:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    shuffled = list(records)
    random.seed(seed)
    random.shuffle(shuffled)
    n = len(shuffled)
    n_val = max(0, int(n * val_ratio))
    n_train = n - n_val
    return shuffled[:n_train], shuffled[n_train:]


class ImageTextDataset(Dataset):
    """
    PyTorch Dataset that yields (PIL.Image, text) pairs from pre-resolved records.

    Records must have "image" (absolute path) and "text" keys.
    """

    def __init__(self, records: List[dict]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, str]:
        rec = self.records[idx]
        path = rec["image"]
        text = rec["text"]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {path}: {e}") from e
        return img, text


def get_train_val_datasets(
    dataset_path: str | Path,
    images_root: str | Path,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[ImageTextDataset, ImageTextDataset]:
    """
    Load JSON dataset, resolve paths, split train/val, return two ImageTextDatasets.
    """
    records = load_dataset_json(dataset_path)
    records = resolve_image_paths(records, images_root)
    if not records:
        raise ValueError("Dataset is empty after loading.")
    train_records, val_records = train_val_split(records, val_ratio, seed)
    return ImageTextDataset(train_records), ImageTextDataset(val_records)
