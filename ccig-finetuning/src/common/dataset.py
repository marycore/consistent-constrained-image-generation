from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .types import TrainExample


def load_examples(
    dataset_path: str | Path, images_dir: str | Path, prompt_field: str = "medium"
) -> Iterator[TrainExample]:
    """Read data/finetune-dataset/finetune-dataset.json and yield one training
    example per record, pairing each CLEVR image with its constraint description.
    """
    images_dir = Path(images_dir)
    with Path(dataset_path).open("r", encoding="utf-8") as f:
        records = json.load(f)

    for rec in records:
        yield TrainExample(
            id=rec["id"],
            image_path=images_dir / rec["image"],
            prompt=rec["constraints"][0][prompt_field],
        )
