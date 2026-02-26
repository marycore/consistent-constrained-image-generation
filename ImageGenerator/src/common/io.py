from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, List

from PIL import Image

from .types import PromptRecord, GenerationMetadata


def read_jsonl(path: str | Path) -> List[PromptRecord]:
    """Read a JSONL file into a list of PromptRecord dictionaries."""

    records: List[PromptRecord] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(
                PromptRecord(
                    id=str(obj["id"]),
                    prompt=str(obj["prompt"]),
                    constraints_general=str(obj.get("constraints_general", "")),
                    constraints_specific=str(obj.get("constraints_specific", "")),
                )
            )
    return records


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def save_image_and_metadata(
    *,
    image: Image.Image,
    metadata: GenerationMetadata,
    output_root: str | Path,
    category: str,
    model_id: str,
    mode: str,
) -> None:
    base_dir = Path(output_root) / category / model_id / mode
    base_dir.mkdir(parents=True, exist_ok=True)

    h = prompt_hash(metadata.full_prompt)
    filename_stem = f"{h}__seed{metadata.seed}"

    image_path = base_dir / f"{filename_stem}.png"
    json_path = base_dir / f"{filename_stem}.json"

    image.save(str(image_path))

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, indent=2)


def save_train_config(ckpt_dir: str | Path, config: dict) -> None:
    """Write training config to <ckpt_dir>/train_config.json."""
    p = Path(ckpt_dir)
    p.mkdir(parents=True, exist_ok=True)
    with (p / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def append_train_log(ckpt_dir: str | Path, log_entry: dict) -> None:
    """Append a single log line to <ckpt_dir>/train_log.jsonl."""
    p = Path(ckpt_dir)
    p.mkdir(parents=True, exist_ok=True)
    with (p / "train_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def adapters_dir(ckpt_dir: str | Path) -> Path:
    """Return path to <ckpt_dir>/adapters/ for LoRA/QLoRA weights."""
    d = Path(ckpt_dir) / "adapters"
    d.mkdir(parents=True, exist_ok=True)
    return d

