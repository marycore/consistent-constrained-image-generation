from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_generation_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("models", {})
    cfg["models"].setdefault("closed", [])
    cfg.setdefault("generation", {})
    g = cfg["generation"]
    g.setdefault("width", 1024)
    g.setdefault("height", 1024)
    g.setdefault("num_images_per_prompt", 1)
    g.setdefault("seeds", [0])
    g.setdefault("max_retries", 3)
    g.setdefault("timeout_seconds", 180)
    cfg.setdefault("budget", {})
    b = cfg["budget"]
    b.setdefault("max_usd", 300)
    b.setdefault("track_estimated_cost", True)
    b.setdefault("cost_overrides_usd", {})
    cfg.setdefault("prompts", {})
    cfg["prompts"].setdefault("path", "configs/prompts_general.jsonl")
    return cfg
