from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_experiment_config(path: str | Path | None = None) -> Dict[str, Any]:
    if path is None:
        base = Path(__file__).resolve().parent.parent.parent
        p = base / "configs" / "experiment.yaml"
    else:
        p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

