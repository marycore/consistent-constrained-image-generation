from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .config import load_generation_config
from .models import COST_DEFAULTS
from .registry import build_closed_registry
from .types import PromptItem


def read_prompts(path: str | Path) -> list[PromptItem]:
    out: list[PromptItem] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(
                PromptItem(
                    prompt_id=str(obj.get("prompt_id", obj.get("id", "unknown"))),
                    complexity_level=str(obj.get("complexity_level", obj.get("level", "L0"))),
                    prompt=str(obj.get("prompt", "")),
                    constraints_general=str(obj.get("constraints_general", obj.get("constraints", ""))),
                    constraints_specific=str(obj.get("constraints_specific", "")),
                    template_family=obj.get("template_family"),
                    raw=obj,
                )
            )
    return out


def estimate_cost(
    *,
    models: Iterable[str],
    num_prompts: int,
    num_seeds: int,
    num_images: int,
    overrides: dict[str, float] | None = None,
) -> float | None:
    overrides = overrides or {}
    total = 0.0
    has_known = False
    for m in models:
        unit = overrides.get(m)
        if unit is None:
            unit = COST_DEFAULTS.get(m)
        if unit is None:
            continue
        has_known = True
        total += unit * num_prompts * num_seeds * num_images
    return total if has_known else None


def output_path_for(model_id: str, level: str, prompt_id: str, seed: int, img_idx: int) -> Path:
    return Path("outputs") / "generations" / model_id / level / prompt_id / f"seed_{seed}_img_{img_idx}.png"


def append_global_metadata(entry: dict) -> None:
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "generation_metadata.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_closed_generation(
    *,
    config_path: str,
    models: list[str] | None = None,
    levels: list[str] | None = None,
    limit_per_level: int | None = None,
    seeds: list[int] | None = None,
    dry_run: bool = False,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, int]:
    cfg = load_generation_config(config_path)
    registry = build_closed_registry()
    selected_models = models or list(cfg["models"].get("closed", []))
    prompts = read_prompts(cfg["prompts"]["path"])
    if levels:
        prompts = [p for p in prompts if p.complexity_level in set(levels)]
    if limit_per_level is not None:
        by_level: dict[str, list[PromptItem]] = {}
        for p in prompts:
            by_level.setdefault(p.complexity_level, []).append(p)
        clipped: list[PromptItem] = []
        for level, items in by_level.items():
            clipped.extend(items[:limit_per_level])
        prompts = clipped
    use_seeds = seeds if seeds is not None else list(cfg["generation"]["seeds"])
    est = estimate_cost(
        models=selected_models,
        num_prompts=len(prompts),
        num_seeds=len(use_seeds),
        num_images=int(cfg["generation"]["num_images_per_prompt"]),
        overrides=cfg["budget"].get("cost_overrides_usd", {}),
    )
    if cfg["budget"].get("track_estimated_cost", True) and est is not None and est > float(cfg["budget"]["max_usd"]):
        print(
            f"WARNING: estimated cost ${est:.2f} exceeds budget ${cfg['budget']['max_usd']:.2f}. "
            "Use --dry-run first or adjust config."
        )
    stats = {"attempted": 0, "succeeded": 0, "failed": 0, "skipped": 0}
    for model_id in selected_models:
        if model_id not in registry:
            print(f"Skipping unknown model: {model_id}")
            continue
        model = registry[model_id]
        for p in prompts:
            for seed in use_seeds:
                out_path = output_path_for(model_id, p.complexity_level, p.prompt_id, seed, 0)
                if resume and out_path.exists() and not overwrite:
                    stats["skipped"] += 1
                    continue
                if dry_run:
                    stats["attempted"] += 1
                    append_global_metadata(
                        {
                            "dry_run": True,
                            "model_id": model_id,
                            "provider": model.provider,
                            "prompt_id": p.prompt_id,
                            "complexity_level": p.complexity_level,
                            "output_path": str(out_path),
                            "seed_requested": seed,
                        }
                    )
                    continue
                stats["attempted"] += 1
                result = model.generate(
                    prompt=p.full_prompt,
                    output_path=str(out_path),
                    prompt_id=p.prompt_id,
                    complexity_level=p.complexity_level,
                    seed=seed,
                    width=int(cfg["generation"]["width"]),
                    height=int(cfg["generation"]["height"]),
                    num_images=int(cfg["generation"]["num_images_per_prompt"]),
                    timeout_seconds=int(cfg["generation"]["timeout_seconds"]),
                    max_retries=int(cfg["generation"]["max_retries"]),
                    constraints_general=p.constraints_general,
                    constraints_specific=p.constraints_specific,
                    template_family=p.template_family,
                )
                append_global_metadata(result.to_json())
                if result.success:
                    stats["succeeded"] += 1
                else:
                    stats["failed"] += 1
    return stats
