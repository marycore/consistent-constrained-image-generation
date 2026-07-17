"""
Compile finetune-dataset.json from data/finetune-dataset/original-clevr-train-scenes.json +
images/. For each image with a matching scene record, reconstructs the scene, grounds one or
more true C1-C9 constraint statements against it, and writes a caption combining a scene-count
preamble with the grounded constraint sentence(s).

Usage (run from clevr-ccig-dataset-gen/ so the src.common / src.finetune_dataset_gen package
imports resolve):
    python -m src.finetune_dataset_gen.compile_dataset \\
        --scenes data/finetune-dataset/original-clevr-train-scenes.json \\
        --images data/finetune-dataset/images \\
        --out data/finetune-dataset/finetune-dataset.json \\
        --constraints_per_image random --granularity medium --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional

from .grounders import ground_constraints
from .scene_reconstruct import reconstruct_scene

# No symlink -- the canonical dataset lives in the repo-root data/ folder, shared with
# eval_dataset_gen. Anchor defaults to it via __file__ so they work regardless of the caller's
# cwd (src/finetune_dataset_gen/compile_dataset.py -> parents[3] == repo root).
_REPO_DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def _parse_constraints_per_image(raw: str):
    if raw in ("random", "all"):
        return raw
    return int(raw)


def _select_constraints(
    grounded: List[dict], mode, rng: random.Random, max_constraints: int
) -> List[dict]:
    """
    Apply the constraints_per_image knob to the full pool of grounded constraints.
    A scene can be truly satisfiable by dozens of constraint sentences (e.g. many distinct
    counting statements). "random" is bounded by `max_constraints` so captions stay a plausible
    prompt length; "all" is intentionally exhaustive (opt-in, produces long captions); an explicit
    int is honored as requested (capped only by pool size).
    """
    if not grounded:
        return []
    if mode == "all":
        return grounded
    if mode == "random":
        k = rng.randint(1, min(max_constraints, len(grounded)))
        return rng.sample(grounded, k)
    k = min(int(mode), len(grounded))
    return rng.sample(grounded, k)


def build_caption(n_objects: int, selected: List[dict], granularity: str) -> str:
    preamble = f"There are {n_objects} objects in the scene."
    sentences = [c[granularity] for c in selected]
    return " ".join([preamble] + sentences)


def compile_dataset(
    scenes_path: Path,
    images_root: Path,
    out_path: Path,
    *,
    constraints_per_image,
    granularity: str = "medium",
    classes: Optional[List[str]] = None,
    seed: int = 0,
    limit: Optional[int] = None,
    max_constraints: int = 3,
) -> List[dict]:
    with scenes_path.open("r", encoding="utf-8") as f:
        scene_records = json.load(f)

    available_images = {p.name for p in images_root.iterdir()}
    records = [r for r in scene_records if r.get("image") in available_images]
    if limit is not None:
        records = records[:limit]

    rng = random.Random(seed)
    out_records = []
    n_no_constraints = 0

    for rec in records:
        scene = reconstruct_scene(rec)
        n_objects = len(scene["objects"])
        grounded = ground_constraints(scene, classes=classes, rng=rng)
        selected = _select_constraints(grounded, constraints_per_image, rng, max_constraints)
        if not selected:
            n_no_constraints += 1
            continue
        caption = build_caption(n_objects, selected, granularity)
        out_records.append({
            "id": rec["id"],
            "image": rec["image"],
            "text": caption,
            "n_objects": n_objects,
            "constraints": selected,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_records, f, indent=2)

    print(
        f"Compiled {len(out_records)} records to {out_path} "
        f"({n_no_constraints} images skipped: no groundable constraint found)."
    )
    return out_records


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compile finetune-dataset.json from CLEVR scene records.")
    p.add_argument("--scenes", default=str(_REPO_DATA_DIR / "finetune-dataset" / "original-clevr-train-scenes.json"))
    p.add_argument("--images", default=str(_REPO_DATA_DIR / "finetune-dataset" / "images"))
    p.add_argument("--out", default=str(_REPO_DATA_DIR / "finetune-dataset" / "finetune-dataset.json"))
    p.add_argument(
        "--constraints_per_image", default="random",
        help="1 | <int> | 'random' | 'all' -- how many grounded constraints go into each caption.",
    )
    p.add_argument("--granularity", choices=["short", "medium", "long"], default="medium")
    p.add_argument(
        "--classes", nargs="*", default=None,
        help="Restrict to specific constraint classes, e.g. --classes C1 C8. Default: all C1-C9.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Only compile the first N matching images (for smoke tests).")
    p.add_argument(
        "--max_constraints", type=int, default=3,
        help="Upper bound on how many constraint sentences 'random' mode may pick (default: 3). "
             "Ignored by 'all' (exhaustive) and explicit-int modes.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    compile_dataset(
        Path(args.scenes),
        Path(args.images),
        Path(args.out),
        constraints_per_image=_parse_constraints_per_image(args.constraints_per_image),
        granularity=args.granularity,
        classes=args.classes,
        seed=args.seed,
        limit=args.limit,
        max_constraints=args.max_constraints,
    )


if __name__ == "__main__":
    main()
