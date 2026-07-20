from __future__ import annotations

import argparse
from pathlib import Path

from src.closed.registry import MODEL_REGISTRY, build_model
from src.common.io import append_manifest, load_prompts
from src.common.types import GenerationRecord


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate images for CCIG eval prompts using a closed-source model."
    )
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument(
        "--dataset", required=True, help="Path to a ccig_eval_dataset_{SAT,UNSAT}.jsonl file"
    )
    parser.add_argument("--prompt-field", default="medium", choices=["short", "medium", "long"])
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    model = build_model(args.model)
    out_dir = Path(args.out) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    for i, item in enumerate(load_prompts(args.dataset, args.prompt_field)):
        if args.limit is not None and i >= args.limit:
            break

        image_path = out_dir / f"{item.id}.png"
        try:
            image = model.generate(item.text)
            image.save(image_path)
            record = GenerationRecord(
                id=item.id,
                model=args.model,
                prompt=item.text,
                image_path=str(image_path),
                success=True,
                error=None,
            )
            print(f"[ok]   {item.id}")
        except Exception as e:
            record = GenerationRecord(
                id=item.id,
                model=args.model,
                prompt=item.text,
                image_path=None,
                success=False,
                error=repr(e),
            )
            print(f"[fail] {item.id}: {e}")

        append_manifest(manifest_path, record)


if __name__ == "__main__":
    main()
