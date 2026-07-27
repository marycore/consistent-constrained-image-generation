from __future__ import annotations

import argparse
from pathlib import Path

from src.closed.registry import MODEL_REGISTRY as CLOSED_MODEL_REGISTRY
from src.closed.registry import build_model as build_closed_model
from src.common.io import append_manifest, load_prompts
from src.common.scene_setup import scene_setup_text
from src.common.types import GenerationRecord
from src.open.registry import MODEL_REGISTRY as OPEN_MODEL_REGISTRY
from src.open.registry import build_model as build_open_model

ALL_MODELS = sorted({**CLOSED_MODEL_REGISTRY, **OPEN_MODEL_REGISTRY})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate images for CCIG eval prompts using a text-to-image model."
    )
    parser.add_argument("--model", required=True, choices=ALL_MODELS)
    parser.add_argument(
        "--dataset", help="Path to a ccig_eval_dataset_{SAT,UNSAT}.jsonl file",
        default="../data/ccig_eval_dataset_SAT.jsonl"
    )
    parser.add_argument("--prompt-field", default="medium", choices=["short", "medium", "long"])
    parser.add_argument("--out", default="../data/generated_images")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to a LoRA finetuned checkpoint directory (open-source models only), "
        "as produced by ccig-finetuning.",
    )
    args = parser.parse_args()

    is_closed_model = args.model in CLOSED_MODEL_REGISTRY
    if args.model in OPEN_MODEL_REGISTRY:
        model = build_open_model(args.model, checkpoint=args.checkpoint)
    else:
        if args.checkpoint is not None:
            parser.error(f"--checkpoint is not supported for closed model '{args.model}'")
        model = build_closed_model(args.model)
    out_dir = Path(args.out) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    for i, item in enumerate(load_prompts(args.dataset, args.prompt_field)):
        if args.limit is not None and i >= args.limit:
            break

        # check if outdir exists... if not, create it
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)


        image_path = out_dir / f"{item.id}-{args.prompt_field}.png"
        setup_text = scene_setup_text(
            item.number_of_objects, item.domain, with_background=is_closed_model
        )
        full_prompt = f"{setup_text} {item.text}"
        try:
            image = model.generate(full_prompt)
            image.save(image_path)
            record = GenerationRecord(
                id=item.id,
                model=args.model,
                prompt=item.text,
                prompt_field=args.prompt_field,
                scene_generation_setup=setup_text,
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
                prompt_field=args.prompt_field,
                scene_generation_setup=setup_text,
                image_path=None,
                success=False,
                error=repr(e),
            )
            print(f"[fail] {item.id}: {e}")

        append_manifest(manifest_path, record)


if __name__ == "__main__":
    main()
