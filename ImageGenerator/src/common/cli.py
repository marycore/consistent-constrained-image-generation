from __future__ import annotations

import argparse

from . import io, prompts, seeds
from .config import load_experiment_config
from .registry import get_runner
from .types import Mode, FinetuneConfig

# Import runners for side-effect registration in the global registry.
# Latent diffusion (open)
from ..categories.latent_diffusion_open.sd15 import runner as _sd15_runner  # noqa: F401
from ..categories.latent_diffusion_open.sdxl_base import runner as _sdxl_runner  # noqa: F401
from ..categories.latent_diffusion_open.pixart_sigma import runner as _pixart_runner  # noqa: F401
from ..categories.latent_diffusion_open.sd3_5 import runner as _sd35_runner  # noqa: F401
from ..categories.latent_diffusion_open.deepfloyd_if import runner as _deepfloyd_if_runner  # noqa: F401

# Latent diffusion, accelerated/distilled
from ..categories.latent_diffusion_accelerated_distilled.z_image_turbo import (  # noqa: F401
    runner as _z_image_turbo_runner,
)
# Rectified flow (open)
from ..categories.rectified_flow_open.flux_1_dev import runner as _flux_dev_runner  # noqa: F401
from ..categories.rectified_flow_open.flux_1_schnell import runner as _flux_schnell_runner  # noqa: F401

# Autoregressive (open)
from ..categories.autoregressive_open.vqgan_transformer import runner as _vqgan_runner  # noqa: F401

# Closed APIs (inference only)
from ..categories.closed_multimodal_transformer_api.dalle3 import runner as _dalle3_runner  # noqa: F401
from ..categories.closed_multimodal_transformer_api.gpt_image_1 import runner as _gpt_image_1_runner  # noqa: F401
from ..categories.closed_multimodal_transformer_api.gemini_image import runner as _gemini_image_runner  # noqa: F401
from ..categories.closed_diffusion_api.imagen_api import runner as _imagen_api_runner  # noqa: F401



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified benchmarking CLI for text-to-image models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    run_p = subparsers.add_parser("run", help="Run one-shot inference.")
    run_p.add_argument("--model", required=True, help="Model identifier (e.g. sd15, sdxl_base).")
    run_p.add_argument("--prompt_file", required=True, help="Path to JSONL prompts file.")
    run_p.add_argument(
        "--mode",
        choices=["general", "general_specific"],
        required=True,
        help="Prompt construction mode.",
    )
    run_p.add_argument("--seed", type=int, required=True, help="Random seed.")

    # finetune
    ft_p = subparsers.add_parser("finetune", help="Fine-tune a supported open model.")
    ft_p.add_argument("--model", required=True, help="Model identifier.")
    ft_p.add_argument(
        "--data",
        required=True,
        help="Path to dataset JSON file (array of records with id, image, and caption field, e.g. text or pred).",
    )
    ft_p.add_argument("--images_root", required=True, help="Root path to resolve image paths in the dataset JSON.")
    ft_p.add_argument("--out", required=True, help="Output checkpoint directory (adapters + train_config.json + train_log.jsonl).")
    ft_p.add_argument("--max_steps", type=int, default=500, help="Max training steps (default: 500).")
    ft_p.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4).")
    ft_p.add_argument("--batch_size", type=int, default=1, help="Per-device batch size (default: 1).")
    ft_p.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps (default: 4).")
    ft_p.add_argument("--seed", type=int, required=True, help="Random seed.")
    ft_p.add_argument("--val_ratio", type=float, default=0.05, help="Validation split ratio 0–1 (default: 0.05).")
    ft_p.add_argument("--resolution", type=int, default=None, help="Optional resolution override (e.g. 512 to reduce VRAM).")
    ft_p.add_argument("--caption_key", default="text", help="Dataset field used as caption for training (default: text, e.g. pred).")
    ft_p.add_argument(
        "--max_sequence_length",
        type=int,
        default=None,
        help="Optional text token length override for supported models (e.g. PixArt/FLUX).",
    )
    ft_p.add_argument(
        "--init_ckpt",
        default=None,
        help="Optional previous fine-tuned checkpoint directory used to initialize another fine-tuning run.",
    )

    # run_finetuned
    rft_p = subparsers.add_parser("run_finetuned", help="Run inference with a fine-tuned checkpoint.")
    rft_p.add_argument("--model", required=True, help="Model identifier.")
    rft_p.add_argument("--ckpt", required=True, help="Checkpoint directory.")
    rft_p.add_argument("--prompt_file", required=True, help="Path to JSONL prompts file.")
    rft_p.add_argument(
        "--mode",
        choices=["general", "general_specific"],
        required=True,
        help="Prompt construction mode.",
    )
    rft_p.add_argument("--seed", type=int, required=True, help="Random seed.")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_experiment_config()

    seeds.set_seed(args.seed)

    if args.command == "run":
        runner = get_runner(args.model)
        prompt_records = io.read_jsonl(args.prompt_file)
        runner.run(
            prompts=prompt_records,
            mode=args.mode,  # type: ignore[arg-type]
            seed=args.seed,
            output_root=cfg.get("defaults", {}).get("output_root", "outputs"),
        )
    elif args.command == "finetune":
        runner = get_runner(args.model)
        config = FinetuneConfig(
            max_steps=args.max_steps,
            lr=args.lr,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            seed=args.seed,
            val_ratio=args.val_ratio,
            resolution=args.resolution,
            caption_key=args.caption_key,
            max_sequence_length=args.max_sequence_length,
        )
        runner.finetune(
            dataset_path=args.data,
            images_root=args.images_root,
            out_dir=args.out,
            config=config,
            init_ckpt_dir=args.init_ckpt,
        )
    elif args.command == "run_finetuned":
        runner = get_runner(args.model)
        prompt_records = io.read_jsonl(args.prompt_file)
        runner.run_finetuned(
            prompts=prompt_records,
            mode=args.mode,  # type: ignore[arg-type]
            seed=args.seed,
            output_root=cfg.get("defaults", {}).get("output_root", "outputs"),
            ckpt_dir=args.ckpt,
        )
    else:  # pragma: no cover - defensive
        raise RuntimeError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()

