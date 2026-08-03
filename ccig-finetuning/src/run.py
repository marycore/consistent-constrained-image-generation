from __future__ import annotations

import argparse

import yaml

from src.common.types import TrainConfig
from src.registry import TRAINER_REGISTRY, build_trainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA finetune an open-source text-to-image model on the CCIG dataset."
    )
    parser.add_argument("--model", required=True, choices=sorted(TRAINER_REGISTRY))
    parser.add_argument("--config", required=True, help="Path to a configs/<model>.yaml file")
    parser.add_argument("--run-name", default=None, help="Overrides run_name from the config")
    parser.add_argument("--max-steps", type=int, default=None, help="Overrides max_steps from the config")
    parser.add_argument("--dataset", default=None, help="Overrides dataset_path from the config")
    parser.add_argument(
        "--init-ckpt",
        default=None,
        help="Path to a previous run's checkpoint directory to continue LoRA training from",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    if args.run_name is not None:
        raw_config["run_name"] = args.run_name
    if args.max_steps is not None:
        raw_config["max_steps"] = args.max_steps
    if args.dataset is not None:
        raw_config["dataset_path"] = args.dataset
    if args.init_ckpt is not None:
        raw_config["init_ckpt"] = args.init_ckpt

    config = TrainConfig(**raw_config)
    trainer = build_trainer(args.model)
    checkpoint_dir = trainer.train(config)
    print(f"[{args.model}] LoRA checkpoint written to {checkpoint_dir}")


if __name__ == "__main__":
    main()
