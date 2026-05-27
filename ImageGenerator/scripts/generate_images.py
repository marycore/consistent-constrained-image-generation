from __future__ import annotations

import argparse

from src.closed_api.pipeline import run_closed_generation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch generation for closed/API models.")
    p.add_argument("--config", required=True, help="Path to generation config yaml.")
    p.add_argument("--models", nargs="*", default=None, help="Closed model ids to run.")
    p.add_argument("--levels", nargs="*", default=None, help="Complexity levels to include (e.g. L0 L1).")
    p.add_argument("--limit-per-level", type=int, default=None, help="Max prompts per complexity level.")
    p.add_argument("--seeds", nargs="*", type=int, default=None, help="Override seeds (e.g. 0 1).")
    p.add_argument("--dry-run", action="store_true", help="Plan and write metadata without API calls.")
    p.add_argument("--resume", action="store_true", help="Skip existing output files.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stats = run_closed_generation(
        config_path=args.config,
        models=args.models,
        levels=args.levels,
        limit_per_level=args.limit_per_level,
        seeds=args.seeds,
        dry_run=args.dry_run,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(f"Done. attempted={stats['attempted']} succeeded={stats['succeeded']} failed={stats['failed']} skipped={stats['skipped']}")


if __name__ == "__main__":
    main()
