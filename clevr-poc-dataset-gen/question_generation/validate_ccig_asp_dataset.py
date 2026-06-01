#!/usr/bin/env python3
"""Validate SAT/UNSAT of constraint programs in a CCIG dataset JSONL using clingo."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ccig_template_lib import load_background, record_to_asp_code, validate_with_clingo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check logical consistency (satisfiability) of records in a CCIG JSONL file."
    )
    p.add_argument(
        "input_jsonl",
        help="Path to ccig_asp_dataset.jsonl (or combo file).",
    )
    p.add_argument("--clingo_bin", default="clingo", help="Path to clingo binary.")
    p.add_argument(
        "--clingo_time_limit",
        type=int,
        default=10,
        help="Seconds per clingo call (--time-limit).",
    )
    p.add_argument(
        "--progress_every",
        type=int,
        default=10,
        help="Print progress every N records (0 = only summary).",
    )
    p.add_argument(
        "--background_asp",
        default="data/general_constraints.txt",
        help="Background ASP ontology (used when record has asp_rules, not asp_code).",
    )
    p.add_argument(
        "--relationship_axioms",
        default="data/ccig_relationship_axioms.txt",
        help="ASP axioms for hasRelationship/3.",
    )
    p.add_argument(
        "--fail_on_unsat",
        action="store_true",
        help="Exit with code 1 if any record is UNSAT.",
    )
    p.add_argument(
        "--show_unsat",
        action="store_true",
        help="Print id and combo_spec_id for UNSAT records.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.input_jsonl)
    if not path.exists():
        raise FileNotFoundError(path)

    if shutil.which(args.clingo_bin) is None:
        print(
            f"Error: clingo not found ({args.clingo_bin}). Install clingo or set --clingo_bin.",
            file=sys.stderr,
        )
        sys.exit(2)

    repo_root = Path(__file__).resolve().parents[1]
    background = load_background(
        repo_root / args.background_asp,
        repo_root / args.relationship_axioms,
    )

    sat = 0
    unsat = 0
    errors = 0
    t0 = time.monotonic()
    total_lines = sum(1 for _ in path.open(encoding="utf-8"))
    print(f"Validating {total_lines} records from {path} (clingo time-limit={args.clingo_time_limit}s)", flush=True)

    for line_no, line in enumerate(path.open(encoding="utf-8"), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            errors += 1
            print(f"line {line_no}: JSON error: {e}", file=sys.stderr)
            continue

        try:
            asp_code = record_to_asp_code(record, background)
        except ValueError as e:
            errors += 1
            print(f"line {line_no}: {e}", file=sys.stderr)
            continue

        ok, _out = validate_with_clingo(
            asp_code, args.clingo_bin, time_limit_sec=args.clingo_time_limit
        )
        if ok:
            sat += 1
        else:
            unsat += 1
            if args.show_unsat:
                print(
                    f"UNSAT line {line_no} id={record.get('id')} "
                    f"spec={record.get('combo_spec_id')}",
                    flush=True,
                )
        if args.progress_every and line_no % args.progress_every == 0:
            elapsed = time.monotonic() - t0
            print(
                f"  [{line_no}/{total_lines}] SAT={sat} UNSAT={unsat} errors={errors} "
                f"({elapsed:.0f}s elapsed)",
                flush=True,
            )

    total = sat + unsat
    elapsed = time.monotonic() - t0
    print(
        f"Checked {total} records in {elapsed:.1f}s: SAT={sat}, UNSAT={unsat}, errors={errors}",
        flush=True,
    )
    if unsat and not args.show_unsat:
        print("Run with --show_unsat to list UNSAT record ids.")

    if args.fail_on_unsat and unsat > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
