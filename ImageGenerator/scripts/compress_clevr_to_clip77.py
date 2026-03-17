#!/usr/bin/env python3
"""
Compress CLEVR-style captions to fit CLIP's 77-token limit (lossless where possible).
Reads dataset.json, writes dataset_clip77.json with same structure and shortened text.
Uses src.common.clevr_compact for conversion (multi-level shortening, never over 77).
"""

import json
import sys
from pathlib import Path

# Use common module so conversion matches prompt_to_compact and stays ≤77
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.common.clevr_compact import parse_clevr, build_compact, count_tokens


def main():
    in_path = Path(sys.argv[1] if len(sys.argv) > 1 else "configs/finetune_dataset/dataset.json")
    out_path = in_path.parent / "dataset_clip77.json"
    if len(sys.argv) > 2:
        out_path = Path(sys.argv[2])

    if not in_path.exists():
        print(f"Error: not found {in_path}")
        sys.exit(1)

    print(f"Reading {in_path}...")
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: dataset must be a JSON array")
        sys.exit(1)

    out_list = []
    over_count = 0
    parse_fail = 0
    for i, rec in enumerate(data):
        text = rec.get("text", "")
        try:
            parsed = parse_clevr(text)
            compact = build_compact(parsed)
            n_tok = count_tokens(compact)
            if n_tok > 77:
                over_count += 1
            out_list.append({
                "id": rec["id"],
                "image": rec["image"],
                "text": compact,
            })
        except Exception:
            parse_fail += 1
            out_list.append({
                "id": rec["id"],
                "image": rec["image"],
                "text": text[:500],
            })
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(data)}...")

    print(f"Writing {out_path}...")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_list, f, indent=2, ensure_ascii=False)

    print(f"Done. Records: {len(out_list)}. Over 77 tokens: {over_count}. Parse fails: {parse_fail}")


if __name__ == "__main__":
    main()
