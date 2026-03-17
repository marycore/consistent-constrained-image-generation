#!/usr/bin/env python3
"""
Convert CLEVR-style prompts to compact form (for use with models fine-tuned on dataset_clip77.json).

Conversion shortens by re-encoding (same info, denser text); it does not truncate or drop
content unless the scene is so large that even the most compressed form exceeds 77 tokens,
in which case only tail relation phrases may be omitted (all objects are always kept).

Usage:
  # Single prompt from argument
  python3 scripts/prompt_to_compact.py "Generate a CLEVR style image... There are 3 objects..."

  # Single prompt from stdin
  echo "Generate a CLEVR style image..." | python3 scripts/prompt_to_compact.py

  # JSONL prompt file (e.g. for run_finetuned): convert "prompt" or "text" field, write new JSONL
  python3 scripts/prompt_to_compact.py --jsonl configs/prompts_finetuned_test.jsonl -o configs/prompts_finetuned_test_compact.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root so we can import from src.common
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.common.clevr_compact import clevr_long_to_compact, count_tokens


def main():
    ap = argparse.ArgumentParser(description="Convert CLEVR long prompts to compact (CLIP 77-token) style.")
    ap.add_argument("prompt", nargs="?", default=None, help="Single prompt string (or read from stdin)")
    ap.add_argument("--jsonl", type=Path, help="Input JSONL file (each line has 'prompt' or 'text')")
    ap.add_argument("-o", "--output", type=Path, help="Output JSONL file (default: stdout for single prompt)")
    ap.add_argument("--max-tokens", type=int, default=77, help="Max tokens (default 77)")
    ap.add_argument("--field", default="prompt", choices=("prompt", "text"), help="Field to convert in JSONL (default: prompt)")
    args = ap.parse_args()

    if args.jsonl:
        if not args.jsonl.exists():
            print(f"Error: not found {args.jsonl}", file=sys.stderr)
            sys.exit(1)
        out_path = args.output or args.jsonl.parent / f"{args.jsonl.stem}_compact.jsonl"
        out_lines = []
        with args.jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                raw = rec.get(args.field) or rec.get("text") or rec.get("prompt") or ""
                compact = clevr_long_to_compact(raw, max_tokens=args.max_tokens)
                rec[args.field] = compact
                # Keep "text" in sync if present (some prompt files use "text")
                if "text" in rec:
                    rec["text"] = compact
                out_lines.append(json.dumps(rec, ensure_ascii=False))
        with out_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
        print(f"Wrote {len(out_lines)} lines to {out_path}", file=sys.stderr)
        return

    # Single prompt
    if args.prompt:
        text = args.prompt
    else:
        text = sys.stdin.read().strip()
    if not text:
        print("No prompt given (use argument or stdin)", file=sys.stderr)
        sys.exit(1)
    compact = clevr_long_to_compact(text, max_tokens=args.max_tokens)
    n = count_tokens(compact)
    if args.output:
        args.output.write_text(compact, encoding="utf-8")
        print(f"Wrote compact prompt ({n} tokens) to {args.output}", file=sys.stderr)
    else:
        print(compact)
        print(f"  # {n} tokens", file=sys.stderr)


if __name__ == "__main__":
    main()
