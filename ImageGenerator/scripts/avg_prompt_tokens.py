#!/usr/bin/env python3
"""Compute average prompt length (in CLIP tokens) for a dataset.json. SD 1.5 uses CLIP max 77 tokens."""

import json
import sys
from pathlib import Path

# Use the same tokenizer as SD 1.5 (CLIP)
from transformers import CLIPTokenizer

def main():
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "configs/finetune_dataset/dataset.json"
    path = Path(dataset_path)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Dataset must be a JSON array")
        sys.exit(1)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    lengths = []
    for rec in data:
        text = rec.get("text", "")
        tok = tokenizer(text, return_tensors="pt", truncation=False)
        lengths.append(tok.input_ids.shape[1])

    n = len(lengths)
    avg = sum(lengths) / n
    min_len = min(lengths)
    max_len = max(lengths)
    over_77 = sum(1 for L in lengths if L > 77)

    print(f"Dataset: {path}")
    print(f"Number of prompts: {n}")
    print(f"Token length (CLIP, same as SD 1.5):")
    print(f"  Average: {avg:.1f}")
    print(f"  Min:     {min_len}")
    print(f"  Max:     {max_len}")
    print(f"  Prompts over 77 tokens (truncated at train time): {over_77} ({100*over_77/n:.1f}%)")

if __name__ == "__main__":
    main()
