import json
import math
from pathlib import Path


def split_json_by_items(input_file, output_dir, items_per_file=5000):
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected the JSON file root to be a list.")

    total_items = len(data)
    total_parts = math.ceil(total_items / items_per_file)
    base_name = input_path.stem

    for part_idx in range(total_parts):
        start = part_idx * items_per_file
        end = min(start + items_per_file, total_items)
        chunk = data[start:end]

        out_file = output_path / f"{base_name}_part_{part_idx + 1:03d}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)

        print(f"Saved {out_file} | items {start} to {end - 1} | count={len(chunk)}")

    print(f"\nDone. Split {total_items} items into {total_parts} files.")


if __name__ == "__main__":
    input_file = "dataset_original.json"     # change this
    output_dir = "split_files"    # change this if needed
    items_per_file = 5000         # fixed as requested

    split_json_by_items(input_file, output_dir, items_per_file)