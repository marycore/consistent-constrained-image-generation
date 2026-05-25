import json
from pathlib import Path


def merge_json_files(input_files, output_file):
    merged_data = []

    for file_path in input_files:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"{file_path} does not contain a JSON list.")

        merged_data.extend(data)
        print(f"Added {len(data)} items from {file_path}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"\nDone.")
    print(f"Saved {len(merged_data)} total items to {output_file}")


if __name__ == "__main__":
    input_files = [
        "split_files/dataset_part_001_with_general_text.json",
        "split_files/dataset_part_002_with_general_text.json",
        "split_files/dataset_part_003_with_general_text.json",
        "split_files/dataset_part_004_with_general_text.json",
        "split_files/dataset_part_005_with_general_text.json",
    ]

    output_file = "dataset.json"

    merge_json_files(input_files, output_file)