import json

file1 = "dataset_clip77.json"    # contains the original "text"
file2 = "dataset.json"   # will receive the new "pred"
output_file = "dataset.json"

print(f"Loading {file1}...")
with open(file1, "r", encoding="utf-8") as f:
    data1 = json.load(f)
    print(f"Loaded {len(data1)} items from {file1}")

with open(file2, "r", encoding="utf-8") as f:
    data2 = json.load(f)

# Build lookup from first file: id -> text
text_by_id = {item["id"]: item.get("text", "") for item in data1}

for item in data2:
    item_id = item.get("id")
    item["pred"] = text_by_id.get(item_id, "")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data2, f, indent=2, ensure_ascii=False)

print(f"Done. Saved to {output_file}")