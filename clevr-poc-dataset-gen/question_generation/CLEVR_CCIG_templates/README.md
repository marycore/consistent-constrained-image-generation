# CLEVR CCIG Constraint Templates (L0–L8)

POC-compatible JSON templates for **constraint-only** image-generation prompts.

Full pipeline (generation, clingo validation, output schema): [`../README_CCIG_ASP_DATASET.md`](../README_CCIG_ASP_DATASET.md).

---

## Template entry structure

Each `L*.json` file is a JSON **array**. Each entry contains:

| Field | Description |
|-------|-------------|
| `asp_template_file` | Single-rule ASP file (e.g. `constraint_templates_L0_exist.txt`) |
| `constraint_family` | Logic variant within the level (`exist`, `forbid`, `chain2`, …) |
| `property_focus` / `relation_focus` | Which attribute or relation this entry fixes |
| `text` | **Array** of NL paraphrases (constraint only; placeholders like `<R1>`, `<V1>`) |
| `nodes`, `params`, `constraints` | POC-style program metadata |

At generation time, placeholders are sampled and **all** `text[]` lines are instantiated into the dataset `text` field (same semantics, different wording). Each JSONL row also includes `asp_rules` and full `asp_code` (background + domain + rules).

ASP source of truth: `image_generation/ConstraintTemplates/CCIG_constraint_templates/` (one rule per `.txt` file). See `README.txt` in that folder.

---

## Levels and constraint families

| Level | JSON file | `constraint_families` |
|-------|-----------|------------------------|
| L0 | `L0_unary_attribute.json` | `exist`, `forbid` |
| L1 | `L1_single_relational.json` | `exist_pair`, `forbid_pair` |
| L2 | `L2_relational_composition.json` | `chain2` |
| L3 | `L3_conjunctive_relational_binding.json` | `shared_hub` |
| L4 | `L4_implication_negation_rules.json` | `implication`, `forbid_conditional_pair` |
| L5 | `L5_universal_dependency.json` | `universal_witness` |
| L6 | `L6_relational_aggregates.json` | `unary_count`, `relational_count` |
| L7 | `L7_injective_matching.json` | `witness_exist`, `witness_unique` |
| L8 | `L8_global_coupling.json` | `count_coupling`, `all_different` |

Authoritative list: `index.json` → `complexity_levels[].constraint_families`.

---

## Combo benchmarks (`combo_pairs.json`)

Single top-level key: **`combos`**. Each combo merges 2+ constraints.

```json
{
  "combos": [
    {
      "id": "2_L0_exist+L2_chain2",
      "components": [
        {"level": "L0", "constraint_family": "exist"},
        {"level": "L2", "constraint_family": "chain2"}
      ]
    },
    {
      "id": "3_L0+L1+L2",
      "instances": 30,
      "components": [
        {"level": "L0"},
        {"level": "L1"},
        {"level": "L2"}
      ]
    }
  ]
}
```

**Per component**

| Field | Required | Meaning |
|-------|----------|---------|
| `level` | yes | `L0` … `L8` |
| `constraint_family` | no | Pin variant; omit = random family at that level |

**Per combo (optional)**

| Field | Meaning |
|-------|---------|
| `id` | Stored as `combo_spec_id` in JSONL |
| `instances` | Override `--instances_per_combo` for this combo only |

---

## Commands

From `clevr-poc-dataset-gen/`:

### Single-level + ASP consistency (SAT-only output)

```bash
PYTHONUNBUFFERED=1 python3 question_generation/generate_ccig_asp_dataset.py \
  --mode single \
  --instances_per_class 100 \
  --validate_with_clingo \
  --clingo_time_limit 10 \
  --max_attempts_per_instance 100 \
  --output_jsonl question_generation/ccig_asp_dataset.jsonl \
  --seed 42
```

Progress on stderr; use `--class_filter L0 --instances_per_class 5` for a quick test.

With `--validate_with_clingo`, UNSAT resamples are saved to `ccig_asp_dataset_unsat.jsonl` (same format + `clingo_result`, `for_target_id`).

### Single-level, one family (e.g. L0 exist)

```bash
python3 question_generation/generate_ccig_asp_dataset.py \
  --mode single \
  --class_filter L0 \
  --constraint_family exist \
  --instances_per_class 100 \
  --validate_with_clingo \
  --output_jsonl question_generation/ccig_asp_dataset_L0_exist.jsonl
```

### Combo + consistency

```bash
python3 question_generation/generate_ccig_asp_dataset.py \
  --mode combo \
  --instances_per_combo 50 \
  --validate_with_clingo \
  --max_attempts_per_instance 200 \
  --output_jsonl question_generation/ccig_asp_dataset_combo.jsonl
```

### Post-hoc validation (no clingo during generation)

```bash
python3 question_generation/validate_ccig_asp_dataset.py \
  question_generation/ccig_asp_dataset.jsonl \
  --show_unsat \
  --fail_on_unsat
```

---

## Regenerating templates

After editing `question_generation/build_ccig_templates.py`:

```bash
python3 question_generation/build_ccig_templates.py
```

---

## Index

| File | Role |
|------|------|
| `index.json` | Level list, template filenames, `constraint_families` |
| `combo_pairs.json` | Fixed multi-constraint benchmarks |
| `L0_unary_attribute.json` … `L8_global_coupling.json` | NL + metadata per level |
