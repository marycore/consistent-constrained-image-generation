# CLEVR CCIG Constraint Templates (C1-C9)

POC-compatible JSON templates for **constraint-only** image-generation prompts.

Full pipeline (generation, clingo validation, output schema): [`../README_CCIG_ASP_DATASET.md`](../README_CCIG_ASP_DATASET.md).

---

## Template entry structure

Each `C*.json` file is a JSON **array**. Each entry contains:

| Field | Description |
|-------|-------------|
| `asp_template_file` | ASP rule file (e.g. `constraint_templates_C1_exist.txt`) |
| `constraint_family` | Logic variant within the class (`exist`, `forbid`, `chain2`, …) |
| `property_focus` / `relation_focus` | Which attribute or relation this entry fixes |
| `text` | **Array** of declarative scene-description paraphrases (placeholders like `<R1>`, `<V1>`), usable directly as image-generation prompts once placeholders are filled |
| `params` | Placeholder types/names sampled at instantiation time |

At generation time, placeholders are sampled and **all** `text[]` lines are instantiated into the dataset `text` field (same semantics, different wording), then passed through `naturalize_constraint_text` (in `ccig_template_lib.py`) which converts relation placeholders (`<D1>` etc.) into natural prepositional phrases ("behind", "to the left of", "in front of") and fixes singular/plural agreement for counts. Each JSONL row also includes `asp_rules` and full `asp_code` (background + domain + rules).

ASP source of truth: `image_generation/ConstraintTemplates/CCIG_constraint_templates/` (one rule block — optional helper rules plus exactly one integrity constraint — per `.txt` file). See `README.txt` in that folder for the formal C1-C9 definitions and the L-tier-to-C-class migration table.

**Note on dropped fields**: legacy `L0`-`L7` templates also carried `nodes` and `constraints` fields (CLEVR functional-program metadata). These are not read anywhere downstream of JSON load (verified against `ccig_template_lib.py` and `generate_ccig_asp_dataset.py`) and are intentionally omitted from the `C*.json` files.

---

## Classes and constraint families

| Class | JSON file | `constraint_families` |
|-------|-----------|------------------------|
| C1 | `C1_existential_object.json` | `exist`, `forbid` |
| C2 | `C2_universal_object.json` | `universal` |
| C3 | `C3_conditional_object.json` | `conditional`, `conditional_negated` |
| C4 | `C4_existential_subgraph.json` | `exist_pair`, `forbid_pair`, `chain2`, `chain3`, `shared_hub` |
| C5 | `C5_conditional_subgraph.json` | `pair_conditional` |
| C6 | `C6_existential_universal.json` | `witness_universal` |
| C7 | `C7_universal_existential.json` | `implication`, `universal_witness`, `unique_witness` |
| C8 | `C8_cardinality.json` | `unary_count`, `relational_count`, `all_different` |
| C9 | `C9_aggregate_comparison.json` | `count_coupling` |

Authoritative list: `index.json` → `complexity_levels[].constraint_families`.

---

## Combo benchmarks (`combo_pairs.json`)

Single top-level key: **`combos`**. Each combo merges 2+ constraints.

```json
{
  "combos": [
    {
      "id": "2_C1_exist+C4_chain2",
      "components": [
        {"level": "C1", "constraint_family": "exist"},
        {"level": "C4", "constraint_family": "chain2"}
      ]
    },
    {
      "id": "3_C1+C4+C7",
      "instances": 30,
      "components": [
        {"level": "C1"},
        {"level": "C4"},
        {"level": "C7"}
      ]
    }
  ]
}
```

**Per component**

| Field | Required | Meaning |
|-------|----------|---------|
| `level` | yes | `C1` … `C9` |
| `constraint_family` | no | Pin variant; omit = random family at that class |

**Per combo (optional)**

| Field | Meaning |
|-------|---------|
| `id` | Stored as `combo_spec_id` in JSONL |
| `instances` | Override `--instances_per_combo` for this combo only |

---

## Commands

From `clevr-poc-dataset-gen/`:

### Single-class + ASP consistency (SAT-only output)

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

Progress on stderr; use `--class_filter C1 --instances_per_class 5` for a quick test.

With `--validate_with_clingo`, UNSAT resamples are saved to `ccig_asp_dataset_unsat.jsonl` (same format + `clingo_result`, `for_target_id`).

`clingo_result` in sidecar rows distinguishes true logical failures and syntax issues:
- `UNSAT` for true unsatisfiable constraints
- `ASP_PARSE_ERROR` for parse/syntax errors

### Single-class, balanced families per class

```bash
python3 question_generation/generate_ccig_asp_dataset.py \
  --mode single \
  --instances_per_class 100 \
  --family_sampling equal \
  --validate_with_clingo \
  --output_jsonl question_generation/ccig_asp_dataset.jsonl \
  --seed 42
```

### Single-class, one family (e.g. C1 exist)

```bash
python3 question_generation/generate_ccig_asp_dataset.py \
  --mode single \
  --class_filter C1 \
  --constraint_family exist \
  --instances_per_class 100 \
  --validate_with_clingo \
  --output_jsonl question_generation/ccig_asp_dataset_C1_exist.jsonl
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
| `index.json` | Class list, template filenames, `constraint_families` |
| `combo_pairs.json` | Fixed multi-constraint benchmarks |
| `C1_existential_object.json` … `C9_aggregate_comparison.json` | NL + metadata per class |
