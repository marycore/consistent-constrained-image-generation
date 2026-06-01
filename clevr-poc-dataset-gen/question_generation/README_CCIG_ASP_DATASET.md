# CCIG ASP Dataset Generation

Generates a **constraint-only** JSONL dataset for benchmarking image generators on CCIG complexity levels **L0–L8**.

- **Prompts:** natural-language constraints only (no scene preamble, no object list).
- **ASP in JSONL:** each row has `asp_rules` (constraint lines only) and `asp_code` (full clingo program: background + `object(0..n_objects)` + rules).
- **Validation:** clingo checks `asp_code` directly (or rebuilds it from `asp_rules` + `n_objects` if `asp_code` is missing).

---

## Quick start

```bash
cd clevr-poc-dataset-gen
clingo --version   # must succeed if using --validate_with_clingo
```

### Single-level dataset with ASP consistency checking (recommended for production)

Writes only **SAT** instances (resamples on UNSAT, up to `--max_attempts_per_instance`):

```bash
PYTHONUNBUFFERED=1 python3 question_generation/generate_ccig_asp_dataset.py \
  --mode single \
  --instances_per_class 100 \
  --min_objects 5 \
  --max_objects 9 \
  --validate_with_clingo \
  --clingo_bin clingo \
  --clingo_time_limit 10 \
  --max_attempts_per_instance 100 \
  --output_jsonl question_generation/ccig_asp_dataset.jsonl \
  --seed 42
```

Progress prints to **stderr** (timestamps, level, clingo SAT/UNSAT). Each JSONL row is **flushed** immediately.  
Full run (9×100 instances) can take **hours** — test first with `--class_filter L0 --instances_per_class 5`.

Produces **900 rows** by default (100 × 9 levels L0–L8). Use `--class_filter` or `--instances_map_json` to subset.

### Single-level, fast (no clingo during generation)

```bash
python3 question_generation/generate_ccig_asp_dataset.py \
  --mode single \
  --instances_per_class 100 \
  --output_jsonl question_generation/ccig_asp_dataset.jsonl \
  --seed 42
```

Then audit:

```bash
python3 question_generation/validate_ccig_asp_dataset.py \
  question_generation/ccig_asp_dataset.jsonl \
  --show_unsat \
  --fail_on_unsat
```

---

## Prerequisites

| Requirement | Purpose |
|-------------|---------|
| Python 3.8+ | Generator and validator scripts |
| [clingo](https://potassco.org/clingo/) | SAT/UNSAT checks (`--validate_with_clingo` or `validate_ccig_asp_dataset.py`) |

Install clingo (Ubuntu/Debian):

```bash
sudo apt-get install clingo
# or: conda install -c conda-forge clingo
clingo --version
```

All commands assume working directory:

```bash
cd clevr-poc-dataset-gen
```

---

## File layout

| Role | Path |
|------|------|
| Generator | `question_generation/generate_ccig_asp_dataset.py` |
| Validator | `question_generation/validate_ccig_asp_dataset.py` |
| Template builder | `question_generation/build_ccig_templates.py` |
| Library | `question_generation/ccig_template_lib.py` |
| Complexity index | `question_generation/CLEVR_CCIG_templates/index.json` |
| NL templates | `question_generation/CLEVR_CCIG_templates/L*.json` |
| Combo specs | `question_generation/CLEVR_CCIG_templates/combo_pairs.json` |
| Template docs | `question_generation/CLEVR_CCIG_templates/README.md` |
| ASP rules (one rule per file) | `image_generation/ConstraintTemplates/CCIG_constraint_templates/constraint_templates_L*_{family}.txt` |
| ASP background | `data/general_constraints.txt`, `data/ccig_relationship_axioms.txt` |

Rebuild JSON templates after editing `build_ccig_templates.py`:

```bash
python3 question_generation/build_ccig_templates.py
```

---

## Generator CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `single` | `single` or `combo` |
| `--instances_per_class` | `10` | Rows per level (L0…L8) in single mode |
| `--instances_per_combo` | `10` | Rows per combo spec in combo mode |
| `--instances_map_json` | — | Per-level counts, e.g. `'{"L0":200,"L1":150}'` |
| `--class_filter` | all levels | Subset, e.g. `L0 L1 L2` |
| `--constraint_family` | random | Pin logic variant (use with `--class_filter`) |
| `--constraint_family_map_json` | — | Per-level family, e.g. `'{"L0":"exist","L1":"exist_pair"}'` |
| `--min_objects` / `--max_objects` | `5` / `9` | Sampled `n_objects` for `object(0..N)` |
| `--seed` | `42` | RNG seed |
| `--output_jsonl` | `ccig_asp_dataset.jsonl` | Output path |
| `--validate_with_clingo` | off | Keep only SAT instances |
| `--clingo_bin` | `clingo` | clingo executable |
| `--clingo_time_limit` | `10` | Seconds per clingo call (avoids hangs) |
| `--max_attempts_per_instance` | `50` | Resample cap when validating |
| `--quiet` | off | Level summaries only (less stderr) |
| `--log_clingo_attempts` | off | Log every clingo retry (very verbose) |
| `--unsat_output_jsonl` | `<stem>_unsat.jsonl` | UNSAT sidecar path (with `--validate_with_clingo`) |
| `--no_unsat_sidecar` | off | Skip writing UNSAT attempts |
| `--text_joiner` | ` And ` | Joiner for combo `text` variants |

Combo-only: `--combo_pairs_json`, `--combo_levels`, `--combo_size`.

---

## 1. Single-level dataset

One constraint per record. Random template (and family, unless pinned) within each level.

**All levels, 100 per level, with clingo:**

```bash
python3 question_generation/generate_ccig_asp_dataset.py \
  --mode single \
  --instances_per_class 100 \
  --validate_with_clingo \
  --clingo_bin clingo \
  --max_attempts_per_instance 100 \
  --output_jsonl question_generation/ccig_asp_dataset.jsonl \
  --seed 42
```

**Subset of levels:**

```bash
python3 question_generation/generate_ccig_asp_dataset.py \
  --mode single \
  --class_filter L0 L1 L2 \
  --instances_per_class 50 \
  --validate_with_clingo \
  --output_jsonl question_generation/ccig_asp_dataset_L0_L2.jsonl
```

**Pin constraint family** (names in `index.json` → `constraint_families`):

```bash
# L0 exist-only
python3 question_generation/generate_ccig_asp_dataset.py \
  --mode single \
  --class_filter L0 \
  --constraint_family exist \
  --instances_per_class 100 \
  --validate_with_clingo \
  --output_jsonl question_generation/ccig_asp_dataset_L0_exist.jsonl

# L4 forbid_conditional_pair only
python3 question_generation/generate_ccig_asp_dataset.py \
  --mode single \
  --class_filter L4 \
  --constraint_family forbid_conditional_pair \
  --instances_per_class 100 \
  --validate_with_clingo \
  --output_jsonl question_generation/ccig_asp_dataset_L4_forbid.jsonl
```

**Different family per level:**

```bash
python3 question_generation/generate_ccig_asp_dataset.py \
  --mode single \
  --class_filter L0 L1 \
  --constraint_family_map_json '{"L0":"exist","L1":"exist_pair"}' \
  --instances_per_class 50 \
  --validate_with_clingo \
  --output_jsonl question_generation/ccig_asp_dataset.jsonl
```

**Per-level instance counts:**

```bash
python3 question_generation/generate_ccig_asp_dataset.py \
  --mode single \
  --instances_map_json '{"L0":200,"L1":150,"L2":100}' \
  --validate_with_clingo \
  --output_jsonl question_generation/ccig_asp_dataset.jsonl
```

---

## 2. Combo dataset (2+ constraints)

See `CLEVR_CCIG_templates/README.md` for `combo_pairs.json` schema.

```bash
python3 question_generation/generate_ccig_asp_dataset.py \
  --mode combo \
  --instances_per_combo 50 \
  --validate_with_clingo \
  --clingo_bin clingo \
  --max_attempts_per_instance 200 \
  --output_jsonl question_generation/ccig_asp_dataset_combo.jsonl \
  --seed 42
```

Random combos (no `combo_pairs.json`):

```bash
python3 question_generation/generate_ccig_asp_dataset.py \
  --mode combo \
  --combo_pairs_json /dev/null \
  --combo_levels L0 L1 L2 L3 \
  --combo_size 3 \
  --instances_per_combo 50 \
  --validate_with_clingo \
  --max_attempts_per_instance 300 \
  --output_jsonl question_generation/ccig_asp_dataset_combo_random.jsonl
```

---

## 3. Logical consistency (clingo)

### What “consistent” means

A record is **SAT** if some scene exists that satisfies:

1. `data/general_constraints.txt` (CLEVR ontology)
2. `data/ccig_relationship_axioms.txt` (`hasRelationship/3`)
3. `object(0..n_objects).` from the record’s `n_objects`
4. Every rule in `asp_rules`

**UNSAT** = contradictory constraints. This does **not** judge rendered images.

### When to validate

| Approach | When |
|----------|------|
| `--validate_with_clingo` during generation | Production datasets; only SAT rows written |
| `validate_ccig_asp_dataset.py` after generation | Fast generation + audit; filter UNSAT rows manually |

Validator assembles the full program (same as generation-time check):

```bash
python3 question_generation/validate_ccig_asp_dataset.py \
  question_generation/ccig_asp_dataset.jsonl \
  --show_unsat

python3 question_generation/validate_ccig_asp_dataset.py \
  question_generation/ccig_asp_dataset.jsonl \
  --show_unsat \
  --fail_on_unsat
```

---

## Output schema

One JSON object per line. Example (single, L4):

```json
{
  "id": "scene_000481",
  "mode": "single",
  "complexity_classes": ["L4"],
  "constraint_families": ["forbid_conditional_pair"],
  "text": [
    "It is forbidden for a rubber object in region 3 to be to the right of a cube object in region 2.",
    "No two distinct objects in regions 3 and 2 with a rubber object and a cube object may be right-related."
  ],
  "asp_rules": [
    ":- object(X), object(Y), X != Y, at(X, 3), at(Y, 2), hasProperty(X, material, rubber), hasProperty(Y, shape, cube), hasRelationship(X, Y, right)."
  ],
  "asp_code": "<full program: general_constraints + axioms + object(0..7) + asp_rules>",
  "n_objects": 7,
  "template_files": ["L4_implication_negation_rules.json"],
  "asp_template_files": ["constraint_templates_L4_forbid_conditional_pair.txt"],
  "param_assignments": [
    {"<R1>": "3", "<V1>": "rubber", "<R2>": "2", "<P2>": "shape", "<D1>": "right", "<V2>": "cube"}
  ],
  "property_focus": "material"
}
```

| Field | Meaning |
|-------|---------|
| `id` | Stable instance id (`scene_XXXXXX`) |
| `mode` | `single` or `combo` |
| `text` | All NL paraphrases for this assignment (use `text[0]` as default prompt) |
| `asp_rules` | Instantiated constraint rule(s) only |
| `asp_code` | Full clingo program (background + `object(0..n_objects)` + `asp_rules`) |
| `n_objects` | Domain size (`object(0..n_objects)`); also stored for rebuilding `asp_code` |
| `complexity_classes` | Levels involved, e.g. `["L0"]` or `["L0","L2"]` |
| `constraint_families` | Logic variants, aligned with `asp_rules` |
| `template_files` | Source `L*.json` file(s) |
| `asp_template_files` | Source `constraint_templates_*.txt` file(s) |
| `param_assignments` | One dict per constraint (placeholder → value) |
| `property_focus` / `relation_focus` | Optional (single mode) |
| `combo_size`, `combo_key`, `combo_spec_id` | Combo mode only |
| `clingo_result` | `"SAT"` on main file when using `--validate_with_clingo` |

### UNSAT sidecar (`*_unsat.jsonl`)

With `--validate_with_clingo`, failed resamples are also written to  
`question_generation/ccig_asp_dataset_unsat.jsonl` (same stem + `_unsat`).

Same schema as SAT rows, plus:

| Field | Meaning |
|-------|---------|
| `id` | `unsat_000001`, … |
| `clingo_result` | `"UNSAT"` |
| `for_target_id` | SAT slot, e.g. `scene_000012` |
| `resample_attempt` | Failed try number (1, 2, …) |

Use for benchmarks where models must recognize **impossible** constraint sets.

**Counts (single mode):** SAT rows ≈ sum over levels of `instances_per_class`. UNSAT rows = failed clingo attempts during generation (varies).

---

## Using the dataset

**Image-model prompt:** pick one paraphrase, e.g. `record["text"][0]`.

**ASP evaluation:** use `record["asp_code"]` directly, or rebuild from parts:

```python
asp_code = record["asp_code"]  # preferred when present
# or: record_to_asp_code(record, background)  # from asp_rules + n_objects
```

---

## Notes

- `text` uses natural phrasing where possible (e.g. “rubber object” rather than “object with material rubber”).
- `asp_code` does not fix per-object `at/2` or `hasProperty/3` facts — the scene remains free for the solver/model.
- `asp_rules` is the same constraint slice embedded at the end of `asp_code` (handy without loading background).
- One ASP file = one rule; NL↔ASP alignment via `asp_template_files` + `constraint_families`.
- Legacy JSONL with embedded `asp_code` / `textual_description` still validates if present.
