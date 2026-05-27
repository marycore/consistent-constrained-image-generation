# CCIG ASP Dataset Generation (Minimal JSONL)

This guide generates a minimal JSONL dataset from ASP constraint templates in:

- `image_generation/ConstraintTemplates/CCIG_constraint_templates/constraint_templates_L*.txt`

Complexity classes (`L0` ... `L8`) are read from:

- `question_generation/CLEVR_CCIG_templates/index.json`

Output fields:

- `id`
- `complexity_class` (`L0` ... `L8`)
- `asp_code`
- `textual_description` (prompt format: scene specification + constraint)

Script:

- `question_generation/generate_ccig_asp_dataset.py`

---

## 1) Go to project root

```bash
cd /home/marjan.alirezaie/myworks/code/miscellaneous/ccig/CCIG_Eval/clevr-poc-dataset-gen
```

## 2) Same number of instances for all complexity classes

```bash
python3 question_generation/generate_ccig_asp_dataset.py \
  --index_json question_generation/CLEVR_CCIG_templates/index.json \
  --constraint_templates_dir image_generation/ConstraintTemplates/CCIG_constraint_templates \
  --output_jsonl question_generation/ccig_asp_dataset.jsonl \
  --instances_per_class 100 \
  --min_objects 5 \
  --max_objects 9 \
  --seed 42
```

## 2b) Optional: validate each generated instance with clingo

This retries sampling until the generated ASP code is satisfiable (or until max attempts).

```bash
python3 question_generation/generate_ccig_asp_dataset.py \
  --index_json question_generation/CLEVR_CCIG_templates/index.json \
  --constraint_templates_dir image_generation/ConstraintTemplates/CCIG_constraint_templates \
  --output_jsonl question_generation/ccig_asp_dataset_validated.jsonl \
  --instances_per_class 100 \
  --min_objects 5 \
  --max_objects 9 \
  --validate_with_clingo \
  --clingo_bin clingo \
  --max_attempts_per_instance 100 \
  --seed 42
```

## 3) Custom number of instances per complexity class

```bash
python3 question_generation/generate_ccig_asp_dataset.py \
  --index_json question_generation/CLEVR_CCIG_templates/index.json \
  --constraint_templates_dir image_generation/ConstraintTemplates/CCIG_constraint_templates \
  --output_jsonl question_generation/ccig_asp_dataset.jsonl \
  --instances_per_class 50 \
  --min_objects 5 \
  --max_objects 9 \
  --instances_map_json '{"L0":200,"L1":150,"L2":120,"L3":100,"L4":80,"L5":60,"L6":40,"L7":25,"L8":10}' \
  --seed 42
```

## 4) Generate only selected classes

```bash
python3 question_generation/generate_ccig_asp_dataset.py \
  --index_json question_generation/CLEVR_CCIG_templates/index.json \
  --constraint_templates_dir image_generation/ConstraintTemplates/CCIG_constraint_templates \
  --output_jsonl question_generation/ccig_asp_dataset_L0_L1.jsonl \
  --instances_per_class 100 \
  --min_objects 5 \
  --max_objects 9 \
  --class_filter L0 L1 \
  --seed 42
```

---

## Output format example (one line in JSONL)

```json
{"id":"scene_000001","complexity_class":"L0","asp_code":"object(0..5).\nat(0,1).\nhasProperty(0,color,red).\n...\n:- not 1 { X : object(X), at(X, 2), hasProperty(X, color, green) }.","textual_description":"Create a scene with 4 regions and 6 objects. The objects are: o0(...), o1(...), ... Satisfy this constraint: This enforces that at least one object in region 2 has property color=green."}
```

## Notes

- `--validate_with_clingo` is optional and can be slower.
- If clingo is not in PATH, set `--clingo_bin /full/path/to/clingo`.
- Increase `--max_attempts_per_instance` for stricter/rare constraints.

