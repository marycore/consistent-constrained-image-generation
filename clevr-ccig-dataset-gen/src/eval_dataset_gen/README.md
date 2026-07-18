# Eval Dataset Generation Pipeline

Generates a JSONL dataset of ASP constraint instances with natural language descriptions.
Each record pairs an instantiated ASP rule with short/medium/long NL text and clingo solver output.
This is the evaluation dataset — see [../finetune_dataset_gen/README.md](../finetune_dataset_gen/README.md)
for the sibling pipeline that captions existing images instead.

## Pipeline steps

1. **Load** — read an ASP template (`.txt`) from `constraint_templates/`
2. **Instantiate** — randomly assign primed placeholders (`P1'`, `V1'`, `D1'`, `N'`, …) respecting `!=` constraints in the rule
3. **Verbalize** — produce short, medium, and long NL descriptions for the constraint
4. **Solve** — run clingo on the full ASP program (background + instantiated rule), record SAT/UNSAT and grounded models
5. **Write** — append a JSON record to the output `.jsonl` file

## Files

| File | Role |
|------|------|
| `domain.py` | Property/value/direction domains (derives from `../common/domain.py`, excluding `material`); background ASP program generator |
| `instantiate.py` | Template loading, placeholder extraction, valid assignment sampling |
| `../common/verbalize.py` | NL verbalization for all 9 constraint classes (C1–C9) — shared with `finetune_dataset_gen`, which grounds it against real scenes instead of instantiating it randomly |
| `solve.py` | Clingo wrapper; scene formatter for grounded answer sets |
| `run.py` | CLI entry point orchestrating the full pipeline |

## Run

```bash
cd clevr-ccig-dataset-gen

# Default: 10 random samples per template, 4-object scenes
python -m src.eval_dataset_gen.run

# Custom number of samples and output path
python -m src.eval_dataset_gen.run --samples 20 --output ../my_dataset.jsonl

# Restrict to specific constraint classes
python -m src.eval_dataset_gen.run --classes C1 C3 C8

# Skip clingo solving (NL verbalization only, no SAT/UNSAT)
python -m src.eval_dataset_gen.run --no_solve

```

Outputs two files derived from `--output` (default: `../data/ccig_eval_dataset.jsonl`, i.e. the
repo-root `data/` folder):
- `ccig_eval_dataset_SAT.jsonl`
- `ccig_eval_dataset_UNSAT.jsonl`

## Output format

Each line in both files is one JSON record:

```json
{
  "id": "C1-1prop-a3f9b2-000",
  "complexity_class": "C1",
  "constraint_family": "1prop",
  "text": {
    "short": "A red object is in the scene.",
    "medium": "The scene contains at least one red object.",
    "long": "..."
  },
  "asp_code": "...",
  "asp_template_file": "C1_1prop.txt",
  "status": "SAT"
}
```
