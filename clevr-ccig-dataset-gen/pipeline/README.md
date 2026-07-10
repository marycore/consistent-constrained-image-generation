# CCIG Dataset Generation Pipeline

Generates a JSONL dataset of ASP constraint instances with natural language descriptions.
Each record pairs an instantiated ASP rule with short/medium/long NL text and clingo solver output.

## Pipeline steps

1. **Load** — read an ASP template (`.txt`) from `ConstraintTemplates/`
2. **Instantiate** — randomly assign primed placeholders (`P1'`, `V1'`, `D1'`, `N'`, …) respecting `!=` constraints in the rule
3. **Verbalize** — produce short, medium, and long NL descriptions for the constraint
4. **Solve** — run clingo on the full ASP program (background + instantiated rule), record SAT/UNSAT and grounded models
5. **Write** — append a JSON record to the output `.jsonl` file

## Files

| File | Role |
|------|------|
| `domain.py` | Property/value/direction domains; background ASP program generator |
| `instantiate.py` | Template loading, placeholder extraction, valid assignment sampling |
| `verbalize.py` | NL verbalization for all 9 constraint classes (C1–C9) |
| `solve.py` | Clingo wrapper; scene formatter for grounded answer sets |
| `run.py` | CLI entry point orchestrating the full pipeline |

## Run

```bash
cd clevr-ccig-dataset-gen/pipeline

# Default: 10 random samples per template, 4-object scenes
python run.py

# Custom number of samples and output path
python run.py --samples 20 --output ../my_dataset.jsonl

# Restrict to specific constraint classes
python run.py --classes C1 C3 C8

# Skip clingo solving (NL verbalization only, no SAT/UNSAT)
python run.py --no_solve

# Enumerate all valid assignments exhaustively instead of random sampling
python run.py --mode exhaustive
```

Outputs two files derived from `--output` (default: `../ccig_dataset.jsonl`):
- `ccig_dataset_SAT.jsonl`
- `ccig_dataset_UNSAT.jsonl`

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
