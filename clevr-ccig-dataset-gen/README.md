# clevr-ccig-dataset-gen

Generates CLEVR-CCIG datasets — anything built from the shared constraint-class vocabulary
(C1–C9, defined in `src/eval_dataset_gen/constraint_templates/`), for two different purposes:

| Pipeline | Purpose | Docs |
|---|---|---|
| `src/eval_dataset_gen/` | The evaluation dataset: single-constraint ASP instances, randomly instantiated and clingo-solved for SAT/UNSAT. | [src/eval_dataset_gen/README.md](src/eval_dataset_gen/README.md) |
| `src/finetune_dataset_gen/` | The fine-tuning dataset: captions for the *existing* CLEVR-CCIG images, built by grounding true C1–C9 constraints against each image's actual scene. | [src/finetune_dataset_gen/README.md](src/finetune_dataset_gen/README.md) |

Both pipelines are separate (different inputs, different outputs, run independently) but share
`src/common/`:
- `domain.py` — single source of truth for property/value domains (colors, shapes, sizes,
  materials, regions). `eval_dataset_gen` layers one override on top (excludes `material`);
  `finetune_dataset_gen` uses the full domain.
- `verbalize.py` — NL phrasing for all 9 constraint classes. `eval_dataset_gen` calls it on
  randomly-instantiated constraints; `finetune_dataset_gen` calls the exact same functions on
  constraints it finds to be true of a given image. This is what keeps the two datasets'
  vocabulary from drifting apart.

Datasets themselves (inputs and outputs) live in the repo-root `data/` folder, not inside this
package — see each pipeline's README for exact paths.
