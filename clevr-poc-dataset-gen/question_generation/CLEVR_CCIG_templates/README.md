# CLEVR CCIG Constraint Templates (L0-L8)

This folder contains one constraint template file per complexity level defined in Section 5
of `constraint_image_generation.pdf`.

Levels:
- `L0_unary_attribute.json` (F1)
- `L1_single_relational.json` (F2)
- `L2_relational_composition.json` (F3)
- `L3_conjunctive_relational_binding.json` (F3)
- `L4_implication_negation_rules.json` (F4)
- `L5_universal_dependency.json` (F5)
- `L6_relational_aggregates.json` (F6)
- `L7_injective_matching.json` (F7)
- `L8_global_coupling.json` (F8)

Each `L*.json` file follows the same template object structure used in
`question_generation/CLEVR_POC_templates`:
- top-level JSON array
- each item has `text`, `nodes`, `params`, `constraints`

Notes:
- Parameter placeholders follow CLEVR style (for example `<Z>`, `<C>`, `<M>`, `<S>`, `<R>`).
- The logical complexity mapping (L0-L8) is encoded by file name and by the relational / aggregate
  program pattern represented in each template's `nodes`.
