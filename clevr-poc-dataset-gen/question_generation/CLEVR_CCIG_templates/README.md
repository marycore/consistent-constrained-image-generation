# CLEVR CCIG Constraint Templates (C1–C9)

JSON template files that drive constraint-only image-generation prompt creation.
One file per constraint class (`C1`–`C9`), plus `index.json` and `combo_pairs.json`.

Full pipeline (generation, clingo validation, output schema): [`../README_CCIG_ASP_DATASET.md`](../README_CCIG_ASP_DATASET.md).

---

## How a JSON entry is turned into a prompt (step by step)

Each entry in a `C*.json` file is a **template**, not a finished sentence. The library
(`ccig_template_lib.py`) turns it into a fully-worded image-generation prompt at
dataset-generation time. Here is exactly what happens, in order.

### Step 1 — Pick a template entry

The generator picks one entry from a `C*.json` file, either randomly or by filtering on
`constraint_family` and/or `property_focus`. For example, to produce a C1 constraint
about regions it picks the entry with `"constraint_family": "1prop"` and
`"property_focus": "region"`.

```json
{
  "asp_template_file": "C1_1prop.txt",
  "constraint_family": "1prop",
  "text": [
    "An object is in <V1>.",
    "The scene contains at least one object located in <V1>.",
    "Among all the objects placed in the scene, at least one of them must be situated in <V1>; no restriction is placed on the other objects."
  ],
  "params": [
    { "type": "Region", "name": "<V1>" }
  ],
  "property_focus": "region"
}
```

### Step 2 — Sample concrete values for every placeholder (`sample_param_assignment`)

The library reads the `"params"` list and draws a random concrete value for each
placeholder according to its `"type"`:

| Param type | Samples from | Example values |
|------------|-------------|----------------|
| `Region`   | `REGION_VALUES` | `region_1`, `region_2`, `region_3`, `region_4` |
| `Shape`    | `SHAPE_VALUES`  | `cube`, `cylinder`, `sphere`, `cone` |
| `Size`     | `SIZE_VALUES`   | `small`, `large`, `medium` |
| `Relation` | `REL_VALUES`    | `left`, `right`, `front`, `behind` |
| `Count`    | `COUNT_VALUES`  | `1`, `2`, `3` |
| `Property` | `PROP_VALUES`   | `shape`, `size`, `region` |
| `Value`    | domain of its paired `Property` param | e.g. if `<P2>` = `"shape"` → `cube` |

If the entry has `"property_focus": "region"`, the library already knows that `<V1>`
must be a region value, so it samples directly from `REGION_VALUES` for a
`Region`-typed param.

**Result** (a `param_assignment` dict):
```
{ "<V1>": "region_2" }
```

### Step 3 — Map to ASP primed placeholders (`build_asp_assignment`)

The ASP rule files use primed placeholder names (`V1'`, `P1'`, `D1'`, `N'`, …). The
library converts the JSON param names to their ASP equivalents:

| JSON param name | ASP placeholder |
|-----------------|----------------|
| `<V1>`          | `V1'`          |
| `<V2>`          | `V2'`          |
| `<V3>`          | `V3'`          |
| `<V4>`          | `V4'`          |
| `<D1>`          | `D1'`          |
| `<D2>`          | `D2'`          |
| `<D3>`          | `D3'`          |
| `<N>`           | `N'`           |
| `<P2>`          | `P2'`          |

Additionally, `"property_focus"` is written directly into `P1'` (the ASP placeholder
for the first property axis), and `"relation_focus"` is written into `D1'`.

Any ASP placeholder that still has no value at this point is sampled from its domain
(e.g. an unresolved `P2'` picks a random property from `PROP_VALUES`).

**Result** (an `asp_assignment` dict):
```
{ "P1'": "region", "V1'": "region_2" }
```

### Step 4 — Fill the text placeholders (`instantiate_texts`)

Each of the three strings in `"text"` has its `<...>` placeholders replaced by the
concrete values from step 2:

```
"An object is in <V1>."   →   "An object is in region_2."
"The scene contains at least one object located in <V1>."
  →  "The scene contains at least one object located in region_2."
"Among all the objects placed in the scene, at least one of them must be situated in <V1>; ..."
  →  "Among all the objects placed in the scene, at least one of them must be situated in region_2; ..."
```

### Step 5 — Post-process the text (`naturalize_constraint_text`)

The library runs a set of regex substitutions to make the text more natural:

- **Relation phrases**: raw direction words become proper English prepositional phrases.  
  `"left of"` → `"to the left of"`, `"front of"` → `"in front of"`, etc.
- **Singular/plural agreement**: `"Exactly 1 objects are …"` → `"Exactly 1 object is …"`.
- **Property-value collapsing**: formal constructions like `"with shape cube"` → `"cube"`;
  `"must have size large"` → `"must be large"`.

The three resulting strings are stored as `texts[0]` (short), `texts[1]` (medium),
`texts[2]` (long) in the output record.

### Step 6 — Fill the ASP rule (`apply_assignment`)

The raw ASP rule from `C1_1prop.txt` contains primed placeholders:

```prolog
:- not 1 { hasProperty(X, P1', V1') : object(X) }.
```

The library substitutes the `asp_assignment` dict (step 3) to produce the concrete rule:

```prolog
:- not 1 { hasProperty(X, region, region_2) : object(X) }.
```

This rule is stored in `asp_rule` in the output record.

### Step 7 — Build the full ASP program

The concrete rule is combined with a background theory (object domain, property
axioms) and `object(0..N)` to form a complete clingo program. This program is
optionally validated: clingo must return `SATISFIABLE` (the constraint can be met).
UNSAT instances are discarded (or saved to a sidecar `.jsonl`).

### Step 8 — Write the JSONL output record

Each output row has at minimum:

```json
{
  "id": "C1-0042",
  "complexity_class": "C1",
  "constraint_family": "1prop",
  "property_focus": "region",
  "param_assignment": { "<V1>": "region_2" },
  "asp_assignment":   { "P1'": "region", "V1'": "region_2" },
  "asp_rule":  ":- not 1 { hasProperty(X, region, region_2) : object(X) }.",
  "texts": [
    "An object is in region_2.",
    "The scene contains at least one object located in region_2.",
    "Among all the objects placed in the scene, at least one of them must be situated in region_2; no restriction is placed on the other objects."
  ]
}
```

---

## Template entry fields reference

| Field | Required | Description |
|-------|----------|-------------|
| `asp_template_file` | yes | Filename in `image_generation/ConstraintTemplates/CCIG_constraint_templates/`. Contains exactly one ASP rule block. |
| `constraint_family` | yes | Logic variant within the class (e.g. `1prop`, `2prop_neg`, `relational_exact`). Must match a family listed in `index.json`. |
| `property_focus` | no | Which property axis (`shape`, `size`, `region`) this entry anchors. Written into `P1'` in the ASP rule and used to constrain value sampling. |
| `relation_focus` | no | Which direction (`left`, `right`, `front`, `behind`) this entry fixes. Written into `D1'`. Used by entries where the relation is hard-coded in the text rather than sampled. |
| `text` | yes | **Array of exactly 3 strings**, ordered short → medium → long. Placeholders use `<Name>` syntax. Write these as natural scene descriptions — they become image-generation prompts. |
| `params` | yes | Ordered list of placeholders the library must sample. Each has a `"type"` (see table above) and a `"name"` like `<V1>`. |

---

## Placeholder naming conventions

In the `"text"` strings and `"params"` list, placeholders follow this convention:

| Placeholder | Meaning |
|-------------|---------|
| `<V1>`, `<V2>`, `<V3>`, `<V4>` | Property **values** (e.g. `region_2`, `cube`, `large`) |
| `<P2>` | Property **axis** for the second value (e.g. `shape`, `size`) |
| `<D1>`, `<D2>`, `<D3>` | Spatial **directions** (`left`, `right`, `front`, `behind`) |
| `<N>` | Integer **count** (`1`, `2`, `3`) |

`<V1>` is always the primary value (its property axis is given by `property_focus`).
`<V2>` always pairs with `<P2>`: if `<P2>` = `"size"`, then `<V2>` is sampled from
`SIZE_VALUES`. `<V3>` and `<V4>` follow the same pattern when a third/fourth property
is needed (multi-property templates).

---

## Property space

The three active property axes and their value sets:

| Property axis (`<P>`) | Values |
|----------------------|--------|
| `color` | `gray`, `red`, `blue`, `green`, `brown`, `purple`, `cyan`, `yellow` |
| `shape` | `cube`, `cylinder`, `sphere`, `cone` |
| `size`  | `small`, `large`, `medium` |
| `region`| `region_1`, `region_2`, `region_3`, `region_4` |

In ASP: `hasProperty(X, region, region_1)` — region is a first-class object property,
not a spatial predicate.

---

## Writing and editing text entries

Each `"text"` array must have exactly three entries. Write them as image-generation
scene descriptions — natural, declarative sentences, not logical formulas.

**For shape/size entries** the value fills an adjective slot:
```
"A <V1> object is in the scene."   →   "A cube object is in the scene."
```

**For region entries** the value fills a location slot — use `"in <V1>"` phrasing, not
`"a <V1> object"` (region values are not adjectives):
```
"An object is in <V1>."   →   "An object is in region_2."
```

**Direction placeholders** (`<D1>`, `<D2>`) are substituted as direction words (`left`,
`behind`, etc.) and then `naturalize_constraint_text` wraps them in prepositional
phrases automatically, so you can write either form:
```
"... is <D2> of ..."          works (naturalized to "to the left of", etc.)
"... is to the left of ..."   also works (hard-coded, bypasses naturalization)
```

---

## Classes and constraint families

| Class | JSON file | `constraint_families` |
|-------|-----------|----------------------|
| C1 | `C1_existential_object.json` | `1prop`, `2prop`, `4prop`, `1prop_neg`, `1prop_2val_neg`, `2prop_neg`, `2prop_mix_neg` |
| C2 | `C2_universal_object.json` | `1prop`, `1prop_neg`, `2prop`, `2prop_neg`, `2prop_mix_neg` |
| C3 | `C3_conditional_object.json` | `1propA_1propC`, `1propA_1prop_neg`, `1propA_neg_1propC`, `2propA_1propC`, `1propA_2propC` |
| C4 | `C4_existential_subgraph.json` | `2hop`, `3hop`, `shared_hub`, `prop_2hop`, `prop_shared_hub` |
| C5 | `C5_conditional_subgraph.json` | `pair_propA_relC`, `pair_propRelA_RelC`, `pair_propRelA_propC`, `pair_relA_propC`, `triple_propA_RelC`, `triple_propRelA_relC`, `triple_propRelA_propC` |
| C6 | `C6_existential_universal.json` | `witness_1prop`, `witness_1prop_neg`, `witness_2prop` |
| C7 | `C7_universal_existential.json` | `propRel`, `propRel_neg`, `propRel_propRel`, `exact`, `exact_neg` |
| C8 | `C8_cardinality.json` | `1prop_exact`, `1prop_atleast`, `1prop_atmost`, `1prop_exact_neg`, `2prop_exact`, `relational_exact`, `relational_atleast`, `relational_atmost` |
| C9 | `C9_aggregate_comparison.json` | `1prop`, `2prop`, `2prop_mix` |

Authoritative list: `index.json → complexity_levels[].constraint_families`.

---

## Combo benchmarks (`combo_pairs.json`)

Each combo merges 2+ constraints from different classes into a single scene description
and a single ASP program. The library generates each component independently (steps 1–6
above), then concatenates the ASP rules and joins the text strings with `" And "`.

```json
{
  "combos": [
    {
      "id": "C1_1prop+C8_relational_exact",
      "components": [
        { "level": "C1", "constraint_family": "1prop" },
        { "level": "C8", "constraint_family": "relational_exact" }
      ]
    }
  ]
}
```

| Component field | Required | Meaning |
|-----------------|----------|---------|
| `level` | yes | `C1` … `C9` |
| `constraint_family` | no | Pin to a specific family; omit = random family at that class |
| `property_focus` | no | Pin to a specific property axis |

---

## Generation commands

From `clevr-poc-dataset-gen/`:

### Single-class (all families, all property axes)

```bash
python3 question_generation/generate_ccig_asp_dataset.py \
  --mode single \
  --instances_per_class 100 \
  --validate_with_clingo \
  --output_jsonl question_generation/ccig_asp_dataset.jsonl \
  --seed 42
```

### Single-class, balanced across families

```bash
python3 question_generation/generate_ccig_asp_dataset.py \
  --mode single \
  --instances_per_class 100 \
  --family_sampling equal \
  --validate_with_clingo \
  --output_jsonl question_generation/ccig_asp_dataset.jsonl \
  --seed 42
```

### One class, one family (quick test)

```bash
python3 question_generation/generate_ccig_asp_dataset.py \
  --mode single \
  --class_filter C1 \
  --constraint_family 1prop \
  --instances_per_class 5 \
  --validate_with_clingo \
  --output_jsonl question_generation/ccig_asp_dataset_C1_1prop.jsonl
```

### Combo constraints

```bash
python3 question_generation/generate_ccig_asp_dataset.py \
  --mode combo \
  --instances_per_combo 50 \
  --validate_with_clingo \
  --output_jsonl question_generation/ccig_asp_dataset_combo.jsonl
```

---

## File index

| File | Role |
|------|------|
| `index.json` | Class list, template filenames, valid `constraint_families` per class |
| `combo_pairs.json` | Multi-constraint combo specifications |
| `C1_existential_object.json` | Templates for C1 (∃ at least one object with property P) |
| `C2_universal_object.json` | Templates for C2 (∀ objects have property P) |
| `C3_conditional_object.json` | Templates for C3 (if P then Q, conditional) |
| `C4_existential_subgraph.json` | Templates for C4 (spatial chain / hub subgraph) |
| `C5_conditional_subgraph.json` | Templates for C5 (conditional spatial chain) |
| `C6_existential_universal.json` | Templates for C6 (∃ object P, ∀ of Q) |
| `C7_universal_existential.json` | Templates for C7 (∀ P objects have ∃ Q neighbor) |
| `C8_cardinality.json` | Templates for C8 (exact / at-least / at-most count) |
| `C9_aggregate_comparison.json` | Templates for C9 (count equality between two groups) |
