# perception

The core new capability: turn a generated image into a scene graph, then check it
against the original logical constraint with clingo. Per-image pipeline
(`run.py::_perceive_scene` + `run_perception`):

1. **Detect** (`detectors/`) — open-vocabulary bounding boxes for the domain's
   class prompts (CLEVR shapes / COCO categories).
2. **Region + relation** (`regions.py`) — from each bbox's center, compute a
   `r0`-`r3` quadrant region, then derive pairwise `left/right/front/behind`
   relations from a static adjacency table transcribed from
   `ccig-dataset-gen/src/eval_dataset_gen/asp_background/background.lp`. This step
   is plain Python, not clingo — the adjacency is a fixed 2x2 lookup, not worth a
   solver call; clingo is reserved for step 4.
3. **Attributes** (`attributes/`) — one classifier per property, built once per run
   (not per object, to avoid reloading a CLIP checkpoint per crop). CLEVR:
   shape → color (shape-conditioned) → material → size, in that order (see
   `attributes/README.md`). COCO: category comes straight from the detector's
   matched label; only color is classified.
4. **ASP + clingo** (`scene_graph.py`, `common/dataset_gen.py`) — `build_scene_facts()`
   emits `object/1`, `hasProperty/3`, `hasRelationship/3` facts in the exact shape
   `ccig-dataset-gen`'s `format_scene()` parses. The prompt record's own
   `instantiated_rule` (a `:- ...` clingo constraint, used as-is) is appended and
   solved with `solve()` (imported live, not ported — see `common/README.md`).
   `instantiated_rule` strings use free ASP variables (`X`, `Y`, ...) bound over
   `object(X)`, not literal object ids, so they apply unchanged no matter how
   perception numbered the detected objects (0..N-1, assignment order only).
   `predicted_status` (`SAT`/`UNSAT`) is compared against the record's own
   ground-truth `status` for an `agrees_with_dataset` bool in the output.

## Object numbering

`DetectedObject.obj_id` is just the detector's output order (0, 1, 2, ...) — it has
no relationship to any id in the original CLEVR/COCO scene the prompt was generated
from. That's fine because `instantiated_rule` constraints are id-agnostic (see above).

## Why perception detects more than color/shape/material

CLEVR's `domain_clevr.PROPERTIES` also defines `size` (small/large). Despite
`ccig-dataset-gen/src/eval_dataset_gen/domain.py`'s docstring claiming `material` is
excluded from its ASP constraint search space, the code there actually excludes
`size` — and real dataset records do contain `size`-based `instantiated_rule`s. So
attribute classification here is driven generically by `domain_module.PROPERTIES`,
not a hardcoded property list, to cover whatever the dataset actually constrains on.
