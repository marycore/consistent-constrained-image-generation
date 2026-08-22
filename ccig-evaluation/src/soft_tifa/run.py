from __future__ import annotations

import math
from pathlib import Path

from ..common.io import write_json
from ..common.types import MatchedItem, SoftTifaResult
from .base import VQABackend
from .scoring import score_subqa

# Cardinality constraints in this dataset only ever target N' in {1, 2, 3} (see
# ccig-dataset-gen/src/eval_dataset_gen/domain.py: COUNTS), and scenes have 3-9
# objects -- this just needs to comfortably cover both with room for the model to
# be wrong in either direction, not match either bound exactly.
_COUNT_HEADROOM = 2
_MIN_MAX_COUNT = 10


def _aggregate(scores: list[float]) -> tuple[float | None, float | None]:
    """Arithmetic and geometric mean of one image's sub-question scores.
    (None, None) if there were no scorable sub-questions at all."""
    if not scores:
        return None, None
    am = sum(scores) / len(scores)
    if any(s <= 0 for s in scores):
        gm = 0.0
    else:
        gm = math.exp(sum(math.log(s) for s in scores) / len(scores))
    return am, gm


def run_soft_tifa(
    items: list[MatchedItem], domain_module, backend: VQABackend, out_path: str | Path
) -> None:
    from PIL import Image

    results: list[SoftTifaResult] = []
    n_missing_subqa = 0

    for item in items:
        subqa = item.record.subqa
        if not subqa:
            # Dataset generated before the subqa field existed (or a template with no
            # subqa yet) -- not evaluable by this method, but not fatal to the run.
            n_missing_subqa += 1
            results.append(
                SoftTifaResult(
                    id=item.id,
                    prompt_field=item.prompt_field,
                    image_path=str(item.image_path),
                    score_am=None,
                    score_gm=None,
                    subquestions=[],
                    success=False,
                    error="record has no subqa (regenerate the dataset with the current verbalize.py)",
                )
            )
            print(f"[skip] {item.id}: no subqa on this record")
            continue

        try:
            image = Image.open(item.image_path).convert("RGB")
            max_count = max(item.record.number_of_objects + _COUNT_HEADROOM, _MIN_MAX_COUNT)
            subquestions = score_subqa(image, subqa, backend, domain_module, max_count)
            scorable = [sq.score for sq in subquestions if not sq.excluded_from_score]
            score_am, score_gm = _aggregate(scorable)
            results.append(
                SoftTifaResult(
                    id=item.id,
                    prompt_field=item.prompt_field,
                    image_path=str(item.image_path),
                    score_am=score_am,
                    score_gm=score_gm,
                    subquestions=subquestions,
                    success=True,
                    error=None,
                )
            )
            shown = f"AM={score_am:.3f} GM={score_gm:.3f}" if score_am is not None else "no scorable sub-questions"
            print(f"[ok]   {item.id}: {shown}")
        except Exception as e:
            results.append(
                SoftTifaResult(
                    id=item.id,
                    prompt_field=item.prompt_field,
                    image_path=str(item.image_path),
                    score_am=None,
                    score_gm=None,
                    subquestions=[],
                    success=False,
                    error=repr(e),
                )
            )
            print(f"[fail] {item.id}: {e}")

    if n_missing_subqa:
        print(
            f"[warn] {n_missing_subqa}/{len(items)} records had no subqa field and were skipped -- "
            "regenerate ccig_eval_dataset_{SAT,UNSAT}.jsonl with the current eval_dataset_gen/run.py "
            "to get one."
        )

    scored = [r for r in results if r.score_am is not None]
    dataset_am = sum(r.score_am for r in scored) / len(scored) if scored else None
    dataset_gm = sum(r.score_gm for r in scored) / len(scored) if scored else None

    write_json(
        out_path,
        {
            "method": "soft-tifa",
            "backend": backend.name,
            "dataset_score_am": dataset_am,  # mean of each image's own AM (soft-TIFA AM)
            "dataset_score_gm": dataset_gm,  # mean of each image's own GM (soft-TIFA GM)
            "results": [r.to_json() for r in results],
        },
    )
