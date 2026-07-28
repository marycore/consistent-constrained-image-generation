from __future__ import annotations

from pathlib import Path

from ..common.io import write_json
from ..common.types import JudgeResult, MatchedItem
from .base import VLMJudge


def run_judge(items: list[MatchedItem], judge: VLMJudge, out_path: str | Path) -> None:
    from PIL import Image

    results: list[JudgeResult] = []
    for item in items:
        try:
            image = Image.open(item.image_path).convert("RGB")
            verdict = judge.judge(image, item.prompt_text)
            results.append(
                JudgeResult(
                    id=item.id,
                    prompt_field=item.prompt_field,
                    prompt=item.prompt_text,
                    image_path=str(item.image_path),
                    score=verdict.score,
                    raw_score=verdict.raw_score,
                    rationale=verdict.rationale,
                    success=True,
                    error=None,
                )
            )
            print(f"[ok]   {item.id}: {verdict.score:.2f}")
        except Exception as e:
            results.append(
                JudgeResult(
                    id=item.id,
                    prompt_field=item.prompt_field,
                    prompt=item.prompt_text,
                    image_path=str(item.image_path),
                    score=None,
                    raw_score=None,
                    rationale=None,
                    success=False,
                    error=repr(e),
                )
            )
            print(f"[fail] {item.id}: {e}")

    write_json(
        out_path,
        {
            "method": "vlm-judge",
            "backend": judge.name,
            "results": [r.to_json() for r in results],
        },
    )
