from __future__ import annotations

import json
from pathlib import Path

from . import io as common_io, prompts as prompt_utils
from .api_utils import env_bool
from .types import GenerationMetadata, Mode, PromptRecord, now_timestamp
from ..closed_api.registry import build_closed_registry


def run_closed_via_registry(
    *,
    legacy_model_id: str,
    provider_model_id: str,
    category: str,
    prompts: list[PromptRecord],
    mode: Mode,
    seed: int,
    output_root: str,
) -> None:
    """
    Bridge legacy category runners to the unified closed_api registry implementation.
    Keeps legacy output layout:
      outputs/<category>/<legacy_model_id>/<mode>/<prompt_hash>__seed<seed>.png|.json
    """
    registry = build_closed_registry()
    if provider_model_id not in registry:
        raise RuntimeError(f"Provider model '{provider_model_id}' is not registered.")
    model = registry[provider_model_id]
    continue_on_error = env_bool("IMAGE_API_CONTINUE_ON_ERROR", True)

    total = len(prompts)
    for idx, record in enumerate(prompts, start=1):
        full_prompt = prompt_utils.build_full_prompt(record, mode)
        h = common_io.prompt_hash(full_prompt)
        base_dir = Path(output_root) / category / legacy_model_id / mode
        base_dir.mkdir(parents=True, exist_ok=True)
        image_path = base_dir / f"{h}__seed{seed}.png"
        json_path = base_dir / f"{h}__seed{seed}.json"

        result = model.generate(
            prompt=full_prompt,
            output_path=str(image_path),
            prompt_id=record["id"],
            complexity_level="NA",
            seed=seed,
            width=1024,
            height=1024,
            num_images=1,
            timeout_seconds=180,
            max_retries=3,
            constraints_general=record.get("constraints_general", ""),
            constraints_specific=record.get("constraints_specific", ""),
            template_family=None,
        )

        if result.success:
            width = result.generation_parameters.get("width")
            height = result.generation_parameters.get("height")
            resolution = (int(width), int(height)) if width and height else None
            metadata = GenerationMetadata(
                model_id=legacy_model_id,
                category=category,  # type: ignore[arg-type]
                mode=mode,
                full_prompt=full_prompt,
                seed=seed,
                steps=None,
                guidance_scale=None,
                resolution=resolution,
                dtype="api",
                device="api",
                scheduler=None,
                timestamp=now_timestamp(),
            )
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(metadata.to_dict(), f, indent=2)
        else:
            common_io.append_api_error(
                output_root=output_root,
                category=category,
                model_id=legacy_model_id,
                mode=mode,
                error_entry={
                    "id": record["id"],
                    "seed": seed,
                    "timestamp": now_timestamp(),
                    "error": result.error_message or "unknown error",
                    "provider_model_id": provider_model_id,
                },
            )
            if not continue_on_error:
                raise RuntimeError(
                    f"{legacy_model_id} failed for prompt id={record['id']}: {result.error_message}"
                )

        if idx % 25 == 0 or idx == total:
            print(f"[{legacy_model_id}] completed {idx}/{total}")
