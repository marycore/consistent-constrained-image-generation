from __future__ import annotations

from typing import Dict

from .types import Runner, Category


_REGISTRY: Dict[str, Runner] = {}


def register(runner: Runner) -> None:
    if runner.model_id in _REGISTRY:
        raise ValueError(f"Runner already registered for model_id={runner.model_id}")
    _REGISTRY[runner.model_id] = runner


def get_runner(model_id: str) -> Runner:
    try:
        return _REGISTRY[model_id]
    except KeyError as e:
        raise KeyError(
            f"Unknown model_id '{model_id}'. "
            f"Available: {', '.join(sorted(_REGISTRY.keys())) or 'none'}"
        ) from e


def list_models() -> Dict[str, Category]:
    return {mid: r.category for mid, r in _REGISTRY.items()}

