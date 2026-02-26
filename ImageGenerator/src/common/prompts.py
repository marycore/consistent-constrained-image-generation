from __future__ import annotations

from .types import PromptRecord, Mode


def build_full_prompt(record: PromptRecord, mode: Mode) -> str:
    """Construct the full text prompt from base prompt and constraints."""

    if mode == "general":
        return f"{record['prompt']}\n{record['constraints_general']}".strip()
    if mode == "general_specific":
        return (
            f"{record['prompt']}\n"
            f"{record['constraints_general']}\n"
            f"{record['constraints_specific']}"
        ).strip()
    raise ValueError(f"Unknown mode: {mode}")

