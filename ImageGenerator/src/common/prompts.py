from __future__ import annotations

from .types import PromptRecord, Mode

_CLEVR_PREAMBLE = (
    "Create a CLEVR-style scene on a white background divided into four regions:\n"
    "Region 0 (top-left), Region 1 (top-right), Region 2 (bottom-left), and Region 3 (bottom-right).\n"
    "\n"
    "The scene contains {n} objects. Each object can be a cube, sphere, cylinder, or cone; "
    "can be gray, red, blue, green, brown, purple, cyan, or yellow; "
    "can be small, medium, or large; and can be made of metal or rubber.\n"
    "\n"
    "Place {n} objects so that: {constraint}"
)


def build_clevr_prompt(constraint_text: str, n_objects: int) -> str:
    return _CLEVR_PREAMBLE.format(n=n_objects, constraint=constraint_text)


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

