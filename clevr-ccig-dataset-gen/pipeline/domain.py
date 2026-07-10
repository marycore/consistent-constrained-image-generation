"""Property domains and ASP background loader for CCIG scene generation."""

from pathlib import Path

PROPERTIES: dict[str, list[str]] = {
    "color":    ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"],
    "shape":    ["cube", "cylinder", "sphere"],
    "size":     ["small", "large"],
    "material": ["rubber", "metal"],
    "region":   ["reg_0", "reg_1", "reg_2", "reg_3"],
}
RELATIONS: list[str] = ["left", "right", "front", "behind"]
COUNTS: list[str] = ["1", "2", "3"]

_BACKGROUND_LP = Path(__file__).parent / "asp_background" / "background.lp"


def background_asp(n_objects: int = 4) -> str:
    """Return the full background ASP program for a scene with n_objects.

    Property facts and choice rules are generated from PROPERTIES (single source
    of truth). Structural rules (layout, relationships, axioms) come from background.lp.
    """
    lines = [f"object(0..{n_objects - 1}).", ""]

    for prop, vals in PROPERTIES.items():
        for val in vals:
            lines.append(f"property({prop}, {val}).")
        lines.append("")

    for prop in PROPERTIES:
        lines.append(f"1 {{ hasProperty(X, {prop}, V) : property({prop}, V) }} 1 :- object(X).")
    lines.append("")

    lines.append(_BACKGROUND_LP.read_text())
    return "\n".join(lines)
