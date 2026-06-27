#!/usr/bin/env python3
"""One-off builder: writes CLEVR_CCIG_templates/C*.json prompt templates.

Each entry's "text" array is a set of declarative scene-description
paraphrases (suitable as image-generation prompts) that encode the exact
logical structure (existence/universal/conditional/count/negation) of the
matching ASP rule in image_generation/ConstraintTemplates/CCIG_constraint_templates/.

"nodes" and "constraints" fields from the legacy L0-L7 template schema are
intentionally omitted: they are CLEVR functional-program leftovers that
ccig_template_lib.py never reads after loading (verified: only
asp_template_file, constraint_family, text, params, property_focus, and
relation_focus are consumed downstream).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "CLEVR_CCIG_templates"
PROPS = ["color", "material", "shape", "size"]
RELS = ["left", "right", "front", "behind"]
# Natural-language phrase for each relation, baked in directly wherever the
# relation is known at template-build time (relation_focus loops). Runtime
# relation placeholders (<D1>, <D2>, <D3>) are naturalized at instantiation
# time by ccig_template_lib.naturalize_constraint_text instead.
REL_PHRASE = {
    "left": "to the left of",
    "right": "to the right of",
    "front": "in front of",
    "behind": "behind",
}

# Maps asp_key -> ASP template file under CCIG_constraint_templates/.
# See ConstraintTemplates/CCIG_constraint_templates/README.txt for the
# formal C1-C9 definitions these implement.
ASP = {
    "C1_exist":               "constraint_templates_C1_exist.txt",
    "C1_forbid":              "constraint_templates_C1_forbid.txt",
    "C2_universal":           "constraint_templates_C2_universal.txt",
    "C3_conditional":         "constraint_templates_C3_conditional.txt",
    "C3_conditional_negated": "constraint_templates_C3_conditional_negated.txt",
    "C4_exist_pair":          "constraint_templates_C4_exist_pair.txt",
    "C4_forbid_pair":         "constraint_templates_C4_forbid_pair.txt",
    "C4_chain2":              "constraint_templates_C4_chain2.txt",
    "C4_chain3":              "constraint_templates_C4_chain3.txt",
    "C4_shared_hub":          "constraint_templates_C4_shared_hub.txt",
    "C5_pair_conditional":    "constraint_templates_C5_pair_conditional.txt",
    "C6_witness_universal":  "constraint_templates_C6_witness_universal.txt",
    "C7_implication":        "constraint_templates_C7_implication.txt",
    "C7_universal_witness":  "constraint_templates_C7_universal_witness.txt",
    "C7_unique_witness":     "constraint_templates_C7_unique_witness.txt",
    "C8_unary_count":        "constraint_templates_C8_unary_count.txt",
    "C8_relational_count":   "constraint_templates_C8_relational_count.txt",
    "C8_all_different":      "constraint_templates_C8_all_different.txt",
    "C9_count_coupling":     "constraint_templates_C9_count_coupling.txt",
}


def entry(
    *,
    asp_key: str,
    constraint_family: str,
    text: list[str],
    params: list[dict],
    property_focus: str | None = None,
    relation_focus: str | None = None,
) -> dict:
    out = {
        "asp_template_file": ASP[asp_key],
        "constraint_family": constraint_family,
        "text": text,
        "params": params,
    }
    if property_focus:
        out["property_focus"] = property_focus
    if relation_focus:
        out["relation_focus"] = relation_focus
    return out


def region_param(name: str = "<R1>") -> dict:
    return {"type": "Region", "name": name}


def value_param(ptype: str, name: str = "<V1>") -> dict:
    return {"type": ptype, "name": name}


def relation_param(name: str = "<D1>") -> dict:
    return {"type": "Relation", "name": name}


# ---------------------------------------------------------------------------
# C1: Existential Object -- exists x. Phi(x)
# ---------------------------------------------------------------------------
def c1_entries() -> list[dict]:
    entries = []
    for asp_key, family in [("C1_exist", "exist"), ("C1_forbid", "forbid")]:
        for prop in PROPS:
            ptype = prop.capitalize()
            if family == "exist":
                texts = [
                    f"Region <R1> contains at least one <V1> object.",
                    f"At least one <V1> object appears in region <R1>.",
                    f"There is a <V1> object somewhere in region <R1>.",
                ]
            else:
                texts = [
                    f"Region <R1> contains no <V1> object.",
                    f"No object in region <R1> is <V1>.",
                    f"Region <R1> has zero <V1> objects.",
                ]
            entries.append(
                entry(
                    asp_key=asp_key,
                    constraint_family=family,
                    property_focus=prop,
                    text=texts,
                    params=[region_param("<R1>"), value_param(ptype, "<V1>")],
                )
            )
    return entries


# ---------------------------------------------------------------------------
# C2: Universal Object -- forall x. Phi(x)
# ---------------------------------------------------------------------------
def c2_entries() -> list[dict]:
    entries = []
    for prop in PROPS:
        ptype = prop.capitalize()
        entries.append(
            entry(
                asp_key="C2_universal",
                constraint_family="universal",
                property_focus=prop,
                text=[
                    f"Every object in region <R1> has {prop} <V1>.",
                    f"All objects in region <R1> are <V1>.",
                    f"Region <R1> contains only <V1> objects.",
                ],
                params=[region_param("<R1>"), value_param(ptype, "<V1>")],
            )
        )
    return entries


# ---------------------------------------------------------------------------
# C3: Conditional Object -- forall x. Phi1(x) -> Phi2(x)
# ---------------------------------------------------------------------------
def c3_entries() -> list[dict]:
    entries = []
    for asp_key, family in [
        ("C3_conditional", "conditional"),
        ("C3_conditional_negated", "conditional_negated"),
    ]:
        for prop in PROPS:
            ptype = prop.capitalize()
            if family == "conditional":
                texts = [
                    f"Every <V1> object in region <R1> is also <V2>.",
                    f"All <V1> objects in region <R1> are also <V2>.",
                ]
            else:
                texts = [
                    f"Every <V1> object in region <R1> is never <V2>.",
                    f"No <V1> object in region <R1> is <V2>.",
                ]
            entries.append(
                entry(
                    asp_key=asp_key,
                    constraint_family=family,
                    property_focus=prop,
                    text=texts,
                    params=[
                        region_param("<R1>"),
                        value_param(ptype, "<V1>"),
                        {"type": "Property", "name": "<P2>"},
                        {"type": "Value", "name": "<V2>"},
                    ],
                )
            )
    return entries


# ---------------------------------------------------------------------------
# C4: Existential (Subgraph) -- exists x1..xt. Phi(x1..xt) and Rel_E(x1..xt)
# ---------------------------------------------------------------------------
def c4_entries() -> list[dict]:
    entries = []

    for asp_key, family in [("C4_exist_pair", "exist_pair"), ("C4_forbid_pair", "forbid_pair")]:
        for prop in PROPS:
            ptype = prop.capitalize()
            if family == "exist_pair":
                texts = [
                    f"A <V1> object in region <R1> is <D1> of a <V2> object in region <R2>.",
                    f"Some <V1> object in region <R1> is <D1> of a <V2> object in region <R2>.",
                ]
            else:
                texts = [
                    f"No <V1> object in region <R1> is ever <D1> of a <V2> object in region <R2>.",
                    f"A <V1> object in region <R1> is never <D1> of any <V2> object in region <R2>.",
                ]
            entries.append(
                entry(
                    asp_key=asp_key,
                    constraint_family=family,
                    property_focus=prop,
                    text=texts,
                    params=[
                        region_param("<R1>"),
                        value_param(ptype, "<V1>"),
                        region_param("<R2>"),
                        {"type": "Property", "name": "<P2>"},
                        {"type": "Value", "name": "<V2>"},
                        relation_param("<D1>"),
                    ],
                )
            )

    for asp_key, family, n_hops in [("C4_chain2", "chain2", 2), ("C4_chain3", "chain3", 3)]:
        for rel in RELS:
            phrase = REL_PHRASE[rel]
            if n_hops == 2:
                texts = [
                    f"Three objects form a chain: the first is {phrase} the second, and the second is <D2> of the third.",
                    f"A chain of three distinct objects exists where the first is {phrase} the second and the second is <D2> of the third.",
                ]
                params = [relation_param("<D1>"), relation_param("<D2>")]
            else:
                texts = [
                    f"Four objects form a chain: the first is {phrase} the second, the second is <D2> of the third, and the third is <D3> of the fourth.",
                    f"A chain of four distinct objects exists where the first is {phrase} the second, the second is <D2> of the third, and the third is <D3> of the fourth.",
                ]
                params = [relation_param("<D1>"), relation_param("<D2>"), relation_param("<D3>")]
            entries.append(
                entry(
                    asp_key=asp_key,
                    constraint_family=family,
                    relation_focus=rel,
                    text=texts,
                    params=params,
                )
            )

    for rel in RELS:
        phrase = REL_PHRASE[rel]
        entries.append(
            entry(
                asp_key="C4_shared_hub",
                constraint_family="shared_hub",
                relation_focus=rel,
                text=[
                    f"One object is {phrase} a second object and <D2> of a third, distinct object.",
                    f"A hub object is {phrase} one object and <D2> of another, distinct object.",
                ],
                params=[relation_param("<D1>"), relation_param("<D2>")],
            )
        )
    return entries


# ---------------------------------------------------------------------------
# C5: Conditional (Subgraph) -- forall x1,x2. (Phi_cond and Rel_cond) -> Rel_req
# ---------------------------------------------------------------------------
def c5_entries() -> list[dict]:
    entries = []
    for prop in PROPS:
        ptype = prop.capitalize()
        entries.append(
            entry(
                asp_key="C5_pair_conditional",
                constraint_family="pair_conditional",
                property_focus=prop,
                text=[
                    f"Whenever a <V1> object in region <R1> is <D1> of a <V2> object in region <R2>, it is also <D2> of that same object.",
                    f"Any <V1> object in region <R1> that is <D1> of a <V2> object in region <R2> must also be <D2> of it.",
                ],
                params=[
                    region_param("<R1>"),
                    value_param(ptype, "<V1>"),
                    region_param("<R2>"),
                    {"type": "Property", "name": "<P2>"},
                    {"type": "Value", "name": "<V2>"},
                    relation_param("<D1>"),
                    relation_param("<D2>"),
                ],
            )
        )
    return entries


# ---------------------------------------------------------------------------
# C6: Existential-Universal -- exists x. Phi_sel(x) and forall y. (Phi_tar(y) -> Rel(x,y))
# ---------------------------------------------------------------------------
def c6_entries() -> list[dict]:
    entries = []
    for prop in PROPS:
        ptype = prop.capitalize()
        entries.append(
            entry(
                asp_key="C6_witness_universal",
                constraint_family="witness_universal",
                property_focus=prop,
                text=[
                    f"Some <V1> object in region <R1> is <D1> of every <V2> object in region <R2>.",
                    f"At least one <V1> object in region <R1> is <D1> of all <V2> objects in region <R2>.",
                ],
                params=[
                    region_param("<R1>"),
                    value_param(ptype, "<V1>"),
                    region_param("<R2>"),
                    {"type": "Property", "name": "<P2>"},
                    {"type": "Value", "name": "<V2>"},
                    relation_param("<D1>"),
                ],
            )
        )
    return entries


# ---------------------------------------------------------------------------
# C7: Universal-Existential -- forall x. Phi_cond(x) -> exists y. (Phi_condY(y) and Rel_req(x,y))
# ---------------------------------------------------------------------------
def c7_entries() -> list[dict]:
    entries = []

    for prop in PROPS:
        ptype = prop.capitalize()
        entries.append(
            entry(
                asp_key="C7_implication",
                constraint_family="implication",
                property_focus=prop,
                text=[
                    f"Every <V1> object in region <R1> is <D1> of at least one other object.",
                    f"Each <V1> object in region <R1> is <D1> of some other object.",
                ],
                params=[region_param("<R1>"), value_param(ptype, "<V1>"), relation_param("<D1>")],
            )
        )

    for prop in PROPS:
        ptype = prop.capitalize()
        entries.append(
            entry(
                asp_key="C7_universal_witness",
                constraint_family="universal_witness",
                property_focus=prop,
                text=[
                    f"Every <V1> object in region <R1> has at least one <V2> object in region <R2> that is <D1> of it.",
                    f"Each <V1> object in region <R1> is matched by some <V2> object in region <R2> that is <D1> of it.",
                ],
                params=[
                    region_param("<R1>"),
                    value_param(ptype, "<V1>"),
                    region_param("<R2>"),
                    {"type": "Property", "name": "<P2>"},
                    {"type": "Value", "name": "<V2>"},
                    relation_param("<D1>"),
                ],
            )
        )

    for prop in PROPS:
        ptype = prop.capitalize()
        entries.append(
            entry(
                asp_key="C7_unique_witness",
                constraint_family="unique_witness",
                property_focus=prop,
                text=[
                    f"Every <V1> object in region <R1> has exactly one <V2> object in region <R2> that is <D1> of it.",
                    f"Each <V1> object in region <R1> is matched by exactly one <V2> object in region <R2> that is <D1> of it.",
                ],
                params=[
                    region_param("<R1>"),
                    value_param(ptype, "<V1>"),
                    region_param("<R2>"),
                    {"type": "Property", "name": "<P2>"},
                    {"type": "Value", "name": "<V2>"},
                    relation_param("<D1>"),
                ],
            )
        )
    return entries


# ---------------------------------------------------------------------------
# C8: Cardinality -- |{(x1..xt) : Phi and Rel_E}| theta k
# ---------------------------------------------------------------------------
def c8_entries() -> list[dict]:
    entries = []

    for prop in PROPS:
        ptype = prop.capitalize()
        entries.append(
            entry(
                asp_key="C8_unary_count",
                constraint_family="unary_count",
                property_focus=prop,
                text=[
                    f"Exactly <N1> <V1> objects are located in region <R1>.",
                    f"Region <R1> contains exactly <N1> <V1> objects.",
                ],
                params=[
                    region_param("<R1>"),
                    value_param(ptype, "<V1>"),
                    {"type": "Count", "name": "<N1>"},
                ],
            )
        )

    for prop in PROPS:
        ptype = prop.capitalize()
        entries.append(
            entry(
                asp_key="C8_relational_count",
                constraint_family="relational_count",
                property_focus=prop,
                text=[
                    f"Exactly <N1> <V1> objects are <D1> of some <V2> object in region <R2>.",
                    f"There are exactly <N1> <V1> objects that are <D1> of a <V2> object in region <R2>.",
                ],
                params=[
                    value_param(ptype, "<V1>"),
                    {"type": "Count", "name": "<N1>"},
                    region_param("<R2>"),
                    {"type": "Property", "name": "<P2>"},
                    {"type": "Value", "name": "<V2>"},
                    relation_param("<D1>"),
                ],
            )
        )

    for prop in PROPS:
        ptype = prop.capitalize()
        entries.append(
            entry(
                asp_key="C8_all_different",
                constraint_family="all_different",
                property_focus=prop,
                text=[
                    f"At most one object in region <R1> has {prop} <V1>.",
                    f"No two distinct objects in region <R1> both have {prop} <V1>.",
                ],
                params=[region_param("<R1>"), value_param(ptype, "<V1>")],
            )
        )
    return entries


# ---------------------------------------------------------------------------
# C9: Aggregate Comparison -- |{x1..xt : Phi_x and Rel_x}| theta |{y1..ym : Phi_y and Rel_y}|
# ---------------------------------------------------------------------------
def c9_entries() -> list[dict]:
    entries = []
    for prop in PROPS:
        ptype = prop.capitalize()
        entries.append(
            entry(
                asp_key="C9_count_coupling",
                constraint_family="count_coupling",
                property_focus=prop,
                text=[
                    f"The number of <V1> objects in region <R1> equals the number of <V2> objects in region <R2>.",
                    f"Region <R1> has exactly as many <V1> objects as region <R2> has <V2> objects.",
                ],
                params=[
                    region_param("<R1>"),
                    value_param(ptype, "<V1>"),
                    region_param("<R2>"),
                    {"type": "Property", "name": "<P2>"},
                    {"type": "Value", "name": "<V2>"},
                ],
            )
        )
    return entries


FILES = {
    "C1_existential_object.json":   c1_entries,
    "C2_universal_object.json":     c2_entries,
    "C3_conditional_object.json":   c3_entries,
    "C4_existential_subgraph.json": c4_entries,
    "C5_conditional_subgraph.json": c5_entries,
    "C6_existential_universal.json": c6_entries,
    "C7_universal_existential.json": c7_entries,
    "C8_cardinality.json":          c8_entries,
    "C9_aggregate_comparison.json": c9_entries,
}


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for filename, builder in FILES.items():
        path = ROOT / filename
        data = builder()
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote {path} ({len(data)} entries)")


if __name__ == "__main__":
    main()
