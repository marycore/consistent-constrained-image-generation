#!/usr/bin/env python3
"""One-off builder: writes CLEVR_CCIG_templates/L*.json in POC-compatible structure."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "CLEVR_CCIG_templates"
PROPS = ["color", "material", "shape", "size"]
RELS = ["left", "right", "front", "behind"]

# One ASP file per logic variant (single rule per file).
ASP = {
    "L0_exist": "constraint_templates_L0_exist.txt",
    "L0_forbid": "constraint_templates_L0_forbid.txt",
    "L1_exist_pair": "constraint_templates_L1_exist_pair.txt",
    "L1_forbid_pair": "constraint_templates_L1_forbid_pair.txt",
    "L2_chain2": "constraint_templates_L2_chain2.txt",
    "L3_shared_hub": "constraint_templates_L3_shared_hub.txt",
    "L4_implication": "constraint_templates_L4_implication.txt",
    "L4_forbid_conditional_pair": "constraint_templates_L4_forbid_conditional_pair.txt",
    "L5_universal_witness": "constraint_templates_L5_universal_witness.txt",
    "L6_unary_count": "constraint_templates_L6_unary_count.txt",
    "L6_relational_count": "constraint_templates_L6_relational_count.txt",
    "L7_witness_exist": "constraint_templates_L7_witness_exist.txt",
    "L7_witness_unique": "constraint_templates_L7_witness_unique.txt",
    "L8_count_coupling": "constraint_templates_L8_count_coupling.txt",
    "L8_all_different": "constraint_templates_L8_all_different.txt",
}


def node_scene() -> dict:
    return {"inputs": [], "type": "scene"}


def prop_terminal(prop: str) -> dict:
    return {"inputs": [1], "type": f"property_{prop}"}


def entry(
    *,
    asp_key: str,
    constraint_family: str,
    text: list[str],
    nodes: list[dict],
    params: list[dict],
    constraints: list[dict],
    property_focus: str | None = None,
    relation_focus: str | None = None,
) -> dict:
    out = {
        "asp_template_file": ASP[asp_key],
        "constraint_family": constraint_family,
        "text": text,
        "nodes": nodes,
        "params": params,
        "constraints": constraints,
    }
    if property_focus:
        out["property_focus"] = property_focus
    if relation_focus:
        out["relation_focus"] = relation_focus
    return out


def l0_entries() -> list[dict]:
    entries = []
    for asp_key, family in [("L0_exist", "exist"), ("L0_forbid", "forbid")]:
        for prop in PROPS:
            ptype = prop.capitalize()
            pname = "<V>"
            if family == "exist":
                texts = [
                    f"At least one object in region <R> must have {prop} <V>.",
                    f"Region <R> must contain at least one object whose {prop} is <V>.",
                    f"There must exist an object in region <R> with {prop} <V>.",
                ]
                ntype = "constraint_exist_region"
            else:
                texts = [
                    f"No object in region <R> may have {prop} <V>.",
                    f"Region <R> must not contain any object with {prop} <V>.",
                    f"It is forbidden for any object in region <R> to have {prop} <V>.",
                ]
                ntype = "constraint_forbid_region"
            entries.append(
                entry(
                    asp_key=asp_key,
                    constraint_family=family,
                    property_focus=prop,
                    text=texts,
                    nodes=[
                        node_scene(),
                        {"side_inputs": ["<R>", pname], "inputs": [0], "type": ntype},
                        prop_terminal(prop),
                    ],
                    params=[
                        {"type": "Region", "name": "<R>"},
                        {"type": ptype, "name": pname},
                    ],
                    constraints=[{"params": [pname], "type": "NULL"}],
                )
            )
    return entries


def l1_entries() -> list[dict]:
    entries = []
    for asp_key, family in [("L1_exist_pair", "exist_pair"), ("L1_forbid_pair", "forbid_pair")]:
        for prop in PROPS:
            ptype = prop.capitalize()
            if family == "exist_pair":
                texts = [
                    f"There must exist two distinct objects such that one in region <R1> has {prop} <V1>, "
                    f"one in region <R2> has <P2> <V2>, and the first is <D1> of the second.",
                    f"Some object in region <R1> with {prop} <V1> must be <D1> of an object in region <R2> with <P2> <V2>.",
                ]
                ntype = "constraint_exist_pair"
            else:
                texts = [
                    f"No pair of objects may have one in region <R1> with {prop} <V1>, "
                    f"one in region <R2> with <P2> <V2>, with the first <D1> of the second.",
                    f"It is forbidden for an object in region <R1> with {prop} <V1> to be <D1> of "
                    f"an object in region <R2> with <P2> <V2>.",
                ]
                ntype = "constraint_forbid_pair"
            entries.append(
                entry(
                    asp_key=asp_key,
                    constraint_family=family,
                    property_focus=prop,
                    text=texts,
                    nodes=[
                        node_scene(),
                        {
                            "side_inputs": ["<R1>", "<V1>", "<R2>", "<P2>", "<V2>", "<D1>"],
                            "inputs": [0],
                            "type": ntype,
                        },
                        prop_terminal(prop),
                    ],
                    params=[
                        {"type": "Region", "name": "<R1>"},
                        {"type": ptype, "name": "<V1>"},
                        {"type": "Region", "name": "<R2>"},
                        {"type": "Property", "name": "<P2>"},
                        {"type": "Value", "name": "<V2>"},
                        {"type": "Relation", "name": "<D1>"},
                    ],
                    constraints=[{"params": ["<V1>"], "type": "NULL"}],
                )
            )
    return entries


def l2_entries() -> list[dict]:
    entries = []
    for rel in RELS:
        entries.append(
            entry(
                asp_key="L2_chain2",
                constraint_family="chain2",
                relation_focus=rel,
                text=[
                    f"There must exist a chain of three distinct objects X1, X2, X3 such that X1 is <D1> of X2 and X2 is <D2> of X3, with <D1> fixed to {rel}.",
                    f"A 2-hop relational chain X1->{rel}->X2-><D2>->X3 must exist among three distinct objects.",
                    f"Some triple of objects must satisfy relations {rel} then <D2> in sequence.",
                ],
                nodes=[
                    node_scene(),
                    {"side_inputs": ["<D1>", "<D2>"], "inputs": [0], "type": "constraint_chain2"},
                    {"inputs": [1], "type": f"relation_{rel}"},
                ],
                params=[
                    {"type": "Relation", "name": "<D1>"},
                    {"type": "Relation", "name": "<D2>"},
                ],
                constraints=[{"params": ["<D1>"], "type": "NULL"}],
            )
        )
    return entries


def l3_entries() -> list[dict]:
    entries = []
    for rel in RELS:
        entries.append(
            entry(
                asp_key="L3_shared_hub",
                constraint_family="shared_hub",
                relation_focus=rel,
                text=[
                    f"There must exist an object Z and two distinct objects X1 and X2 such that Z is <D1> of X1 and Z is <D2> of X2, with <D1> fixed to {rel}.",
                    f"Some object Z must simultaneously satisfy relation {rel} to X1 and relation <D2> to X2.",
                    f"A shared object Z must link X1 and X2 via relations {rel} and <D2> respectively.",
                ],
                nodes=[
                    node_scene(),
                    {"side_inputs": ["<D1>", "<D2>"], "inputs": [0], "type": "constraint_shared_hub"},
                    {"inputs": [1], "type": f"relation_{rel}"},
                ],
                params=[
                    {"type": "Relation", "name": "<D1>"},
                    {"type": "Relation", "name": "<D2>"},
                ],
                constraints=[{"params": ["<D1>"], "type": "NULL"}],
            )
        )
    return entries


def l4_entries() -> list[dict]:
    entries = []
    for asp_key, family in [
        ("L4_implication", "implication"),
        ("L4_forbid_conditional_pair", "forbid_conditional_pair"),
    ]:
        for prop in PROPS:
            ptype = prop.capitalize()
            if family == "implication":
                texts = [
                    f"Every object in region <R1> with {prop} <V1> must be <D1> of at least one other object.",
                    f"If an object is in region <R1> and has {prop} <V1>, it must have the <D1> relation to some other object.",
                ]
                ntype = "constraint_implication"
            else:
                texts = [
                    f"No two distinct objects in regions <R1> and <R2> with {prop} <V1> and <P2> <V2> may satisfy relation <D1>.",
                    f"It is forbidden for an object in region <R1> with {prop} <V1> to be <D1> of an object in region <R2> with <P2> <V2>.",
                    f"It is forbidden for a <V1> object in region <R1> to be <D1> of a <V2> object in region <R2>.",
                ]
                ntype = "constraint_forbid_conditional_pair"
            entries.append(
                entry(
                    asp_key=asp_key,
                    constraint_family=family,
                    property_focus=prop,
                    text=texts,
                    nodes=[
                        node_scene(),
                        {
                            "side_inputs": ["<R1>", "<V1>", "<R2>", "<P2>", "<V2>", "<D1>"],
                            "inputs": [0],
                            "type": ntype,
                        },
                        prop_terminal(prop),
                    ],
                    params=[
                        {"type": "Region", "name": "<R1>"},
                        {"type": ptype, "name": "<V1>"},
                        {"type": "Region", "name": "<R2>"},
                        {"type": "Property", "name": "<P2>"},
                        {"type": "Value", "name": "<V2>"},
                        {"type": "Relation", "name": "<D1>"},
                    ],
                    constraints=[{"params": ["<V1>"], "type": "NULL"}],
                )
            )
    return entries


def l5_entries() -> list[dict]:
    entries = []
    for prop in PROPS:
        ptype = prop.capitalize()
        entries.append(
            entry(
                asp_key="L5_universal_witness",
                constraint_family="universal_witness",
                property_focus=prop,
                text=[
                    f"Every object in region <R1> with {prop} <V1> must have at least one witness object in region <R2> with <P2> <V2> related by <D1>.",
                    f"Each trigger object in region <R1> with {prop} <V1> requires a witness in region <R2> with <P2> <V2> standing in relation <D1> to it.",
                ],
                nodes=[
                    node_scene(),
                    {
                        "side_inputs": ["<R1>", "<V1>", "<R2>", "<P2>", "<V2>", "<D1>"],
                        "inputs": [0],
                        "type": "constraint_universal_witness",
                    },
                    prop_terminal(prop),
                ],
                params=[
                    {"type": "Region", "name": "<R1>"},
                    {"type": ptype, "name": "<V1>"},
                    {"type": "Region", "name": "<R2>"},
                    {"type": "Property", "name": "<P2>"},
                    {"type": "Value", "name": "<V2>"},
                    {"type": "Relation", "name": "<D1>"},
                ],
                constraints=[{"params": ["<V1>"], "type": "NULL"}],
            )
        )
    return entries


def l6_entries() -> list[dict]:
    entries = []
    for asp_key, family in [("L6_unary_count", "unary_count"), ("L6_relational_count", "relational_count")]:
        for prop in PROPS:
            ptype = prop.capitalize()
            if family == "unary_count":
                texts = [
                    f"Exactly <N1> objects in region <R1> must have {prop} <V1>.",
                    f"The count of objects in region <R1> with {prop} <V1> must equal <N1>.",
                ]
                ntype = "constraint_exact_count"
            else:
                texts = [
                    f"Exactly <N1> objects X must satisfy relation <D1> to some object in region <R2> with <P2> <V2>, with X having {prop} <V1>.",
                    f"The number of objects with {prop} <V1> that are <D1> of an object in region <R2> with <P2> <V2> must equal <N1>.",
                ]
                ntype = "constraint_relational_count"
            entries.append(
                entry(
                    asp_key=asp_key,
                    constraint_family=family,
                    property_focus=prop,
                    text=texts,
                    nodes=[
                        node_scene(),
                        {
                            "side_inputs": ["<R1>", "<V1>", "<N1>", "<R2>", "<P2>", "<V2>", "<D1>"],
                            "inputs": [0],
                            "type": ntype,
                        },
                        prop_terminal(prop),
                    ],
                    params=[
                        {"type": "Region", "name": "<R1>"},
                        {"type": ptype, "name": "<V1>"},
                        {"type": "Count", "name": "<N1>"},
                        {"type": "Region", "name": "<R2>"},
                        {"type": "Property", "name": "<P2>"},
                        {"type": "Value", "name": "<V2>"},
                        {"type": "Relation", "name": "<D1>"},
                    ],
                    constraints=[{"params": ["<V1>"], "type": "NULL"}],
                )
            )
    return entries


def l7_entries() -> list[dict]:
    entries = []
    for asp_key, family in [("L7_witness_exist", "witness_exist"), ("L7_witness_unique", "witness_unique")]:
        for prop in PROPS:
            ptype = prop.capitalize()
            if family == "witness_exist":
                texts = [
                    f"Every object in region <R1> with {prop} <V1> must have at least one witness in region <R2> with <P2> <V2> related by <D1>.",
                    f"Each source object in region <R1> with {prop} <V1> requires at least one matching witness object.",
                ]
                ntype = "constraint_injective_exist"
            else:
                texts = [
                    f"Every object in region <R1> with {prop} <V1> may have at most one witness in region <R2> with <P2> <V2> related by <D1>.",
                    f"Each source object in region <R1> with {prop} <V1> must have a unique witness object satisfying the relation and filter.",
                ]
                ntype = "constraint_injective_unique"
            entries.append(
                entry(
                    asp_key=asp_key,
                    constraint_family=family,
                    property_focus=prop,
                    text=texts,
                    nodes=[
                        node_scene(),
                        {
                            "side_inputs": ["<R1>", "<V1>", "<R2>", "<P2>", "<V2>", "<D1>"],
                            "inputs": [0],
                            "type": ntype,
                        },
                        prop_terminal(prop),
                    ],
                    params=[
                        {"type": "Region", "name": "<R1>"},
                        {"type": ptype, "name": "<V1>"},
                        {"type": "Region", "name": "<R2>"},
                        {"type": "Property", "name": "<P2>"},
                        {"type": "Value", "name": "<V2>"},
                        {"type": "Relation", "name": "<D1>"},
                    ],
                    constraints=[{"params": ["<V1>"], "type": "NULL"}],
                )
            )
    return entries


def l8_entries() -> list[dict]:
    entries = []
    for asp_key, family in [("L8_count_coupling", "count_coupling"), ("L8_all_different", "all_different")]:
        for prop in PROPS:
            ptype = prop.capitalize()
            if family == "count_coupling":
                texts = [
                    f"The number of objects in region <R1> with {prop} <V1> must equal the number of objects in region <R2> with <P2> <V2>.",
                    f"Global coupling: equal counts between objects in region <R1> with {prop} <V1> and objects in region <R2> with <P2> <V2>.",
                ]
                ntype = "constraint_global_count_coupling"
            else:
                texts = [
                    f"No two distinct objects in region <R1> may share the same {prop} value <V1>.",
                    f"All objects in region <R1> must have distinct values of {prop}; value <V1> cannot repeat.",
                ]
                ntype = "constraint_all_different"
            entries.append(
                entry(
                    asp_key=asp_key,
                    constraint_family=family,
                    property_focus=prop,
                    text=texts,
                    nodes=[
                        node_scene(),
                        {
                            "side_inputs": ["<R1>", "<V1>", "<R2>", "<P2>", "<V2>"],
                            "inputs": [0],
                            "type": ntype,
                        },
                        prop_terminal(prop),
                    ],
                    params=[
                        {"type": "Region", "name": "<R1>"},
                        {"type": ptype, "name": "<V1>"},
                        {"type": "Region", "name": "<R2>"},
                        {"type": "Property", "name": "<P2>"},
                        {"type": "Value", "name": "<V2>"},
                    ],
                    constraints=[{"params": ["<V1>"], "type": "NULL"}],
                )
            )
    return entries


FILES = {
    "L0_unary_attribute.json": l0_entries,
    "L1_single_relational.json": l1_entries,
    "L2_relational_composition.json": l2_entries,
    "L3_conjunctive_relational_binding.json": l3_entries,
    "L4_implication_negation_rules.json": l4_entries,
    "L5_universal_dependency.json": l5_entries,
    "L6_relational_aggregates.json": l6_entries,
    "L7_injective_matching.json": l7_entries,
    "L8_global_coupling.json": l8_entries,
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
