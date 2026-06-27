from __future__ import annotations

import itertools
import json
import random
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ComboComponentFilter:
    """Select which template to instantiate within a level."""

    level: str
    constraint_family: str | None = None
    property_focus: str | None = None
    template_index: int | None = None


@dataclass
class ComboSpec:
    """One combination job: list of component filters + optional label."""

    components: list[ComboComponentFilter]
    id: str | None = None
    instances: int | None = None


COLOR_VALUES = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
MATERIAL_VALUES = ["rubber", "metal"]
SHAPE_VALUES = ["cube", "cylinder", "sphere", "cone"]
SIZE_VALUES = ["small", "large", "medium"]
REL_VALUES = ["left", "right", "front", "behind"]
PROP_VALUES = ["color", "material", "shape", "size"]
REGION_VALUES = ["0", "1", "2", "3"]
COUNT_VALUES = ["1", "2", "3"]

DOMAIN: dict[str, list[str]] = {
    "color": COLOR_VALUES,
    "material": MATERIAL_VALUES,
    "shape": SHAPE_VALUES,
    "size": SIZE_VALUES,
}

PLACEHOLDER_PATTERN = re.compile(r"([RPDNV][0-9]')")
PARAM_PATTERN = re.compile(r"<[^>]+>")


@dataclass
class ComponentInstance:
    complexity_class: str
    template_file: str
    template_index: int
    asp_template_file: str
    constraint_family: str
    property_focus: str | None
    relation_focus: str | None
    param_assignment: dict[str, str]
    asp_assignment: dict[str, str]
    texts: list[str]
    asp_rule: str
    nodes: list[dict[str, Any]]
    params: list[dict[str, str]]
    constraints: list[dict[str, Any]]


class AspRuleCache:
    """Load one-rule ASP template files by filename."""

    def __init__(self, constraint_dir: Path) -> None:
        self.constraint_dir = constraint_dir
        self._cache: dict[str, tuple[str, str]] = {}

    def get(self, filename: str) -> tuple[str, str]:
        if filename not in self._cache:
            path = self.constraint_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing ASP template file: {path}")
            rules = parse_constraint_file(path)
            if len(rules) != 1:
                raise RuntimeError(
                    f"Expected exactly one ASP rule in {path}, found {len(rules)}"
                )
            self._cache[filename] = rules[0]
        return self._cache[filename]


def load_index(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_level_templates(templates_dir: Path, filename: str) -> list[dict[str, Any]]:
    with (templates_dir / filename).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_background(*paths: Path) -> str:
    parts: list[str] = []
    for path in paths:
        parts.append(path.read_text(encoding="utf-8").strip())
    return "\n\n".join(p for p in parts if p)


def parse_constraint_file(path: Path) -> list[tuple[str, str]]:
    """Parse a CCIG constraint template file into (comment, rule_block) entries.

    Each file holds exactly one logical rule block: zero or more leading "%"
    comment lines, followed by one or more ASP statements (helper rules
    defining auxiliary predicates, e.g. for the C6/C9 saturation/coupling
    patterns, plus a final integrity constraint). All rule lines are joined
    so helper rules are not silently dropped.
    """
    comments: list[str] = []
    rule_lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("%"):
                comments.append(line.lstrip("%").strip())
                continue
            rule_lines.append(line)
    if not rule_lines:
        return []
    return [(" ".join(comments), "\n".join(rule_lines))]


def value_space_for_placeholder(ph: str) -> list[str]:
    if ph.startswith("R"):
        return REGION_VALUES
    if ph.startswith("P"):
        return PROP_VALUES
    if ph.startswith("D"):
        return REL_VALUES
    if ph.startswith("N"):
        return COUNT_VALUES
    return COLOR_VALUES + MATERIAL_VALUES + SHAPE_VALUES + SIZE_VALUES


def apply_assignment(s: str, assignment: dict[str, str]) -> str:
    out = s
    for ph in sorted(assignment.keys(), key=len, reverse=True):
        out = out.replace(ph, assignment[ph])
        if ph.endswith("'"):
            out = out.replace(ph[:-1], assignment[ph])
    return out


def substitute_params(text: str, param_assignment: dict[str, str]) -> str:
    out = text
    for name, val in param_assignment.items():
        out = out.replace(name, val)
    return " ".join(out.split())


def _article_for(word: str) -> str:
    return "an" if word and word[0].lower() in "aeiou" else "a"


def naturalize_constraint_text(text: str, param_assignment: dict[str, str]) -> str:
    """Prefer value-as-adjective phrasing over explicit property names (material rubber -> rubber)."""
    out = text

    for prop in PROP_VALUES:
        out = re.sub(
            rf"\ban object in region (\d+) with {prop} (\w+)\b",
            lambda m, p=prop: f"{_article_for(m.group(2))} {m.group(2)} object in region {m.group(1)}",
            out,
            flags=re.IGNORECASE,
        )
        out = re.sub(
            rf"\bfor any object in region (\d+) to have {prop} (\w+)\b",
            lambda m: f"for any {_article_for(m.group(2))} {m.group(2)} object in region {m.group(1)}",
            out,
            flags=re.IGNORECASE,
        )
    out = re.sub(
        r"\bregions (\d+) and (\d+) with (material|shape|color|size) (\w+) and (material|shape|color|size) (\w+)\b",
        lambda m: (
            f"regions {m.group(1)} and {m.group(2)} with "
            f"a {m.group(4)} object and a {m.group(6)} object"
        ),
        out,
        flags=re.IGNORECASE,
    )
    for prop in PROP_VALUES:
        out = re.sub(
            rf"\bwith {prop} (\w+)\b",
            r"\1",
            out,
            flags=re.IGNORECASE,
        )
        out = re.sub(
            rf"\bmust have {prop} (\w+)\b",
            rf"must be \1",
            out,
            flags=re.IGNORECASE,
        )
        out = re.sub(
            rf"\bmay have {prop} (\w+)\b",
            rf"may be \1",
            out,
            flags=re.IGNORECASE,
        )
        out = re.sub(
            rf"\bwhose {prop} is (\w+)\b",
            rf"that is \1",
            out,
            flags=re.IGNORECASE,
        )

    p2 = param_assignment.get("<P2>")
    v2 = param_assignment.get("<V2>")
    if p2 and v2 and p2 in PROP_VALUES:
        out = re.sub(
            rf"\bwith {re.escape(p2)} {re.escape(v2)}\b",
            v2,
            out,
            flags=re.IGNORECASE,
        )
        out = re.sub(
            rf"\ban object in region (\d+) with {re.escape(v2)}\b",
            lambda m: f"{_article_for(v2)} {v2} object in region {m.group(1)}",
            out,
            flags=re.IGNORECASE,
        )

    rel_phrase = {
        "right": "to the right of",
        "left": "to the left of",
        "front": "in front of",
        "behind": "behind",
    }
    for rel, phrase in rel_phrase.items():
        out = re.sub(rf"\bto be {rel} of\b", f"to be {phrase}", out, flags=re.IGNORECASE)
        out = re.sub(rf"\bto be {rel} of an\b", f"to be {phrase} a", out, flags=re.IGNORECASE)

    out = re.sub(r"\bto be behind of\b", "to be behind", out, flags=re.IGNORECASE)
    out = re.sub(r"\bto be front of\b", "to be in front of", out, flags=re.IGNORECASE)
    out = re.sub(r"\bmust have the (\w+) relation\b", r"must be \1 of", out, flags=re.IGNORECASE)
    out = re.sub(r"\bsatisfy relation (\w+)\b", r"be \1-related", out, flags=re.IGNORECASE)

    # General bare "<relation> of" -> natural phrase, regardless of surrounding verb
    # tense (covers "is behind of", "stands left of", etc., not just "to be ... of").
    # Negative lookbehind avoids double-wrapping phrases already in final form
    # (e.g. "to the left of" already contains "left of").
    for rel, phrase in rel_phrase.items():
        prefix = phrase[: -len(f"{rel} of")] if phrase.endswith(f"{rel} of") else phrase
        lookbehind = f"(?<!{re.escape(prefix)})" if prefix and prefix != phrase else ""
        out = re.sub(rf"{lookbehind}\b{rel} of\b", phrase, out, flags=re.IGNORECASE)

    # Singular/plural + verb agreement for the common count=1 case, e.g.
    # "Exactly 1 red objects are behind ..." -> "Exactly 1 red object is behind ...".
    out = re.sub(r"\b1 (\w+ )?objects\b", lambda m: f"1 {m.group(1) or ''}object", out, flags=re.IGNORECASE)
    out = re.sub(r"\b(1 (?:\w+ )?object) are\b", r"\1 is", out, flags=re.IGNORECASE)
    out = re.sub(r"\b(1 (?:\w+ )?object) that are\b", r"\1 that is", out, flags=re.IGNORECASE)
    out = re.sub(r"\bThere are exactly 1\b", "There is exactly 1", out, flags=re.IGNORECASE)

    out = re.sub(r"\s+", " ", out).strip()
    return out


def instantiate_texts(
    template_entry: dict[str, Any],
    param_assignment: dict[str, str],
) -> list[str]:
    """All NL paraphrases for one sampled parameter assignment."""
    raw = template_entry.get("text", [])
    if isinstance(raw, str):
        raw = [raw]
    texts: list[str] = []
    seen: set[str] = set()
    for template_line in raw:
        line = substitute_params(template_line, param_assignment)
        line = naturalize_constraint_text(line, param_assignment)
        if not line.endswith("."):
            line += "."
        if line not in seen:
            seen.add(line)
            texts.append(line)
    return texts


def merge_text_variants(
    components: list[ComponentInstance],
    joiner: str = " And ",
) -> list[str]:
    """Cartesian product of per-constraint paraphrases (combo mode)."""
    if not components:
        return []
    merged: list[str] = []
    seen: set[str] = set()
    for parts in itertools.product(*(c.texts for c in components)):
        line = joiner.join(p.rstrip(".") for p in parts) + "."
        if line not in seen:
            seen.add(line)
            merged.append(line)
    return merged


def sample_value_for_property(prop: str, rng: random.Random) -> str:
    return rng.choice(DOMAIN[prop])


def sample_distinct_property(exclude: str, rng: random.Random) -> str:
    choices = [p for p in PROP_VALUES if p != exclude]
    return rng.choice(choices)


def build_asp_assignment(
    *,
    asp_rule: str,
    comment: str,
    template_entry: dict[str, Any],
    param_assignment: dict[str, str],
    rng: random.Random,
) -> dict[str, str]:
    """Map JSON param placeholders to ASP primed placeholders and sample any remaining."""
    property_focus = template_entry.get("property_focus")
    relation_focus = template_entry.get("relation_focus")

    asp_assignment: dict[str, str] = {}

    # Param-name to ASP placeholder conventions used across CCIG templates.
    param_to_asp = {
        "<R>": "R1'",
        "<R1>": "R1'",
        "<R2>": "R2'",
        "<V>": "V1'",
        "<V1>": "V1'",
        "<V2>": "V2'",
        "<D1>": "D1'",
        "<D2>": "D2'",
        "<N1>": "N1'",
    }
    for param_name, asp_name in param_to_asp.items():
        if param_name in param_assignment:
            asp_assignment[asp_name] = param_assignment[param_name]

    if property_focus:
        if "P1'" in asp_rule or "P1'" in comment:
            asp_assignment["P1'"] = property_focus
        if "P2'" in asp_rule or "P2'" in comment:
            if "P2'" not in asp_assignment:
                if "<P2>" in param_assignment:
                    p2 = param_assignment["<P2>"]
                else:
                    p2 = sample_distinct_property(property_focus, rng)
                    param_assignment["<P2>"] = p2
                asp_assignment["P2'"] = p2

    if relation_focus:
        asp_assignment["D1'"] = relation_focus

    if "<V1>" in param_assignment and "V1'" not in asp_assignment:
        asp_assignment["V1'"] = param_assignment["<V1>"]
    if "<V2>" in param_assignment and "V2'" not in asp_assignment:
        asp_assignment["V2'"] = param_assignment["<V2>"]
    if "<V>" in param_assignment and property_focus and "V1'" not in asp_assignment:
        asp_assignment["V1'"] = param_assignment["<V>"]

    placeholders = set(PLACEHOLDER_PATTERN.findall(asp_rule + " " + comment))
    for ph in placeholders:
        if ph in asp_assignment:
            continue
        if ph.startswith("V"):
            p_key = "P" + ph[1] + "'"
            p_val = asp_assignment.get(p_key)
            if p_val and p_val in DOMAIN:
                asp_assignment[ph] = sample_value_for_property(p_val, rng)
            else:
                asp_assignment[ph] = rng.choice(value_space_for_placeholder(ph))
        elif ph.startswith("P"):
            if ph not in asp_assignment:
                asp_assignment[ph] = rng.choice(PROP_VALUES)
        else:
            asp_assignment[ph] = rng.choice(value_space_for_placeholder(ph))

    # Fill param_assignment for any property params used only in ASP
    if property_focus and "<V1>" not in param_assignment and "V1'" in asp_assignment:
        param_assignment["<V1>"] = asp_assignment["V1'"]
    if "<V2>" not in param_assignment and "V2'" in asp_assignment:
        param_assignment["<V2>"] = asp_assignment["V2'"]
    if "<P2>" not in param_assignment and "P2'" in asp_assignment:
        param_assignment["<P2>"] = asp_assignment["P2'"]
    if "<D2>" not in param_assignment and "D2'" in asp_assignment:
        param_assignment["<D2>"] = asp_assignment["D2'"]

    return asp_assignment


def sample_param_assignment(
    template_entry: dict[str, Any],
    rng: random.Random,
) -> dict[str, str]:
    param_assignment: dict[str, str] = {}
    property_focus = template_entry.get("property_focus")
    relation_focus = template_entry.get("relation_focus")

    for param in template_entry.get("params", []):
        name = param["name"]
        ptype = param["type"]
        if ptype == "Region":
            param_assignment[name] = rng.choice(REGION_VALUES)
        elif ptype == "Color":
            param_assignment[name] = rng.choice(COLOR_VALUES)
        elif ptype == "Material":
            param_assignment[name] = rng.choice(MATERIAL_VALUES)
        elif ptype == "Shape":
            param_assignment[name] = rng.choice(SHAPE_VALUES)
        elif ptype == "Size":
            param_assignment[name] = rng.choice(SIZE_VALUES)
        elif ptype == "Relation":
            if relation_focus and name == "<D1>":
                param_assignment[name] = relation_focus
            else:
                param_assignment[name] = rng.choice(REL_VALUES)
        elif ptype == "Count":
            param_assignment[name] = rng.choice(COUNT_VALUES)
        elif ptype == "Property":
            if property_focus and name in ("<P1>", "<P>"):
                param_assignment[name] = property_focus
            else:
                param_assignment[name] = rng.choice(PROP_VALUES)
        elif ptype == "Value":
            pass  # filled after dependent property params are known

    # Fill value params that depend on a property param
    for param in template_entry.get("params", []):
        if param["type"] != "Value":
            continue
        name = param["name"]
        if name == "<V2>" and "<P2>" in param_assignment:
            param_assignment[name] = sample_value_for_property(param_assignment["<P2>"], rng)
        elif name == "<V1>" and property_focus and name not in param_assignment:
            param_assignment[name] = sample_value_for_property(property_focus, rng)

    # Ensure distinct regions when both present
    if "<R1>" in param_assignment and "<R2>" in param_assignment:
        if param_assignment["<R1>"] == param_assignment["<R2>"]:
            choices = [r for r in REGION_VALUES if r != param_assignment["<R1>"]]
            param_assignment["<R2>"] = rng.choice(choices)

    if property_focus:
        v_key = next((p["name"] for p in template_entry.get("params", []) if p["type"].title() == property_focus.title() or p["name"] in ("<V>", "<V1>")), None)
        if v_key and v_key not in param_assignment:
            param_assignment[v_key] = sample_value_for_property(property_focus, rng)

    return param_assignment


def instantiate_component(
    *,
    level: str,
    template_file: str,
    template_index: int,
    template_entry: dict[str, Any],
    asp_cache: AspRuleCache,
    rng: random.Random,
) -> ComponentInstance:
    asp_template_file = template_entry["asp_template_file"]
    constraint_family = template_entry["constraint_family"]
    comment, asp_rule = asp_cache.get(asp_template_file)

    param_assignment = sample_param_assignment(template_entry, rng)
    asp_assignment = build_asp_assignment(
        asp_rule=asp_rule,
        comment=comment,
        template_entry=template_entry,
        param_assignment=param_assignment,
        rng=rng,
    )

    texts = instantiate_texts(template_entry, param_assignment)
    asp_instantiated = apply_assignment(asp_rule, asp_assignment)

    return ComponentInstance(
        complexity_class=level,
        template_file=template_file,
        template_index=template_index,
        asp_template_file=asp_template_file,
        constraint_family=constraint_family,
        property_focus=template_entry.get("property_focus"),
        relation_focus=template_entry.get("relation_focus"),
        param_assignment=param_assignment,
        asp_assignment=asp_assignment,
        texts=texts,
        asp_rule=asp_instantiated,
        nodes=template_entry.get("nodes", []),
        params=template_entry.get("params", []),
        constraints=template_entry.get("constraints", []),
    )


def build_asp_program(
    *,
    background: str,
    n_objects: int,
    constraint_rules: list[str],
) -> str:
    parts = [background.strip(), f"object(0..{n_objects})."]
    parts.extend(rule.strip() for rule in constraint_rules if rule.strip())
    return "\n".join(parts)


def make_combo_key(components: list[ComponentInstance]) -> str:
    parts = []
    for c in components:
        focus = c.property_focus or c.relation_focus or "-"
        parts.append(f"{c.complexity_class}:{c.constraint_family}:{c.asp_template_file}:{focus}")
    return "|".join(sorted(parts))


def record_asp_rules(record: dict[str, Any]) -> list[str]:
    if "asp_rules" in record:
        rules = record["asp_rules"]
        return rules if isinstance(rules, list) else [rules]
    if "asp_rule" in record:
        return [record["asp_rule"]]
    if record.get("components"):
        return [c["asp_rule"] for c in record["components"] if c.get("asp_rule")]
    return []


def record_to_asp_code(
    record: dict[str, Any],
    background: str,
) -> str:
    """Build full clingo program from a JSONL record (legacy asp_code still accepted)."""
    legacy = record.get("asp_code")
    if legacy:
        return legacy
    rules = record_asp_rules(record)
    if not rules:
        raise ValueError(f"Record {record.get('id')}: no asp_rules or asp_code")
    n_objects = record.get("n_objects")
    if n_objects is None:
        raise ValueError(f"Record {record.get('id')}: missing n_objects (needed to assemble asp_code)")
    return build_asp_program(
        background=background,
        n_objects=int(n_objects),
        constraint_rules=rules,
    )


def complexity_class_label(classes: list[str]) -> str:
    return "+".join(sorted(set(classes)))


def validate_with_clingo(
    asp_code: str,
    clingo_bin: str,
    *,
    time_limit_sec: int = 10,
) -> tuple[bool, str]:
    """Return (is_sat, clingo_output). Uses clingo --time-limit to avoid hanging."""
    with tempfile.NamedTemporaryFile("w", suffix=".lp", delete=True, encoding="utf-8") as tf:
        tf.write(asp_code + "\n")
        tf.flush()
        cmd = [clingo_bin, f"--time-limit={time_limit_sec}", "0", tf.name]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=time_limit_sec + 20,
            )
        except subprocess.TimeoutExpired:
            return False, f"TIMEOUT: clingo exceeded {time_limit_sec}s ({' '.join(cmd)})"
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        if "parsing failed" in out.lower() or "syntax error" in out.lower():
            return False, f"ASP_PARSE_ERROR\n{out}"
        if "UNSATISFIABLE" in out:
            return False, out
        if "SATISFIABLE" in out:
            return True, out
        if "TIME LIMIT" in out or "Time limit reached" in out:
            return False, out
        return False, out


def load_combo_specs(path: Path | None) -> list[ComboSpec]:
    """
    Load combo specifications from JSON.

    Supports two formats (can be combined in one file):

    1) Legacy level-only pairs:
       {"pairs": [["L0", "L1"], ...]}

    2) Explicit component filters (level + optional constraint_family):
       {"combos": [
         {"id": "L0_exist+L0_forbid", "components": [
           {"level": "L0", "constraint_family": "exist"},
           {"level": "L0", "constraint_family": "forbid"}
         ]},
         ...
       ]}
    """
    if path is None or not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    specs: list[ComboSpec] = []

    for pair in data.get("pairs", []):
        if not isinstance(pair, list) or len(pair) < 2:
            raise ValueError(f"Invalid legacy pair (need >=2 levels): {pair}")
        specs.append(
            ComboSpec(components=[ComboComponentFilter(level=str(level)) for level in pair])
        )

    for raw in data.get("combos", []):
        if "components" not in raw:
            raise ValueError(f"Combo entry missing 'components': {raw}")
        components = []
        for c in raw["components"]:
            if "level" not in c:
                raise ValueError(f"Combo component missing 'level': {c}")
            components.append(
                ComboComponentFilter(
                    level=str(c["level"]),
                    constraint_family=c.get("constraint_family"),
                    property_focus=c.get("property_focus"),
                    template_index=c.get("template_index"),
                )
            )
        if len(components) < 2:
            raise ValueError(f"Combo needs >=2 components: {raw}")
        specs.append(
            ComboSpec(
                components=components,
                id=raw.get("id"),
                instances=raw.get("instances"),
            )
        )

    return specs


def load_combo_pairs(path: Path | None) -> list[list[str]]:
    """Deprecated: use load_combo_specs. Returns level-only pairs for backward compat."""
    specs = load_combo_specs(path)
    return [[c.level for c in s.components] for s in specs]


def filter_template_indices(
    templates: list[dict[str, Any]],
    filt: ComboComponentFilter,
) -> list[int]:
    indices = list(range(len(templates)))
    if filt.constraint_family is not None:
        indices = [
            i
            for i in indices
            if templates[i].get("constraint_family") == filt.constraint_family
        ]
    if filt.property_focus is not None:
        indices = [
            i
            for i in indices
            if templates[i].get("property_focus") == filt.property_focus
        ]
    if filt.template_index is not None:
        if filt.template_index not in indices:
            raise ValueError(
                f"template_index {filt.template_index} does not match filters "
                f"level={filt.level} family={filt.constraint_family}"
            )
        indices = [filt.template_index]
    if not indices:
        raise ValueError(
            f"No templates match filter: level={filt.level}, "
            f"constraint_family={filt.constraint_family}, "
            f"property_focus={filt.property_focus}"
        )
    return indices


def pick_template_entry(
    templates: list[dict[str, Any]],
    rng: random.Random,
    template_index: int | None = None,
    component_filter: ComboComponentFilter | None = None,
) -> tuple[int, dict[str, Any]]:
    if component_filter is not None:
        if template_index is not None:
            raise ValueError("Pass template_index via component_filter, not both")
        candidates = filter_template_indices(templates, component_filter)
        idx = rng.choice(candidates)
        return idx, templates[idx]
    if template_index is not None:
        return template_index, templates[template_index]
    idx = rng.randrange(len(templates))
    return idx, templates[idx]
