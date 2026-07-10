"""
Parse ASP constraint template files and generate valid placeholder assignments.

Placeholder naming convention (from template .txt files):
  P1', P2', P3', P4'  — property names (color/shape/size/region)
  V1', V2', V3', V4'  — property values (domain of their paired P)
  D1', D2', D3'       — spatial directions (left/right/front/behind)
  N'                  — integer count (1/2/3)

V_i is always paired with P_i (V1'←P1', V2'←P2', etc.).
Inequalities in the rule text (e.g. P1'!=P2') are respected automatically.
"""

import re
import itertools
import random
from pathlib import Path

from domain import PROPERTIES, RELATIONS, COUNTS

# ── Regex patterns ─────────────────────────────────────────────────────────

# Matches any primed placeholder: P1', V2', D3', N'
# Uses (?<!\w) so we don't match mid-word; ' is not a word char so no trailing boundary needed.
_PH_RE = re.compile(r"(?<!\w)([PVD][1-4]'|N')")

# Matches inequality constraints between primed placeholders: P1'!=P2', V1'!=V3', …
_INEQ_RE = re.compile(r"(?<!\w)([PVD][1-4]'|N')\s*!=\s*([PVD][1-4]'|N')")


# ── Template loading ────────────────────────────────────────────────────────

def load_template(path: Path) -> tuple[str, str]:
    """
    Read an ASP template file and return (description, rule_text).

    Lines starting with '%' are comments that form the description.
    All other non-empty lines are the ASP rule text.
    """
    comments, rules = [], []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("%"):
            comments.append(stripped.lstrip("% ").strip())
        else:
            rules.append(stripped)
    description = " ".join(comments)
    rule_text = _normalize(rules)
    return description, rule_text


def _normalize(rule_lines: list[str]) -> str:
    """
    Join rule lines and normalize bare 'N' count placeholders to N'.

    Some C8 2-property templates write '!= N' instead of '!= N''.
    The normalization converts patterns like '!= N.' → '!= N'.' so the
    placeholder regex finds them consistently.
    """
    rule = "\n".join(rule_lines)
    # Replace bare N (count placeholder) that follows a comparison operator
    rule = re.sub(r"([!=<>]\s*)N(?!')", r"\1N'", rule)
    return rule


# ── Placeholder discovery ───────────────────────────────────────────────────

def placeholders(rule: str) -> list[str]:
    """Return unique primed placeholders found in rule, in order of first appearance."""
    seen: set[str] = set()
    result: list[str] = []
    for m in _PH_RE.finditer(rule):
        ph = m.group(1)
        if ph not in seen:
            seen.add(ph)
            result.append(ph)
    return result


def inequalities(rule: str) -> set[frozenset]:
    """Return set of {a, b} frozensets where a != b is required by the rule."""
    return {frozenset([a, b]) for a, b in _INEQ_RE.findall(rule)}


# ── Domain helpers ──────────────────────────────────────────────────────────

def _paired_p(v_ph: str) -> str:
    """Return the P placeholder paired with a V placeholder: V2' → P2'."""
    digit = v_ph[1]  # e.g. '2' from "V2'"
    return f"P{digit}'"


def _domain(ph: str, partial: dict[str, str]) -> list[str]:
    """
    Return the value domain for a placeholder, given the partial assignment so far.
    For V placeholders, the domain is the property values of the paired P placeholder.
    """
    prefix = ph[0]
    if prefix == "P":
        return list(PROPERTIES.keys())
    if prefix == "D":
        return list(RELATIONS)
    if prefix == "N":
        return list(COUNTS)
    # V placeholder — depends on paired P
    p_ph = _paired_p(ph)
    prop = partial.get(p_ph)
    if prop and prop in PROPERTIES:
        return list(PROPERTIES[prop])
    # fallback: union of all value domains
    return [v for vals in PROPERTIES.values() for v in vals]


def _violates(candidate: dict[str, str], ineqs: set[frozenset]) -> bool:
    """Return True if the candidate assignment violates any inequality constraint."""
    return any(
        candidate.get(a) == candidate.get(b)
        for pair in ineqs
        for a, b in [tuple(pair)]
        if a in candidate and b in candidate
    )


# ── Assignment generation ───────────────────────────────────────────────────

def sample_assignment(rule: str, rng: random.Random) -> dict[str, str]:
    """
    Sample one valid random assignment for all primed placeholders in rule.

    Placeholders are filled in order P → V → D → N, with V placeholders
    depending on their paired P. Explicit != constraints in the rule are enforced.

    Raises RuntimeError if a valid assignment cannot be found after many attempts.
    """
    phs = placeholders(rule)
    ineqs = inequalities(rule)

    p_phs = [ph for ph in phs if ph.startswith("P")]
    v_phs = [ph for ph in phs if ph.startswith("V")]
    d_phs = [ph for ph in phs if ph.startswith("D")]
    n_phs = [ph for ph in phs if ph.startswith("N")]

    for _ in range(2000):
        asgn: dict[str, str] = {}

        # Fill P placeholders
        valid = True
        for ph in p_phs:
            excluded = {asgn[o] for o in asgn if frozenset([ph, o]) in ineqs}
            choices = [p for p in PROPERTIES if p not in excluded]
            if not choices:
                valid = False
                break
            asgn[ph] = rng.choice(choices)
        if not valid:
            continue

        # Fill V placeholders (depend on paired P)
        for ph in v_phs:
            domain = _domain(ph, asgn)
            excluded = {asgn[o] for o in asgn if frozenset([ph, o]) in ineqs}
            choices = [v for v in domain if v not in excluded]
            if not choices:
                valid = False
                break
            asgn[ph] = rng.choice(choices)
        if not valid:
            continue

        # Fill D placeholders
        for ph in d_phs:
            excluded = {asgn[o] for o in asgn if frozenset([ph, o]) in ineqs}
            choices = [d for d in RELATIONS if d not in excluded]
            if not choices:
                valid = False
                break
            asgn[ph] = rng.choice(choices)
        if not valid:
            continue

        # Fill N placeholders
        for ph in n_phs:
            asgn[ph] = rng.choice(COUNTS)

        if all(ph in asgn for ph in phs) and not _violates(asgn, ineqs):
            return asgn

    raise RuntimeError(
        f"Could not find a valid assignment for rule after 2000 attempts.\n"
        f"Placeholders: {phs}\nInequalities: {ineqs}"
    )


def all_assignments(rule: str) -> list[dict[str, str]]:
    """
    Enumerate every valid assignment of primed placeholders in rule (exhaustive).

    Warning: can be large for templates with many placeholders. Use sample_assignment
    for dataset generation and this function only for analysis or small templates.
    """
    phs = placeholders(rule)
    ineqs = inequalities(rule)

    p_phs = [ph for ph in phs if ph.startswith("P")]
    v_phs = [ph for ph in phs if ph.startswith("V")]
    d_phs = [ph for ph in phs if ph.startswith("D")]
    n_phs = [ph for ph in phs if ph.startswith("N")]

    results: list[dict[str, str]] = []

    for p_vals in itertools.product(*[list(PROPERTIES.keys()) for _ in p_phs]):
        p_asgn = dict(zip(p_phs, p_vals))
        if _violates(p_asgn, ineqs):
            continue

        v_domains = [PROPERTIES[p_asgn.get(_paired_p(v_ph), list(PROPERTIES)[0])] for v_ph in v_phs]
        for v_vals in itertools.product(*v_domains):
            v_asgn = dict(zip(v_phs, v_vals))
            combined = {**p_asgn, **v_asgn}
            if _violates(combined, ineqs):
                continue

            for d_vals in itertools.product(*[list(RELATIONS) for _ in d_phs]):
                d_asgn = dict(zip(d_phs, d_vals))
                combined = {**p_asgn, **v_asgn, **d_asgn}
                if _violates(combined, ineqs):
                    continue

                for n_vals in itertools.product(*[list(COUNTS) for _ in n_phs]):
                    n_asgn = dict(zip(n_phs, n_vals))
                    full = {**combined, **n_asgn}
                    if not _violates(full, ineqs):
                        results.append(full)

    return results


# ── Rule instantiation ──────────────────────────────────────────────────────

def apply_assignment(rule: str, assignment: dict[str, str]) -> str:
    """
    Substitute all primed placeholders in rule with their assigned values.

    Longer placeholder names are replaced first to avoid partial matches
    (e.g. P1' before P1 if both existed, though all placeholders end with ').
    """
    out = rule
    for ph in sorted(assignment, key=len, reverse=True):
        out = out.replace(ph, assignment[ph])
    return out
