"""
Natural language verbalization of instantiated CCIG constraints.

Given a template filename stem (e.g. 'C1_2prop_neg') and a completed
placeholder assignment, returns three NL descriptions at different granularities:
  short  — one concise sentence
  medium — one clear sentence with full context
  long   — two-to-three sentences with semantic explanation

Region-specific grammar is handled automatically:
  - Non-region properties use adjective phrasing: "a red object", "every cube object"
  - Region values use locative phrasing: "an object in region_1", "every object in region_2"
"""

from __future__ import annotations

_DIR_PHRASE = {
    "left":   "to the left of",
    "right":  "to the right of",
    "front":  "in front of",
    "behind": "behind",
}

# ── Linguistic helpers ──────────────────────────────────────────────────────

def _adj(p: str, v: str) -> str:
    """Adjectival phrase for a property value: 'red', 'in region_1', 'cube', 'small'."""
    return f"in {v}" if p == "region" else v


def _obj(p: str, v: str) -> str:
    """Indefinite singular object phrase: 'a red object', 'an object in region_1'."""
    if p == "region":
        return f"an object in {v}"
    art = "an" if v[0] in "aeiou" else "a"
    return f"{art} {v} object"


def _Obj(p: str, v: str) -> str:
    """Capitalized version of _obj for sentence start."""
    s = _obj(p, v)
    return s[0].upper() + s[1:]


def _objs(p: str, v: str) -> str:
    """Plural object phrase: 'red objects', 'objects in region_1'."""
    return f"objects in {v}" if p == "region" else f"{v} objects"


def _not_adj(p: str, v: str) -> str:
    """Negated property phrase: 'not red', 'not in region_1', 'non-cube'."""
    return f"not in {v}" if p == "region" else f"non-{v}"


def _not_obj(p: str, v: str) -> str:
    """'an object that is not red', 'an object not in region_1'."""
    if p == "region":
        return f"an object not in {v}"
    return f"an object that is not {v}"

def _not_bare(p: str, v: str) -> str:
    """'object that is not red', 'an object not in region_1'."""
    if p == "region":
        return f"object not in {v}"
    return f"object that is not {v}"


def _bare(p: str, v: str) -> str:
    """
    Noun phrase without article: 'gray object', 'object in region_1'.
    Use after quantifiers or 'at least one': 'at least one gray object'.
    """
    return f"object in {v}" if p == "region" else f"{v} object"


def _quant(q: str, p: str, v: str) -> str:
    """
    Quantified noun phrase: 'every gray object', 'every object in region_1'.
    q is the quantifier word (every, each, no, some, any, …).
    """
    if p == "region":
        return f"{q} object in {v}"
    return f"{q} {v} object"

def _not_quant(q: str, p: str, v: str) -> str:
    """
    Quantified noun phrase: 'every gray object', 'every object in region_1'.
    q is the quantifier word (every, each, no, some, any, …).
    """
    if p == "region":
        return f"{q} object not in {v}"
    return f"{q} object that is not {v}"


def _dir(d: str) -> str:
    """'to the left of', 'in front of', 'behind', 'to the right of'."""
    return _DIR_PHRASE.get(d, d)


def _multi_adj(pairs: list[tuple[str, str]]) -> str:
    """Comma-joined adjectives for multiple (prop, val) pairs."""
    parts = [_adj(p, v) for p, v in pairs]
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def _multi_obj(pairs: list[tuple[str, str]]) -> str:
    """
    Object with multiple properties: 'a red cube object', 'a red object in region_1',
    'a cube object in region_1'.
    """
    non_region = [(p, v) for p, v in pairs if p != "region"]
    region_pairs = [(p, v) for p, v in pairs if p == "region"]

    adjs = " ".join(v for _, v in non_region)
    loc = " and ".join(f"in {v}" for _, v in region_pairs)

    if adjs and loc:
        art = "an" if adjs[0] in "aeiou" else "a"
        return f"{art} {adjs} object {loc}"
    if adjs:
        art = "an" if adjs[0] in "aeiou" else "a"
        return f"{art} {adjs} object"
    # region only
    art = "an"
    return f"{art} object {loc}"


def _not_objs(p: str, v: str) -> str:
    """Plural of objects lacking property: 'non-red objects', 'objects not in region_1'."""
    return f"objects not in {v}" if p == "region" else f"non-{v} objects"


def _n_obj(n: str) -> str:
    """'object' for n='1', 'objects' otherwise."""
    return "object" if n == "1" else "objects"


def _count_word(n: str) -> str:
    return {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five"}.get(n, n)


def _at_least(n: str) -> str:
    return f"at least {_count_word(n)}" if n != "1" else "at least one"


def _at_most(n: str) -> str:
    return f"at most {_count_word(n)}"


def _exactly(n: str) -> str:
    w = _count_word(n)
    return f"exactly {w}" if n != "1" else "exactly one"

   

# ── Per-class verbalizers ───────────────────────────────────────────────────

def _g(a: dict, *keys: str) -> tuple:
    """Quick getter: returns tuple of assignment values for given placeholder keys."""
    return tuple(a.get(k, "?") for k in keys)


def _c1(variant: str, a: dict) -> dict[str, str]:
    
    if variant == "1prop":
        p1, v1 = _g(a, "P1'", "V1'")

        return dict(
            short=f"{_Obj(p1, v1)} is in the scene.",
            medium=f"The scene contains at least one {_bare(p1, v1)}.",
            long=(f"Among all objects placed in the scene, at least one must be {_adj(p1, v1)}; "
                  f"no restriction is placed on the remaining objects."),
        )

    if variant == "1prop_neg":
        p1, v1 = _g(a, "P1'", "V1'")

        return dict(
            short=f"Some object is {_not_adj(p1, v1)}.",
            medium=f"The scene contains at least one object that is {_not_adj(p1, v1)}.",
            long=(f"Not all objects need be {_adj(p1, v1)}; the constraint ensures that at least one "
                  f"object in the scene avoids this property value."),
        )

    if variant == "1prop_2val_neg":
        
        p1, v1, v2 = _g(a, "P1'", "V1'", "V2'")
        return dict(
            short=f"Some object is neither {_adj(p1, v1)} nor {_adj(p1, v2)}.",
            medium=f"The scene contains at least one object that is neither {_adj(p1, v1)} nor {_adj(p1, v2)}.",
            long=(f"Among all objects in the scene, at least one must avoid both {_adj(p1, v1)} "
                  f"and {_adj(p1, v2)} for property {p1}."),
        )

    if variant == "1prop_3val_neg":
        p1, v1, v2, v3 = _g(a, "P1'", "V1'", "V2'", "V3'")
        return dict(
            short=f"Some object is neither {_adj(p1, v1)}, {_adj(p1, v2)}, nor {_adj(p1, v3)}.",
            medium=f"The scene contains at least one object that is none of {_adj(p1, v1)}, {_adj(p1, v2)}, or {_adj(p1, v3)}.",
            long = (f"Among all objects in the scene, at least one object must have a value for {p1} "
                    f"that is neither {_adj(p1, v1)}, {_adj(p1, v2)}, nor {_adj(p1, v3)}."),
        )

    if variant == "2prop":
        p1, v1, p2, v2 = _g(a, "P1'", "V1'", "P2'", "V2'")
        o = _multi_obj([(p1, v1), (p2, v2)])
        return dict(
            short=f"{o.capitalize()} is in the scene.",
            medium=f"The scene contains at least one object that is {_adj(p1, v1)} and {_adj(p2, v2)}.",
            long=(f"At least one object combining {_adj(p1, v1)} with {_adj(p2, v2)} must appear "
                  f"somewhere in the scene; other objects are unconstrained."),
        )

    if variant == "2prop_neg":
        p1, v1, p2, v2 = _g(a, "P1'", "V1'", "P2'", "V2'")
        return dict(
            short=f"Some object is {_not_adj(p1, v1)} and {_not_adj(p2, v2)}.",
            medium=f"The scene contains at least one object that is neither {_adj(p1, v1)} nor {_adj(p2, v2)}.",
            long=(f"At least one object must simultaneously avoid being {_adj(p1, v1)} "
                  f"and avoid being {_adj(p2, v2)}."),
        )
    
    if variant == "2prop_mix_neg":
        p1, v1, p2, v2 = _g(a, "P1'", "V1'", "P2'", "V2'")
        return dict(
            short=f"There exists an object that is {_adj(p1, v1)} and  {_not_adj(p2, v2)}.",
            medium=f"The scene contains at least one object that is {_adj(p1, v1)} but not {_adj(p2, v2)}.",
            long=(f"At least one object must simultaneously be {_adj(p1, v1)} "
                  f"and avoid being {_adj(p2, v2)}."),
        )

    if variant == "3prop":
        p1, v1, p2, v2, p3, v3 = _g(a, "P1'", "V1'",  "P2'", "V2'", "P3'", "V3'")
        o = _multi_obj([(p1, v1), (p2, v2), (p3, v3)])
        return dict(
            short=f"{o.capitalize()} is in the scene.",
            medium=f"The scene contains at least one object that is {_adj(p1, v1)}, {_adj(p2, v2)}, and {_adj(p3, v3)}.",
            long=(f"Among all objects in the scene, at least one must satisfy all three property "
                  f"conditions: {_adj(p1, v1)}, {_adj(p2, v2)}, and {_adj(p3, v3)}."),
        )

    if variant == "3prop_val_mix_neg":
        p1, v1, p2, v2, p3, v3, v4 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "V4'")
        return dict(
            short=(f"Some object that is {_adj(p1, v1)} and {_adj(p2, v2)} "
                   f"is neither {_adj(p3, v3)} nor {_adj(p3, v4)}."),
            medium=(f"The scene contains at least one object that is {_adj(p1, v1)} and {_adj(p2, v2)}, "
                    f"while being neither {_adj(p3, v3)} nor {_adj(p3, v4)}."),
            long=(f"At least one object must satisfy the following conditions: {_adj(p1, v1)} and {_adj(p2, v2)}, "
                  f"but must not be {_adj(p3, v3)} and must not be {_adj(p3, v4)}."),
        )

    if variant == "4prop":
        p1, v1, p2, v2, p3, v3, p4, v4 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "P4'", "V4'")
        return dict(
            short=f"Some object is {_adj(p1, v1)}, {_adj(p2, v2)}, {_adj(p3, v3)}, and {_adj(p4, v4)}.",
            long=(f"The scene contains at least one object satisfying all four properties: "
                    f"{_adj(p1, v1)}, {_adj(p2, v2)}, {_adj(p3, v3)}, and {_adj(p4, v4)}."),
            medium=(f"At least one object must simultaneously be {_adj(p1, v1)}, {_adj(p2, v2)}, "
                  f"{_adj(p3, v3)}, and {_adj(p4, v4)}."),
        )

    if variant == "4prop_mix_neg":
        p1, v1, p2, v2, p3, v3, p4, v4 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "P4'", "V4'")
        return dict(
            short=(f"Some object that is {_adj(p1, v1)} and {_adj(p2, v2)} "
                   f"is {_not_adj(p3, v3)} and {_not_adj(p4, v4)}."),
            medium=(f"The scene contains at least one object that is {_adj(p1, v1)} and {_adj(p2, v2)}, "
                    f"while being {_not_adj(p3, v3)} and {_not_adj(p4, v4)}."),
            long=(f"At least one object must satisfy the following requirements: {_adj(p1, v1)} and {_adj(p2, v2)}, "
                  f"while not being {_adj(p3, v3)} and not being {_adj(p4, v4)}."),
        )

    return None

def _c2(variant: str, a: dict) -> dict[str, str]:
    
    if variant == "1prop":
        p1, v1 = _g(a, "P1'", "V1'")

        return dict(
            short=f"All objects are {_adj(p1, v1)}.",
            medium=f"Every single object in the scene must be {_adj(p1, v1)}.",
            long=(f"The scene enforces a uniform {p1} constraint: each and every object must be "
                  f"{_adj(p1, v1)}, with no exceptions permitted."),
        )

    if variant == "1prop_neg":
        p1, v1 = _g(a, "P1'", "V1'")

        return dict(
            short=f"No object is {_adj(p1, v1)}.",
            medium=f"The scene contains no {_objs(p1, v1)} whatsoever.",
            long=(f"Every object in the scene must avoid being {_adj(p1, v1)}; "
                  f"this property value is globally forbidden."),
        )

    if variant == "1prop_2val_neg":
        p1, v1, v2 = _g(a, "P1'", "V1'", "V2'")
        return dict(
            short=f"No object is either {_adj(p1, v1)} or {_adj(p1, v2)}.",
            medium=f"No object in the scene can be {_adj(p1, v1)} and no object in the scene can be {_adj(p1, v2)}.",
            long=(f"The constraint forbids any object from having the following two values for the property {p1} "
                  f": no object can either be {_adj(p1, v1)} or {_adj(p1, v2)}."),
        )

    if variant == "2prop":
        p1, v1, p2, v2 = _g(a, "P1'", "V1'", "P2'", "V2'")
        return dict(
            short=f"Every object is {_adj(p1, v1)} and {_adj(p2, v2)}.",
            medium=f"Each object in the scene must simultaneously be {_adj(p1, v1)} and {_adj(p2, v2)}.",
            long=(f"The scene enforces two universal requirements: every object must be "
                  f"{_adj(p1, v1)} and also {_adj(p2, v2)}, with no object free to deviate."),
        )

    if variant == "2prop_neg":
        p1, v1, p2, v2 = _g(a, "P1'", "V1'", "P2'", "V2'")
        return dict(
            short=f"No object is {_adj(p1, v1)} and {_adj(p2, v2)} simultaneously.",
            medium=f"The scene forbids any object from being {_adj(p1, v1)} and {_adj(p2, v2)} simultaneously.",
            long=(f"An object that is simultaneously {_adj(p1, v1)} and {_adj(p2, v2)} must never "
                  f"appear; every object must avoid at least one of these two property values."),
        )

    if variant == "2prop_mix_neg":
        p1, v1, p2, v2 = _g(a, "P1'", "V1'", "P2'", "V2'")
        return dict(
            short=f"No object is {_adj(p1, v1)} and {_not_adj(p2, v2)} simultaneously.",
            medium=f"The scene forbids any object from being {_adj(p1, v1)} and {_not_adj(p2, v2)} simultaneously.",
            long=(f"An object that is simultaneously {_adj(p1, v1)} and {_not_adj(p2, v2)} must never "
                  f"appear."),
        )

    if variant == "3prop":
        p1, v1, p2, v2, p3, v3 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'")
        return dict(
            short=f"Every object is {_adj(p1, v1)}, {_adj(p2, v2)}, and {_adj(p3, v3)}.",
            medium=f"Each object in the scene must be {_adj(p1, v1)}, {_adj(p2, v2)}, and {_adj(p3, v3)}.",
            long=(f"The scene enforces three universal requirements: every object must be "
                  f"{_adj(p1, v1)}, {_adj(p2, v2)}, and {_adj(p3, v3)} simultaneously."),
        )

    if variant == "3prop_val_mix_neg":
        p1, v1, p2, v2, p3, v3, v4 = _g(a, "P1'", "V1'","P2'", "V2'", "P3'", "V3'", "V4'")
        return dict(
            short=(f"No object must be simultaneously be {_adj(p1, v1)} and {_adj(p2, v2)} "
                   f"and is neither {_adj(p3, v3)} nor {_adj(p3, v4)}."),
            medium=(f"It is impossible to have an object that is simultaneously {_adj(p1, v1)} and {_adj(p2, v2)} and "
                    f"{_not_adj(p3, v3)} and {_not_adj(p3, v4)}."),
            long=(f"The scene requires that any object with the following combination is not present: {_adj(p1, v1)} with {_adj(p2, v2)} "
                  f"and {_not_adj(p3, v3)} and {_not_adj(p3, v4)}."),
        )

    if variant == "4prop_neg":
        p1, v1, p2, v2, p3, v3, p4, v4 = _g(a, "P1'", "V1'","P2'", "V2'", "P3'", "V3'", "P4'", "V4'")
        return dict(
            short=(f"No object must be simultaneously {_adj(p1, v1)} and {_adj(p2, v2)} "
                   f"and is neither {_adj(p3, v3)} nor {_adj(p4, v4)}."),
            medium=(f"It is impossible to have an object that is simultaneously {_adj(p1, v1)} and {_adj(p2, v2)} and "
                    f"{_not_adj(p3, v3)} and {_not_adj(p4, v4)}."),
            long=(f"The scene requires that any object with the following combination is not present: {_adj(p1, v1)} and {_adj(p2, v2)} "
                  f"and {_not_adj(p3, v3)} and {_not_adj(p4, v4)}."),
        )
        
    if variant == "4prop":
        p1, v1, p2, v2, p3, v3, p4, v4 = _g(a, "P1'", "V1'","P2'", "V2'", "P3'", "V3'", "P4'", "V4'")
        
        return dict(
            short=f"Every object is {_adj(p1, v1)}, {_adj(p2, v2)}, {_adj(p3, v3)}, and {_adj(p4, v4)}.",
            medium=(f"Each object in the scene must simultaneously satisfy all four: "
                    f"{_adj(p1, v1)}, {_adj(p2, v2)}, {_adj(p3, v3)}, {_adj(p4, v4)}."),
            long=(f"The scene enforces four universal requirements at once: every object must be "
                  f"{_adj(p1, v1)}, {_adj(p2, v2)}, {_adj(p3, v3)}, and {_adj(p4, v4)}."),
        )

    return None


def _c3(variant: str, a: dict) -> dict[str, str]:

    if variant == "1propA_1prop_neg":
        p1, v1, p2, v2 = _g(a, "P1'", "V1'", "P2'", "V2'")
        return dict(
            short=(f"At least one object is {_adj(p1, v1)}, and "
                f"{_quant('every', p1, v1)} is {_not_adj(p2, v2)}."),
            medium=(f"There must be at least one object that is {_adj(p1, v1)}. "
                f"Each object that is {_adj(p1, v1)} must not be {_adj(p2, v2)}."),
            long=(f"The scene must contain at least one object that is {_adj(p1, v1)}. "
                f"Whenever an object is {_adj(p1, v1)}, it must avoid being {_adj(p2, v2)}; "
                f"the combination {_adj(p1, v1)} with {_adj(p2, v2)} is forbidden."),
        )

    if variant == "1propA_1propC":
        p1, v1, p2, v2 = _g(a, "P1'", "V1'", "P2'", "V2'")
        return dict(
            short=(f"At least one object is {_adj(p1, v1)}, and "
                f"{_quant('every', p1, v1)} is also {_adj(p2, v2)}."),
            medium=(f"There must be at least one object that is {_adj(p1, v1)}. "
                f"Each object that is {_adj(p1, v1)} must also be {_adj(p2, v2)}."),
            long=(f"The scene must contain at least one object that is {_adj(p1, v1)}. "
                f"Whenever an object is {_adj(p1, v1)}, it must simultaneously be {_adj(p2, v2)}; "
                f"objects that are not {_adj(p1, v1)} are unconstrained."),
        )

    if variant == "1propA_2propC":
        p1, v1, p2, v2, p3, v3 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'")
        return dict(
            short=(f"At least one object is {_adj(p1, v1)}, and "
                f"{_quant('every', p1, v1)} must be {_adj(p2, v2)} and {_adj(p3, v3)}."),
            medium=(f"There must be at least one object that is {_adj(p1, v1)}. "
                    f"Each object that is {_adj(p1, v1)} must simultaneously satisfy "
                    f"{_adj(p2, v2)} and {_adj(p3, v3)}."),
            long=(f"The scene must contain at least one object that is {_adj(p1, v1)}. "
                f"Whenever an object is {_adj(p1, v1)}, it must be both {_adj(p2, v2)} "
                f"and {_adj(p3, v3)}; violating either consequent property is forbidden."),
        )

    
    if variant == "1propA_3propC":
        p1, v1, p2, v2, p3, v3, p4, v4 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "P4'", "V4'")
        return dict(
            short=(f"At least one object is {_adj(p1, v1)}, and "
                f"{_quant('every', p1, v1)} must be {_adj(p2, v2)}, {_adj(p3, v3)}, and {_adj(p4, v4)}."),
            medium=(f"There must be at least one object that is {_adj(p1, v1)}. "
                    f"Each {_bare(p1, v1)} must satisfy all three consequent properties: "
                    f"{_adj(p2, v2)}, {_adj(p3, v3)}, and {_adj(p4, v4)}."),
            long=(f"The scene must contain at least one object that is {_adj(p1, v1)}. "
                f"The condition {_adj(p1, v1)} triggers three simultaneous requirements: "
                f"{_adj(p2, v2)}, {_adj(p3, v3)}, and {_adj(p4, v4)}; any {_bare(p1, v1)} "
                f"missing any of these is forbidden."),
        )


    if variant == "1propA_neg_1propC":
        p1, v1, p2, v2 = _g(a, "P1'", "V1'", "P2'", "V2'")
        return dict(
            short=(f"At least one object is {_not_adj(p1, v1)}, and every object that is "
                f"{_not_adj(p1, v1)} must be {_adj(p2, v2)}."),
            medium=(f"There must be at least one object that is not {_adj(p1, v1)}. "
                    f"Each object that is not {_adj(p1, v1)} is required to be {_adj(p2, v2)}."),
            long=(f"The scene must contain at least one object that is not {_adj(p1, v1)}. "
                f"Among objects that are not {_adj(p1, v1)}, all of them must be {_adj(p2, v2)}; "
                f"only {_adj(p1, v1)} objects escape this requirement."),
        )

    if variant == "2propA_1propC":
        p1, v1, p2, v2, p3, v3 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'")
        return dict(
            short=(f"At least one object is {_adj(p1, v1)} and {_adj(p2, v2)}, and every such object must also {_adj(p3, v3)}."),
            medium=(f"There must be at least one object that is {_adj(p1, v1)} and {_adj(p2, v2)}. "
                f"Each object that is both {_adj(p1, v1)} and {_adj(p2, v2)} must also be {_adj(p3, v3)}."),
            long=(f"The scene must contain at least one object that is {_adj(p1, v1)} and {_adj(p2, v2)}. "
                f"Whenever there is a combination of the following properties, that is, an object is {_adj(p1, v1)} and {_adj(p2, v2)}, "
                f"it must additionally be {_adj(p3, v3)}."),
        )

    if variant == "2propA_2propC":
        p1, v1, p2, v2, p3, v3, p4, v4 = _g(a, "P1'", "V1'","P2'", "V2'", "P3'", "V3'", "P4'", "V4'")
        return dict(
            short=(f"At least one object is {_adj(p1, v1)} and {_adj(p2, v2)}, and every such "
                f"object must be {_adj(p3, v3)} and {_adj(p4, v4)}."),
            medium=(f"There must be at least one object that is {_adj(p1, v1)} and {_adj(p2, v2)}. "
                f"Each object that is {_adj(p1, v1)} and {_adj(p2, v2)} must also be "
                f"{_adj(p3, v3)} and {_adj(p4, v4)}."),
            long=(f"The scene must contain at least one object that is {_adj(p1, v1)} and {_adj(p2, v2)}. "
                f"Whenever there is a combination of the following properties, that is, the object is {_adj(p1, v1)} and {_adj(p2, v2)}, then it "
                f"requires that the object is also {_adj(p3, v3)} and {_adj(p4, v4)}."),
        )
    
    if variant == "2propA_neg_2propC":
        p1, v1, p2, v2, p3, v3, p4, v4 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "P4'", "V4'")
        return dict(
            short=(f"At least one object is {_adj(p1, v1)} and {_not_adj(p2, v2)}, and every such "
                f"object must be {_adj(p3, v3)} and {_adj(p4, v4)}."),
            medium=(f"There must be at least one object that is {_adj(p1, v1)} but not "
                    f"{_adj(p2, v2)}. Each object that is {_adj(p1, v1)} but not {_adj(p2, v2)} "
                    f"must be {_adj(p3, v3)} and {_adj(p4, v4)}."),
            long=(f"The scene must contain at least one object that is {_adj(p1, v1)} while not being {_adj(p2, v2)}. "
                f"An object that is {_adj(p1, v1)} while not being {_adj(p2, v2)} must "
                f"satisfy being {_adj(p3, v3)} and {_adj(p4, v4)} simultaneously."),
        )
    
    if variant == "3propA_1propC_neg":
        p1, v1, p2, v2, p3, v3, p4, v4 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "P4'", "V4'")
        return dict(
            short=(f"At least one object is {_adj(p1, v1)}, {_adj(p2, v2)}, and {_adj(p3, v3)}, "
                f"and every such object must not be {_adj(p4, v4)}."),
            medium=(f"There must be at least one object that is {_adj(p1, v1)}, {_adj(p2, v2)}, "
                    f"and {_adj(p3, v3)}. Each object that is {_adj(p1, v1)}, {_adj(p2, v2)}, and "
                    f"{_adj(p3, v3)} must not be {_adj(p4, v4)}."),
            long=(f"The scene must contain at least one object that is {_adj(p1, v1)}, "
                f"{_adj(p2, v2)}, and {_adj(p3, v3)}. "
                f"An object satisfying the following three antecedent conditions of being ({_adj(p1, v1)}, "
                f"{_adj(p2, v2)}, and {_adj(p3, v3)}) is forbidden from being {_adj(p4, v4)}."),
        )



    if variant == "3propA_1propC":
        p1, v1, p2, v2, p3, v3, p4, v4 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "P4'", "V4'")
        return dict(
            short=(f"At least one object is {_adj(p1, v1)}, {_adj(p2, v2)}, and {_adj(p3, v3)}, "
                f"and every such object must be {_adj(p4, v4)}."),
            medium=(f"There must be at least one object that is {_adj(p1, v1)}, {_adj(p2, v2)}, "
                    f"and {_adj(p3, v3)}. Each object that is {_adj(p1, v1)}, {_adj(p2, v2)}, and "
                    f"{_adj(p3, v3)} must also be {_adj(p4, v4)}."),
            long=(f"The scene must contain at least one object that is {_adj(p1, v1)}, "
                f"{_adj(p2, v2)}, and {_adj(p3, v3)}. "
                f"An object that is {_adj(p1, v1)}, {_adj(p2, v2)}, and {_adj(p3, v3)} "
                f"is required to additionally be {_adj(p4, v4)}."),
        )




    return None


def _c4(variant: str, a: dict) -> dict[str, str]:
    
    if variant == "1prop_2hop":

        p1, v1, p2, v2, p3, v3, d1, d2 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "D1'", "D2'")
        return dict(
            short=(f"There exists {_obj(p1, v1)} that is {_dir(d1)} {_obj(p2, v2)}, "
                   f"which is {_dir(d2)} {_obj(p3, v3)}."),
            medium=(f"The scene contains a 2-hop chain: {_obj(p1, v1)} is {_dir(d1)} "
                    f"{_obj(p2, v2)}, and the latter is {_dir(d2)} {_obj(p3, v3)}."),
            long=(f"At least one triple of distinct objects must form a 2-hop chain with specific "
                  f"properties: {_obj(p1, v1)} {_dir(d1)} {_obj(p2, v2)} that is {_dir(d2)} {_obj(p3, v3)}."),
        )

    if variant == "1prop_2hop_mix":

        p1, v1, p2, v2, p3, v3, d1, d2 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "D1'", "D2'")
        return dict(
            short=(f"There exists {_obj(p1, v1)} that is {_dir(d1)} {_obj(p2, v2)}, "
                   f"which is {_dir(d2)} {_not_obj(p3, v3)}."),
            medium=(f"The scene contains a 2-hop chain: {_obj(p1, v1)} is {_dir(d1)} "
                    f"{_obj(p2, v2)}, and the latter is {_dir(d2)} {_not_obj(p3, v3)}."),
            long=(f"At least one triple of distinct objects must form a 2-hop chain with specific "
                  f"properties: {_obj(p1, v1)} {_dir(d1)} {_obj(p2, v2)} that is {_dir(d2)} {_not_obj(p3, v3)}."),
        )
        
    if variant == "1prop_shared":
        p1, v1, p2, v2, p3, v3, d1, d2 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "D1'", "D2'")
        
        return dict(
            short=(f"There exists {_obj(p3, v3)} that is {_dir(d1)} {_obj(p1, v1)}, "
                   f"and {_dir(d2)} {_obj(p2, v2)}."),
            medium=(f"The scene contains an object that is {_adj(p3, v3)} and it is {_dir(d1)} "
                    f"{_obj(p1, v1)}, and is {_dir(d2)} {_obj(p2, v2)}."),
            long=(f"There exists at least one triple of distinct objects such that the following constraints are satisfied: "
                  f" {_obj(p3, v3)} {_dir(d1)} {_obj(p1, v1)}, and is {_dir(d2)} {_obj(p2, v2)}."),
        )
        
    if variant == "1prop_shared_mix":
        p1, v1, p2, v2, p3, v3, d1, d2 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "D1'", "D2'")
        
        return dict(
            short=(f"There exists {_obj(p3, v3)} that is {_dir(d1)} {_obj(p1, v1)}, "
                   f"and {_dir(d2)} {_not_obj(p2, v2)}."),
            medium=(f"The scene contains an object that is {_adj(p3, v3)} and it is {_dir(d1)} "
                    f"{_obj(p1, v1)}, and is {_dir(d2)} {_not_obj(p2, v2)}."),
            long=(f"There exists at least one triple of distinct objects such that the following constraints are satisfied: "
                  f" {_obj(p3, v3)} {_dir(d1)} {_obj(p1, v1)}, and is {_dir(d2)} {_not_obj(p2, v2)}."),
        )
    
    return None


def _c5(variant: str, a: dict) -> dict[str, str]:
    
    
    """
    if variant == "pair_propA_relC":
        p1, v1, p2, v2, d1 = _g(a, "P1'", "V1'", "P2'", "V2'", "D1'")
        return dict(
            short=(f"If there is {_obj(p1, v1)} and {_obj(p2, v2)}, then the former must be {_dir(d1)} the latter."),
            medium=(f"For every pair of distinct objects where one is {_adj(p1, v1)} and the other "
                  f"is {_adj(p2, v2)}, the first must stand {_dir(d1)} the second."),
            long=(f"Whenever there is a pair of objects - the first is {_adj(p1, v1)} and the second is "
                    f"{_adj(p2, v2)}, it must be ensured that the first is always {_dir(d1)} the second."),
            
        )
    """
    if variant == "pair_propA_relC":
        p1, v1, p2, v2, d1 = _g(a, "P1'", "V1'", "P2'", "V2'", "D1'")
        return dict(
            short=(f"At least one object is {_adj(p1, v1)} and a distinct object is {_adj(p2, v2)}, "
                f"and the former must always be {_dir(d1)} the latter."),
            medium=(f"There must be at least one object that is {_adj(p1, v1)} and at least one "
                f"distinct object that is {_adj(p2, v2)}. For every such pair, the first must "
                f"stand {_dir(d1)} the second."),
            long=(f"The scene must contain at least one object that is {_adj(p1, v1)} and at "
                f"least one distinct object that is {_adj(p2, v2)}. "
                f"Whenever there is a pair of objects - the first is {_adj(p1, v1)} and the second is "
                f"{_adj(p2, v2)}, it must be ensured that the first is always {_dir(d1)} the second."),
        )


# TILL HERE




    if variant == "pair_propRelA_propC":
        p1, v1, p2, v2, d1, p3, v3 = _g(a, "P1'", "V1'", "P2'", "V2'", "D1'", "P3'", "V3'")
        return dict(
            short=(f"If there is {_obj(p1, v1)} {_dir(d1)} {_obj(p2, v2)}, "
            f"then the latter must also be {_adj(p3, v3)}."),
            medium=(f"For every distinct pair of objects where one is {_adj(p1, v1)}, and the other is {_adj(p2, v2)}, "
                  f"and the former is {_dir(d1)} latter, it must be that the latter is also {_adj(p3, v3)}."),
            long=(f"For every pair of distinct {_bare(p1, v1)} and {_bare(p2, v2)}, where the {_bare(p1, v1)} is {_dir(d1)} "
                  f" the {_bare(p2, v2)}, it must be ensured that the {_bare(p2, v2)} is also {_adj(p3, v3)}."),
            
        )

    if variant == "pair_propRelA_RelC":
        p1, v1, p2, v2, d1, d2 = _g(a, "P1'", "V1'", "P2'", "V2'", "D1'",  "D2'")
        
        return dict(
            short=(f"If there is {_obj(p1, v1)} {_dir(d1)} {_obj(p2, v2)}, "
            f"then it must also be {_dir(d2)} the latter."),
            medium=(f"For every distinct pair of objects where one is {_adj(p1, v1)}, and the other is {_adj(p2, v2)}, "
                  f"and the former is {_dir(d1)} the latter, it must be that the former is also {_dir(d2)} the latter."),
            long=(f"For every pair of distinct {_bare(p1, v1)} and {_bare(p2, v2)}, where the {_bare(p1, v1)} is {_dir(d1)} "
                  f" the {_bare(p2, v2)}, it must be ensured that the {_bare(p1, v1)} is also {_dir(d1)} the {_bare(p2, v2)}."),
        )

    if variant == "pair_relA_propC":
        d1, p1, v1 = a.get("D1'", "P1'", "V1'")
        return dict(
            short=(f"Only objects that are {_adj(p1, v1)} can be {_dir(d1)} any object."),
            
            medium=(f"All objects that are {_dir(d1)} some object, "
                   f"must be {_adj(p1, v1)}."),
            long=(f"For every distinct pair of objects where the first stands {_dir(d1)} the second, "
                  f"the first is required to be {_adj(p1, v1)}."),
        )

    if variant == "triple_propA_RelC":
        d1, p1, v1, p2, v2, p3, v3, d2 = a.get("D1'", "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "D2'")
        return dict(
            short = (f"If one object is {_adj(p1, v1)}, another is {_adj(p2, v2)}, and a third is {_adj(p3, v3)}, "
                    f"then the first object must be {_dir(d1)} the second, and the second must be {_dir(d2)} the third."),
            medium=(f"For every triplet of objects, where the first is {_adj(p1,v1)}, the second is {_adj(p2,v2)}, and the third is {_adj(p3,v3)}, then "
                  f" the first must stand {_dir(d1)} the second and the second must stand {_dir(d2)} the third."),

            long=(f"If there are three distinct objects such that the first is {_obj(p1,v1)}, the second is {_obj(p2,v2)} and the third is {_obj(p3,v3)}, then, "
                   f"it must be that the {_obj(p1,v1)} is {_dir(d1)} the {_obj(p2,v2)} and the {_obj(p2,v2)} is {_dir(d2)} the {_obj(p3,v3)}."),
            
        )

    if variant == "triple_propRelA_relC":
        d1, p1, v1, p2, v2, p3, v3, d2 = a.get("D1'", "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "D2'")
        return dict(
            
            short = (f"If one object is {_adj(p1, v1)}, another is {_adj(p2, v2)}, and a third is {_adj(p3, v3)}, and "
                    f"the first object is {_dir(d1)} the second, then, the second must be {_dir(d2)} the third."),
            medium=(f"For every triplet of objects, where the first is {_adj(p1,v1)}, the second is {_adj(p2,v2)}, and the third is {_adj(p3,v3)}, and "
                  f" the first stands {_dir(d1)} the second, then, the second must stand {_dir(d2)} the third."),
            long=(f"If there are three distinct objects such that the first is {_obj(p1,v1)}, the second is {_obj(p2,v2)} and the third is {_obj(p3,v3)}, and "
                   f"the {_obj(p1,v1)} is {_dir(d1)} the {_obj(p2,v2)}, then, the {_obj(p2,v2)} must be {_dir(d2)} the {_obj(p3,v3)}."),
        )

    return None


def _c6(variant: str, a: dict) -> dict[str, str]:
    
    if variant == "1prop":
        p1, v1, p2, v2, d1 = _g(a, "P1'", "V1'", "P2'", "V2'", "D1'")
        return dict(
            short=(f"There exists {_quant('Some', p1, v1)} such that it is {_dir(d1)} {_quant('every', p2, v2)}."),
            medium=(f"At least one {_bare(p1, v1)} stands {_dir(d1)} every single "
                    f"{_bare(p2, v2)} in the scene."),
            long=(f"There exists at least one {_bare(p1, v1)} that, "
                  f"for every {_bare(p2, v2)} in the scene, stands {_dir(d1)} it."),
        )

    if variant == "1prop_neg":
        p1, v1, p2, v2, d1 = _g(a, "P1'", "V1'", "P2'", "V2'", "D1'")
        return dict(
            short=(f"There exists {_quant('some', p1, v1)} such that it is {_dir(d1)} {_not_quant('every', p2, v2)}."),
            medium=(f"At least one {_bare(p1, v1)} exists that is standing {_dir(d1)} "
                    f"{_not_quant('every', p2, v2)} in the scene."),
            long=(f"There exists at least one {_bare(p1, v1)} that, for every {_not_ bare(p2, v2)} in the scene, stands {_dir(d1)} it."),
        )

    if variant == "2prop":
        p1, v1, p2, v2, p3, v3, p4, v4, d1 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "P4'", "V4'", "D1'")
        return dict(
            short=(f"There exists {_quant('Some', p1, v1)} that is {_adj(p2, v2)} such that it is {_dir(d1)} {_quant('every', p2, v2)} that is also {_adj(p4, v4)}."),
            medium=(f"At least one {_bare(p1, v1)} that is {_adj(p2, v2)} stands {_dir(d1)} every single "
                    f"{_bare(p2, v2)} that is also {_adj(p4, v4)} in the scene."),
            long=(f"There exists at least one {_bare(p1, v1)} which is {_adj(p2, v2)}  that, "
                  f"for every {_bare(p2, v2)} that is also {_adj(p4, v4)}  in the scene, stands {_dir(d1)} it."),
        )

    if variant == "2prop_neg":
        p1, v1, p2, v2, p3, v3, p4, v4, d1 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "P4'", "V4'", "D1'")
        return dict(
            short=(f"There exists {_quant('Some', p1, v1)} that is {_adj(p2, v2)} such that it is {_dir(d1)} {_quant('every', p2, v2)} that is {_not_adj(p4, v4)}."),
            medium=(f"At least one {_bare(p1, v1)} that is {_adj(p2, v2)} stands {_dir(d1)} every single "
                    f"{_bare(p2, v2)} that is {_not_adj(p4, v4)} in the scene."),
            long=(f"There exists at least one {_bare(p1, v1)} which is {_adj(p2, v2)}  that, "
                  f"for every {_bare(p2, v2)} that is {_not_adj(p4, v4)}  in the scene, stands {_dir(d1)} it."),
        )
    return None

def _c7(variant: str, a: dict) -> dict[str, str]:
    
    if variant == "1prop_propRel":
        p1, v1, p2, v2, d1 = _g(a, "P1'", "V1'", "P2'", "V2'", "D1'")

        return dict( 
            short=(f"{_quant('Every', p1, v1)} has {_obj(p2, v2)} {_dir(d1)} it."),
            medium=(f"For each {_bare(p1, v1)} in the scene, there must exist "
                    f"at least one {_bare(p2, v2)} standing {_dir(d1)} it."),
            long=(f"Every object that is {_adj(p1, v1)} is required to have a witness: "
                  f"{_obj(p2, v2)} that stands {_dir(d1)} it."),
        )

    if variant == "1prop_propRel_neg":
        p1, v1, p2, v2, d1 = _g(a, "P1'", "V1'", "P2'", "V2'", "D1'")

        return dict(
            short=(f"{_not_quant('Every', p1, v1)} has {_obj(p2, v2)} {_dir(d1)} it."),
            medium=(f"For each {_not_bare(p1, v1)} in the scene, there must be at least one "
                    f"{_bare(p2, v2)} standing {_dir(d1)} it."),
            long=(f"Every object that is {_not_adj(p1, v1)} is required to have a witness: {_obj(p2, v2)} "
                  f"that stands {_dir(d1)} it."),
        )

    
    if variant == "propRel_propRel":
        p1, v1, p2, v2, p3, v3, d1, d2 = _g(a, "P1'", "V1'","P2'", "V2'","P3'", "V3'", "D1'", "D2'")
        return dict(
            short=(f" {_quant('Every', p1, v1)} that is {_dir(d1)} {_obj(p2, v2)} "
                   f"has some {_bare(p3, v3)} {_dir(d2)} it."),
            medium=(f"Every object that is {_adj(p1, v1)} and is {_dir(d1)} {_obj(p2, v2)}, has a certain "
                    f"{_bare(p3, v3)} {_dir(d2)} it."),
            long=(f"Whenever there is an object that is {_adj(p1, v1)} and standing {_dir(d1)} {_obj(p2, v2)}, "
                  f"there must always exists some {_bare(p3, v3)} standing {_dir(d2)} it."),
        )

    
    if variant == "propRel_propRel_neg":
        p1, v1, p2, v2, p3, v3, d1, d2 = _g(a, "P1'", "V1'","P2'", "V2'","P3'", "V3'", "D1'", "D2'")
        return dict(
            short=(f" {_quant('Every', p1, v1)} that is {_dir(d1)} {_obj(p2, v2)} "
                   f"has some {_not_bare(p3, v3)} {_dir(d2)} it."),
            medium=(f"Every object that is {_adj(p1, v1)} and is {_dir(d1)} {_obj(p2, v2)}, has a certain "
                    f"{_not_bare(p3, v3)} {_dir(d2)} it."),
            long=(f"Whenever there is an object that is {_adj(p1, v1)} and standing {_dir(d1)} {_obj(p2, v2)}, "
                  f"there must always exists some {_not_bare(p3, v3)} standing {_dir(d2)} of it."),
        )

    if variant == "1prop_exact":
        p1, v1, p2, v2, d1, n = _g(a, "P1'", "V1'", "P2'", "V2'", "D1'", "N'")

        return dict( 
            short=(f"{_quant('Every', p1, v1)} has {_exactly(n)} {_bare(p2, v2) if n=='1' else _objs(p2, v2)} that {'is' if n=='1' else 'are'} {_dir(d1)} it."),
            medium=(f"For each {_bare(p1, v1)} in the scene, there must exist "
                    f"exactly {n} {'object' if n=='1' else 'objects'} that {'is' if n=='1' else 'are'} {_adj(p2, v2)} standing {_dir(d1)} it."),
            long=(f"Every object that is {_adj(p1, v1)} is required to have {_exactly(n)} {'witness' if n=='1' else 'witnesses'}: "
                  f"A witness is {_obj(p2, v2)} standing {_dir(d1)} it."),
        )

    if variant == "1prop_exact_neg":
        p1, v1, p2, v2, d1, n = _g(a, "P1'", "V1'", "P2'", "V2'", "D1'", "N'")

        return dict( 
            short=(f"{_quant('Every', p1, v1)} has {_exactly(n)} {_not_bare(p2, v2) if n=='1' else _not_objs(p2, v2)} that {'is' if n=='1' else 'are'} {_dir(d1)} it."),
            medium=(f"For each {_bare(p1, v1)} in the scene, there must exist "
                    f"exactly {n} {'object' if n=='1' else 'objects'} that {'is' if n=='1' else 'are'} {_not_adj(p2, v2)} standing {_dir(d1)} it."),
            long=(f"Every object that is {_adj(p1, v1)} is required to have {_exactly(n)} {'witness' if n=='1' else 'witnesses'}: "
                  f"A witness is {_not_obj(p2, v2)} standing {_dir(d1)} it."),
        )
        
    return None


def _c8(variant: str, a: dict) -> dict[str, str]:
    
    if variant == "1prop_atleast":
        p1, v1, n = _g(a, "P1'", "V1'", "N'")
        return dict(
            short=f"There {'is' if n=='1' else 'are'} {_at_least(n)} {_bare(p1, v1) if n=='1' else _objs(p1, v1)} in the scene.",
            medium=f"The scene must contain {_at_least(n)} {_bare(p1, v1) if n=='1' else _objs(p1, v1)}.",
            long=(f"The following cardinality constraint is to be satisfied by the scene generated: The count of objects that are {_adj(p1, v1)} in the scene must be "
                  f"at least {n}."),
        )
    
    if variant == "1prop_atleast_neg":
        p1, v1, n = _g(a, "P1'", "V1'", "N'")

        return dict(
            short=f"There {'is' if n == '1' else 'are'} {_at_least(n)} {_not_bare(p1, v1) if n=='1' else _not_objs(p1, v1)} in the scene.",
            medium=f"The scene must contain {_at_least(n)} {_n_obj(n)} that {'is' if n == '1' else 'are'} {_not_adj(p1, v1)}.",
            long=(f"The following cardinality constraint is to be satisfied by the scene generated: At least {n} {_n_obj(n)} in the scene must not be {_adj(p1, v1)}."),
        )

    
    if variant == "1prop_exact":
        p1, v1, n = _g(a, "P1'", "V1'", "N'")
        return dict(
            short=f"There {'is' if n=='1' else 'are'} {_exactly(n)} {_bare(p1, v1) if n=='1' else _objs(p1, v1)} in the scene.",
            medium=f"The scene must contain {_exactly(n)} {_bare(p1, v1) if n=='1' else _objs(p1, v1)}.",
            long=(f"The following cardinality constraint is to be satisfied by the scene generated: The count of objects that are {_adj(p1, v1)} in the scene must reach "
                  f"exactly {n}."),
        )
    
    if variant == "1prop_exact_neg":
        p1, v1, n = _g(a, "P1'", "V1'", "N'")

        return dict(
            short=f"There {'is' if n == '1' else 'are'} {_exactly(n)} {_not_bare(p1, v1) if n=='1' else _not_objs(p1, v1)} in the scene.",
            medium=f"The scene must contain {_exactly(n)} {_n_obj(n)} that {'is' if n == '1' else 'are'} {_not_adj(p1, v1)}.",
            long=(f"The following cardinality constraint is to be satisfied by the scene generated: Exactly {n} {_n_obj(n)} in the scene must not be {_adj(p1, v1)}."),
        )
    
    if variant == "1prop_atmost":
        p1, v1, n = _g(a, "P1'", "V1'", "N'")
        return dict(
            short=f"There {'is' if n=='1' else 'are'} {_at_most(n)} {_bare(p1, v1) if n=='1' else _objs(p1, v1)} in the scene.",
            medium=f"The scene must contain {_at_most(n)} {_bare(p1, v1) if n=='1' else _objs(p1, v1)}.",
            long=(f"The following cardinality constraint is to be satisfied by the scene generated: The count of objects that are {_adj(p1, v1)} in the scene must not exceed "
                  f"{n}."),
        )
    
    if variant == "1prop_atmost_neg":
        p1, v1, n = _g(a, "P1'", "V1'", "N'")

        return dict(
            short=f"There {'is' if n == '1' else 'are'} {_at_most(n)} {_not_bare(p1, v1) if n=='1' else _not_objs(p1, v1)} in the scene.",
            medium=f"The scene must contain {_at_most(n)} {_n_obj(n)} that {'is' if n == '1' else 'are'} {_not_adj(p1, v1)}.",
            long=(f"The following cardinality constraint is to be satisfied by the scene generated: Not more than {n} {_n_obj(n)} in the scene can be {_not_adj(p1, v1)}."),
        )
    
    
    if variant in ("1prop_exact_relational_neg", "1prop_atleast_relational_neg",
                   "1prop_atmost_relational_neg"):
        p1, v1, p2, v2, d1 = _g(a, "P1'", "V1'", "P2'", "V2'", "D1'")
        op = ("exactly" if "exact" in variant
              else "at least" if "atleast" in variant else "at most")
        return dict(
            short=(f"There {'is' if n=='1' else 'are'} {op} {n} {_bare(p1, v1) if n=='1' else _objs(p1, v1)} that {'does' if n=='1' else 'do'} not "
                   f"stand {_dir(d1)} any {_bare(p2, v2)}."),
            medium=(f"The scene has {op} {n} {_bare(p1, v1) if n=='1' else _objs(p1, v1)} that "
                    f"{'does' if n=='1' else 'do'} not stand {_dir(d1)} any {_bare(p2, v2)}."),
            long=(f"The scene must satisfy the following cardinality constraint. The number of {_objs(p1, v1)} that fail to stand {_dir(d1)} "
                  f"any {_bare(p2, v2)} must be {op} {n}."),
        )

    if variant in ("1prop_exact_relational", "1prop_atleast_relational",
                   "1prop_atmost_relational"):
        p1, v1, p2, v2, d1 = _g(a, "P1'", "V1'", "P2'", "V2'", "D1'")
        op = ("exactly" if "exact" in variant
              else "at least" if "atleast" in variant else "at most")
        return dict(
            short=(f"There {'is' if n=='1' else 'are'} {op} {n} {_bare(p1, v1) if n=='1' else _objs(p1, v1)} that {'is' if n=='1' else 'are'} "
                   f"standing {_dir(d1)} any {_bare(p2, v2)}."),
            medium=(f"The scene has {op} {n} {_bare(p1, v1) if n=='1' else _objs(p1, v1)} that "
                    f"{'is' if n=='1' else 'are'} standing {_dir(d1)} any {_bare(p2, v2)}."),
            long=(f"The scene must satisfy the following cardinality constraint. The number of {_objs(p1, v1)} that stand {_dir(d1)} "
                  f"any {_bare(p2, v2)} must be {op} {n}."),
        )
    
    
    
    if variant == "2prop_exact":
        p1, v1, p2, v2, n = _g(a, "P1'", "V1'", "P2'", "V2'","N'")
        return dict(
            short=f"There {'is' if n=='1' else 'are'} {_exactly(n)} {_n_obj(n)} that {'is' if n=='1' else 'are'} {_adj(p1, v1) and _adj(p2, v2)}.",
            medium=f"The scene must contain {_exactly(n)} {_n_obj(n)} that {'is' if n=='1' else 'are'} {_adj(p1, v1) and _adj(p2, v2)} in the scene.",
            long=(f"The following cardinality constraint is to be satisfied by the scene generated: The count of objects that are {_adj(p1, v1)} and {_adj(p2, v2)} simultaneously in the scene must reach "
                  f"exactly {n}."),
        )
    
    if variant == "2prop_exact_neg":
        p1, v1, p2, v2, n = _g(a, "P1'", "V1'", "P2'", "V2'","N'")
        return dict(
            short=f"There {'is' if n=='1' else 'are'} {_exactly(n)} {_n_obj(n)} that {'is' if n=='1' else 'are'} {_adj(p1, v1) and _not_adj(p2, v2)}.",
            medium=f"The scene must contain {_exactly(n)} {_n_obj(n)} that {'is' if n=='1' else 'are'} {_adj(p1, v1) and _not_adj(p2, v2)} in the scene.",
            long=(f"The following cardinality constraint is to be satisfied by the scene generated: The count of objects that are {_adj(p1, v1)} and {_not_adj(p2, v2)} simultaneously in the scene must reach "
                  f"exactly {n}."),
        )
    
    if variant == "2prop_atleast":
        p1, v1, p2, v2, n = _g(a, "P1'", "V1'", "P2'", "V2'","N'")
        return dict(
            short=f"There {'is' if n=='1' else 'are'} {_at_least(n)} {_n_obj(n)} that {'is' if n=='1' else 'are'} {_adj(p1, v1) and _adj(p2, v2)}.",
            medium=f"The scene must contain {_at_least(n)} {_n_obj(n)} that {'is' if n=='1' else 'are'} {_adj(p1, v1) and _adj(p2, v2)} in the scene.",
            long=(f"The following cardinality constraint is to be satisfied by the scene generated: The count of objects that are {_adj(p1, v1)} and {_adj(p2, v2)} simultaneously in the scene must be "
                  f"at least {n}."),
        )
    
    if variant == "2prop_atleast_neg":
        p1, v1, p2, v2, n = _g(a, "P1'", "V1'", "P2'", "V2'","N'")
        return dict(
            short=f"There {'is' if n=='1' else 'are'} {_at_least(n)} {_n_obj(n)} that {'is' if n=='1' else 'are'} {_adj(p1, v1) and _not_adj(p2, v2)}.",
            medium=f"The scene must contain {_at_least(n)} {_n_obj(n)} that {'is' if n=='1' else 'are'} {_adj(p1, v1) and _not_adj(p2, v2)} in the scene.",
            long=(f"The following cardinality constraint is to be satisfied by the scene generated: The count of objects that are {_adj(p1, v1)} and {_not_adj(p2, v2)} simultaneously in the scene must be "
                  f"at least {n}."),
        )
    
    if variant == "2prop_atmost":
        p1, v1, p2, v2, n = _g(a, "P1'", "V1'", "P2'", "V2'","N'")
        return dict(
            short=f"There {'is' if n=='1' else 'are'} {_at_most(n)} {_n_obj(n)} that {'is' if n=='1' else 'are'} {_adj(p1, v1) and _adj(p2, v2)}.",
            medium=f"The scene must contain {_at_most(n)} {_n_obj(n)} that {'is' if n=='1' else 'are'} {_adj(p1, v1) and _adj(p2, v2)} in the scene.",
            long=(f"The following cardinality constraint is to be satisfied by the scene generated: The count of objects that are {_adj(p1, v1)} and {_adj(p2, v2)} simultaneously in the scene can not exceed "
                  f"{n}."),
        )
    
    if variant == "2prop_atmost_neg":
        p1, v1, p2, v2, n = _g(a, "P1'", "V1'", "P2'", "V2'","N'")
        return dict(
            short=f"There {'is' if n=='1' else 'are'} {_at_most(n)} {_n_obj(n)} that {'is' if n=='1' else 'are'} {_adj(p1, v1) and _not_adj(p2, v2)}.",
            medium=f"The scene must contain {_at_most(n)} {_n_obj(n)} that {'is' if n=='1' else 'are'} {_adj(p1, v1) and _not_adj(p2, v2)} in the scene.",
            long=(f"The following cardinality constraint is to be satisfied by the scene generated: The count of objects that are {_adj(p1, v1)} and {_not_adj(p2, v2)} simultaneously in the scene can not exceed "
                  f"{n}."),
        )
    
    return None


def _c9(variant: str, a: dict) -> dict[str, str]:
    
    if variant == "1prop":
        p1, v1, p2, v2 = _g(a, "P1'", "V1'", "P2'", "V2'")

        return dict(
            short=(f"The count of {_objs(p1, v1)} equals the count of {_objs(p2, v2)}."),
            medium=(f"The number of objects that are {_adj(p1, v1)} in the scene must equal "
                    f"the number of objects that are {_adj(p2, v2)}."),
            long=(f"The scene enforces a balance: the cardinality of objects thar are {_adj(p1, v1)} "
                  f"must exactly match the cardinality of objects that are {_adj(p2, v2)} "),
        )

    if variant == "2prop":
        p1, v1, p2, v2, p3, v3, p4, v4 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "P4'", "V4'")
        return dict(
            short=(f"The count of objects that are both ({_adj(p1, v1)} and {_adj(p2, v2)}) equals "
                   f"the count of objects that are both ({_adj(p3, v3)} and {_adj(p4, v4)})."),
            medium=(f"The number of objects that are {_adj(p1, v1)} and {_adj(p2, v2)} simultaneously "
                    f"must equal the number of objects that are both {_adj(p3, v3)} and {_adj(p4, v4)}."),
            long=(f"The scene requires equal cardinality between two groups: the first group being the collection of objects that are both "
                  f"{_adj(p1, v1)} and {_adj(p2, v2)}, and the second group being the collection of objects that are both "
                  f"{_adj(p3, v3)} and {_adj(p4, v4)}."),
        )

    if variant == "2prop_mix":
        p1, v1, p2, v2, p3, v3, p4, v4 = _g(a, "P1'", "V1'", "P2'", "V2'", "P3'", "V3'", "P4'", "V4'")
        return dict(
            short=(f"The count of object that are ({_not_adj(p1, v1)} but are {_adj(p2, v2)}) equals "
                   f"the count of objects that are ({_not_adj(p3, v3)} but {_adj(p4, v4)})."),
            medium=(f"The number of objects that are {_not_adj(p1, v1)} but {_adj(p2, v2)} "
                    f"must equal the number that are {_not_adj(p3, v3)} but {_adj(p4, v4)}."),
            long=(f"The scene balances two groups: the first group is the collection of objects that are not "
                  f"{_adj(p1, v1)} but  {_adj(p2, v2)}, and the second group is the collection of objects that are not "
                  f"{_adj(p3, v3)} but {_adj(p4, v4)}."),
        )

    return None


# ── Dispatch table ──────────────────────────────────────────────────────────

_HANDLERS = {
    "C1": _c1, "C2": _c2, "C3": _c3, "C4": _c4,
    "C5": _c5, "C6": _c6, "C7": _c7, "C8": _c8, "C9": _c9,
}


# ── Public API ──────────────────────────────────────────────────────────────

def verbalize(template_stem: str, assignment: dict[str, str]) -> dict[str, str]:
    """
    Generate short, medium, and long NL descriptions for an instantiated constraint.

    Args:
        template_stem: Filename stem without extension, e.g. 'C1_2prop_neg'.
        assignment:    Completed placeholder dict, e.g. {"P1'": "color", "V1'": "red"}.

    Returns:
        {'short': ..., 'medium': ..., 'long': ...}
    """
    parts = template_stem.split("_", 1)
    cls = parts[0]  # 'C1', 'C2', …
    variant = parts[1] if len(parts) > 1 else ""
    handler = _HANDLERS.get(cls)
    if handler:
        return handler(variant, assignment)
    