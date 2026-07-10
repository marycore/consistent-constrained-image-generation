% NOTE: property value declarations and choice rules are generated from
% PROPERTIES in domain.py — do not hardcode them here.

% ── Region capacity (at most 3 objects per region) ────────────────────────
:- #count { X : object(X), hasProperty(X, region, R) } >= 4, property(region, R).

% ── Region spatial layout (2×2 grid) ─────────────────────────────────────
% reg_0=top-left  | reg_1=top-right
% reg_2=bot-left  | reg_3=bot-right
right_R(reg_1, reg_0). right_R(reg_3, reg_0).
right_R(reg_1, reg_2). right_R(reg_3, reg_2).
left_R(R1, R2) :- right_R(R2, R1).

front_R(reg_2, reg_0). front_R(reg_3, reg_0).
front_R(reg_2, reg_1). front_R(reg_3, reg_1).
behind_R(R1, R2) :- front_R(R2, R1).

% ── Spatial relationships derived from region placement ───────────────────
hasRelationship(X, Y, left)   :- object(X), object(Y), X != Y,
    hasProperty(X, region, R1), hasProperty(Y, region, R2), left_R(R1, R2).
hasRelationship(X, Y, right)  :- object(X), object(Y), X != Y,
    hasProperty(X, region, R1), hasProperty(Y, region, R2), right_R(R1, R2).
hasRelationship(X, Y, front)  :- object(X), object(Y), X != Y,
    hasProperty(X, region, R1), hasProperty(Y, region, R2), front_R(R1, R2).
hasRelationship(X, Y, behind) :- object(X), object(Y), X != Y,
    hasProperty(X, region, R1), hasProperty(Y, region, R2), behind_R(R1, R2).

% Same-region objects: disjunctive left/right and front/behind
hasRelationship(X, Y, left); hasRelationship(X, Y, right) :-
    object(X), object(Y), X != Y,
    hasProperty(X, region, R), hasProperty(Y, region, R).
hasRelationship(X, Y, front); hasRelationship(X, Y, behind) :-
    object(X), object(Y), X != Y,
    hasProperty(X, region, R), hasProperty(Y, region, R).

% Anti-symmetry
:- hasRelationship(X1, X2, R), hasRelationship(X2, X1, R), X1 != X2.

% Transitivity
hasRelationship(X1, X3, left)   :- hasRelationship(X1, X2, left),
    hasRelationship(X2, X3, left),   X1 != X2, X2 != X3.
hasRelationship(X1, X2, left)   :- hasRelationship(X2, X1, right).
hasRelationship(X1, X3, behind) :- hasRelationship(X1, X2, behind),
    hasRelationship(X2, X3, behind), X1 != X2, X2 != X3.
hasRelationship(X1, X2, behind) :- hasRelationship(X2, X1, front).
