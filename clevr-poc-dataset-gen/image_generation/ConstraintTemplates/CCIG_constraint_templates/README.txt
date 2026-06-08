CCIG constraint templates — one ASP rule per file.

Naming: constraint_templates_L{level}_{family}.txt

Each file contains exactly one comment line + one ASP integrity constraint.

Hierarchy (L0–L7, 8 levels):

L0: Unary Attribute Constraints
- constraint_templates_L0_exist.txt
- constraint_templates_L0_forbid.txt

L1: Single Relational Constraint
- constraint_templates_L1_exist_pair.txt
- constraint_templates_L1_forbid_pair.txt

L2: Relational Composition
- constraint_templates_L2_chain2.txt   (2-hop chain: X1->X2->X3)
- constraint_templates_L2_chain3.txt   (3-hop chain: X1->X2->X3->X4)

L3: Conjunctive Relational Binding
- constraint_templates_L3_shared_hub.txt

L4: Implication / Universal Dependency  [merged from old L4 + L5]
- constraint_templates_L4_implication.txt        (forall x A(x)->exists y R(x,y))
- constraint_templates_L4_universal_witness.txt  (forall x A(x)->exists y(B(y) and R(y,x)))

L5: Relational Aggregates  [was L6]
- constraint_templates_L5_unary_count.txt
- constraint_templates_L5_relational_count.txt

L6: Injective Matching  [was L7; encodes exactly-one witness]
- constraint_templates_L6_witness_unique.txt

L7: Global Coupling  [was L8]
- constraint_templates_L7_count_coupling.txt
- constraint_templates_L7_all_different.txt

JSON prompt templates link via asp_template_file (see CLEVR_CCIG_templates/).
