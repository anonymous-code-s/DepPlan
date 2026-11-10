# Semantic-unit role rendering

The linearizer identifies a core argument by its predicate frame and argument
index, as in `person fills the ARG1 role of frame predicate-01.` It retains the
predicate sense suffix and expresses core arguments through frame-specific role
identifiers. We use this representation because the meaning of each ARG index
depends on the predicate frame. Roles above ARG5 use the same pattern.

An inverse edge `(a, :ARG1-of, p)` renders the same role assignment as
`(p, :ARG1, a)`: `a` fills ARG1 of the frame at `p`. The recorded triples retain
their original direction. One-hop context is collected by AMR variable identity;
two nodes with equal concept labels do not become the same node. Unnamed repeated
concepts retain their variable IDs in the rendered text, and compact prompts
preserve the context links as unit IDs.

Entity-name, possession, location, time, negation and other relation templates
verbalize their relations directly. Predicate labels retain their sense suffix
in those relations as well, so every unit carries the predicate sense encoded by
the graph. The template library Ω is deterministic: each instantiated triple and
its context map to exactly one unit text, and repeated runs over the same graph
produce identical units and identical numbering. Each output records the renderer
identity `frame-indexed-roles-with-variable-aware-one-hop-context-v2`.
