# Interlinear review proposal

The immutable original PDF is the authority. All sixteen pages (PDF 93–108,
printed 83–98) were visually reviewed. Per-page render hashes are retained; only
one working render is stored because accumulating sixteen images was rejected
under the disk-space restriction.

`whole-expression-proposal.jsonl` accounts for all 126 phonetic blocks as 97
narrative expressions and ten target questions, plus ten separately printed
target answers. C4 n3 has no free translation. C4 n20's visible bare `g` marker
is separated from the phonetic form by an explicit editorial overlay; the
original 637-event scaffold remains unchanged for comparison.

`c5-column-proposal.jsonl` preserves all 31 column-aligned phonetic/gloss pairs
as 192 child/group candidates. Two internal spans remain unresolved. The
literal source column pairing is preserved even when the resulting English
association seems surprising; no modern-language correction is substituted.

`continuous-semantic-proposal.jsonl` is a manually partitioned proposal for all
85 continuous-gloss blocks in C2–C4. It does not zip whitespace tokens. Each
partition is checked against the exact complete target and gloss sequences.
Single-token proposals require independent semantic review. Unresolved
multi-token spans retain their raw English analysis with blank lexical Gloss;
they do not invent individual word correspondences. The containing whole
expression remains fully represented.

Only explicit source parentheticals `(past)`, `(present)`, and `(future)` are
structured as tense tags in this proposal. No verbal POS is inferred merely
from tense or English wording. Unknown `cm` and `?` analyses remain literal,
typed unresolved evidence with blank lexical glosses. Repeated occurrences
retain distinct physical keys. No component, borrowing, or variant edges are
invented. No sound conversion is applied.

Reproduction uses `prepare_whole_expressions.py`, `prepare_c5_columns.py`,
`inspect_continuous.py`, and `prepare_continuous_alignment.py`. These are
source-local proposal generators, not canonical installers. Their checks cover
token conservation, exact keys, physical scope, and manual partition structure;
they do not replace independent scholarly review.
