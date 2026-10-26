# Nair 2013: complete Malaveda/Veda chapter candidate

The source is Ravi Sankar S. Nair, *Tribal Languages of Kerala*, Language in
India 13(7), July 2013, printed/PDF pp. 40–44. The pinned 58-page original is
`tmp/pdfs/tribal-languages-kerala-nair2013.pdf`, SHA256
`4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af`.
The source states no explicit reuse licence. Only extracted linguistic facts,
audits and scripts are proposed for installation; the source-page images are
local review artifacts.

All five chapter pages were inventoried and visually reread. The 57 physical
blocks contain 238 form witnesses: 223 target witnesses, ten expressly labelled
Malayalam/Tamil comparisons, and five documentary community labels. All target
witnesses produce separate stable-key rows. Controls and documentary labels are
explicitly excluded in the audit. Two additional audit records preserve
phonological prose and the six reported theses with no form witness.

The native text layer is usable with known legacy dot/tilde font decoding.
Fourteen independent drawn lowlines were located geometrically and represented
with U+0332. No OCR or IPA conversion is used. Literal dashes, separator spaces,
breves, macrons and doubled vowels remain. Case/number labels move from lexical
glosses into tags. Printed lexical/gloss oddities remain uncertain rather than
being silently corrected. Conjectural copula and lexical comparisons remain
source commentary; no ancestry or borrowing edge is inferred.

The proposed base is source-named Malaveda, with a source-qualified Veda lect.
Glottocode and coordinates remain blank. Nair explicitly distinguishes
Pathanamthitta 1977 speech from Kollam 1991 speech and warns that the named
communities may differ. The examples themselves are not individually located.
See `metadata/identity-decision.json` and the independent identity review.

`prepare_candidate.py` produces `proposed.csv`, per-witness
`proposal-audit.jsonl`, symbol inventory and twelve independently reviewed
source-explicit variant links. Eight source-supported case-pronoun/kinship tag
repairs and those twelve relation fields are the only CSV changes from the first
independently reviewed candidate; `review-repair-continuity.json` proves this.
`prepare_metadata.py` produces registry/settings/BibTeX proposals.
`validate_candidate.py` checks complete counts, keys, scoped grammar, literal
edge cases, Unicode, all fifteen fields through Jambu's actual parser, and all
seven formatted references. These scripts operate only in this source directory.

This package is a **source-local candidate**, not an accepted or installed
ingestion. The first independent sample passed all literal/gloss/scope checks
but found a missing grammatical tag class; repairs now await a targeted reread
and fresh disjoint output sample. Guarded installation and canonical focused
checks remain pending. The full CLDF
pipeline, compiled graph/identity verification and full suite are deferred under
the user's current instructions. Browser refresh/QA awaits an explicit request.
