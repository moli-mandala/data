# Manual audit: items 66-70

The repository ingestion checklist, survey/comparative-table addendum, PDF
inspection policy, and strict rendered-page manual-review policy controlled
this block. Every one of the 80 conceptual cells was independently read by eye
from physical pp.43-44 / printed pp.38-39 at 400 dpi and rechecked in 900-dpi
crops. PDF text, OCR, and the legacy CSV did not supply, complete, normalize,
infer, correct, or verify any transcription.

## Accounting

- Items: 66 `fruit`, 67 `mango`, 68 `banana`, 69 `wheat`, 70 `millet`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 77; explicit source blanks: 3; ambiguous: 0; illegible: 0.
- Expanded occurrences: 79 (73 target candidates; 6 controls).
- Source blanks: item 66/DDK, item 69/CCC, and item 70/RNS_Sisana.
- Item 70 crosses physical pp.43-44 / printed pp.38-39; the Hindi cell itself
  spans the break and retains responses from groups 1 and 4.

Every form, repeated response, group label, and cell coordinate was visually
rechecked against the rendered source. Tight crops preserve item 68's dotted
`i` in `tʃʰijã`/`tʃʰija`, BNM `ɡeɾkibʰʌɾi`, item 69's visibly rounded
`õ`, item 70/HIN `dʒʌvaɾ / dʒɔ`, DkR `dʒolʌɾi`, and source superscript
`ᵘ` in KkP `dʒoᵘ` and DKS `dʒaᵘ`. All strings are NFC. The deterministic
ledger SHA-256 is
`0dc8923453211b2d3d76cb8f0d4e451000683f50f3f9c56f5cc9f974f6c694d4`.

## Unresolved site identities

No lexical reading is ambiguous or illegible. Ten coordinates remain
medium-confidence site assignments because both locality lists print as `RNS`.
Item 70 has one RNS response in group 1 and one in group 2, with no second RNS
occurrence in either group. Under the standing occurrence-order policy, both
visible responses are provisionally assigned to metadata row 1 (Sisaikhara)
and row 2 (Sisana) is an explicit source blank. This is a site-identity issue,
not a guessed lexical reading. Every coordinate and visible-response
description is enumerated in `../unresolved_readings.tsv`.

## Post-entry reconciliation

Only after the independent ledger was frozen did the guarded importer compare
reviewed target occurrences with `20230530-tharu2.csv`. The block has 73 manual
and 73 legacy target occurrences: 63 agree exactly, leaving 10 paired multiset
differences. Rendered-source rechecks retain dotted `i` rather than legacy
small-capital `ɪ` in five banana occurrences, visible `õ` rather than legacy
`ʊ̃` in three wheat occurrences, and source superscript `ᵘ` rather than legacy
`ᶸ` in two millet occurrences. The legacy data were never accepted as
transcription evidence.

Cumulatively through item 70, 1,026/1,117 legacy target occurrences agree
exactly; the multiset retains 111 manual-only and 91 legacy-only occurrences.
Staging remains refused at 1,120/3,360 reviewed cells. Item 71 `salt`, physical
p.44 / printed p.39, left column, is next.
