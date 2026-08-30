# Manual audit: items 86-90

The repository ingestion checklist, survey/comparative-table addendum, PDF
inspection policy, and strict rendered-page manual-review policy controlled
this block. Every one of the 80 conceptual cells was independently read by eye
from physical p.47 / printed p.42 at 400 dpi and rechecked in tight 900-dpi
crops. PDF text, OCR, and the legacy CSV did not supply, complete, normalize,
infer, correct, or verify any transcription.

## Accounting

- Items: 86 `fish`, 87 `chicken`, 88 `egg`, 89 `cow`, 90 `buffalo`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 79; explicit source blanks: 1; ambiguous: 0; illegible: 0.
- Expanded occurrences: 80 (74 target candidates; 6 controls).
- Source blank: item 87/CCC.
- Item 88 crosses the left/right column boundary at its final DKS row.

Every form, repeated response, group label, page/column coordinate, and cell
boundary was visually rechecked against the rendered source. Item 90/HIN
preserves two separately printed group-1 responses, `bʰæs / bʰæ̃s`. Tight
crops preserve item 86's aspiration-plus-length sequence `ʰː`, item 87's
dotted `i`, item 88's retroflex nasal/stop/flap distinctions, item 89/SkP
`ɡʌjã` without an inferred length mark, and item 90/KkP superscript `ⁱ`.
All strings are NFC. The deterministic ledger SHA-256 is
`b77465e4bb3a093df92a2072a5d1c8bc42e2d440b2a49c6b39b26ef0d0ee5a8d`.

## Unresolved site identities

No lexical reading is ambiguous or illegible. Ten coordinates remain
medium-confidence site assignments because both locality lists print as `RNS`.
Every physical/printed page, item, site key, column, visible response
description, and candidate locality is enumerated in
`../unresolved_readings.tsv`.

## Post-entry reconciliation

Only after the independent ledger was frozen did the guarded importer compare
reviewed target occurrences with `20230530-tharu2.csv`. The block has 74 manual
and 74 legacy target occurrences: 58 agree exactly, leaving sixteen
manual-only and sixteen legacy-only multiset occurrences. Eleven pairs retain
source dotted `i` where the legacy file has small-capital `ɪ` (two in item 87
and nine in item 90). Item 88 contributes three source-visible retroflex and
nasal-vowel differences. Item 89/SkP retains source `ɡʌjã` against legacy
`ɡʌjːã`, and item 90/CCC retains source `bʰæsi` against legacy `bʰæːsi`.
Every difference was rechecked visually after the comparison. The legacy data
were never accepted as transcription evidence.

Cumulatively through item 90, 1,310/1,451 legacy target occurrences agree
exactly; the multiset retains 162 manual-only and 141 legacy-only occurrences.
Staging remains refused at 1,440/3,360 reviewed cells. Item 91 `milk`, physical
p.47 / printed p.42, is next.
