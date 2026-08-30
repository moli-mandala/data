# Manual audit: items 76-80

The repository ingestion checklist, survey/comparative-table addendum, PDF
inspection policy, and strict rendered-page manual-review policy controlled
this block. Every one of the 80 conceptual cells was independently read by eye
from physical pp.45-46 / printed pp.40-41 at 400 dpi and rechecked in tight
900-dpi crops. PDF text, OCR, and the legacy CSV did not supply, complete,
normalize, infer, correct, or verify any transcription.

## Accounting

- Items: 76 `turmeric`, 77 `garlic`, 78 `onion`, 79 `cauliflower`, 80 `tomato`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 78; explicit source blanks: 2; ambiguous: 0; illegible: 0.
- Expanded occurrences: 103 (96 target candidates; 7 controls).
- Source blanks: item 76/CCC and item 79/CCC.
- Item 78 crosses the left/right column boundary; item 80 crosses physical
  pp.45-46 / printed pp.40-41.

Every form, repeated response, group label, page/column coordinate, and cell
boundary was visually rechecked against the rendered source. Item 78 contains
41 occurrences: eleven cells each preserve three printed responses from
groups 1-3, BNM preserves a fourth group-4 response, and SkP, CCC, KkP, and
BNT each preserve one response. The source-visible distinctions `piadʒu`
(CCC), `pedʒ` (KkP), and `ɡʌɳʈʰi` (BNT/BNM) remain literal.

Tight crops preserve source dotted `i` in item 76/RKB and item 80/RKB,
DKS, and KkP, even where the legacy file differs. They also preserve the
retroflex stops in item 80, source superscript aspiration in BNT
`ʈʌmʈʌmbʰʌʈa`, and plain `n` in CCC `ɾambʰʌnʈa`. All strings are NFC. The
deterministic ledger SHA-256 is
`ded5f015247d708d4e9a1e2974967d482862d653be2b82dd74ea8932fe83073a`.

## Unresolved site identities

No lexical reading is ambiguous or illegible. Ten coordinates remain
medium-confidence site assignments because both locality lists print as `RNS`.
Every physical/printed page, item, site key, column, visible response
description, and candidate locality is enumerated in
`../unresolved_readings.tsv`.

## Post-entry reconciliation

Only after the independent ledger was frozen did the guarded importer compare
reviewed target occurrences with `20230530-tharu2.csv`. The block has 96 manual
and 96 legacy target occurrences: 91 agree exactly, leaving five paired
multiset differences. Four retain source dotted `i` where the legacy file has
small-capital `ɪ` (item 76/RKB and item 80/RKB, DKS, KkP); the fifth retains
source superscript `ʰ` in item 80/BNT where the legacy file has plain `h`.
All five were rechecked visually after the comparison. The legacy data were
never accepted as transcription evidence.

Cumulatively through item 80, 1,176/1,288 legacy target occurrences agree
exactly; the multiset retains 132 manual-only and 112 legacy-only occurrences.
Staging remains refused at 1,280/3,360 reviewed cells. Item 81 `cabbage`,
physical p.46 / printed p.41, is next.
