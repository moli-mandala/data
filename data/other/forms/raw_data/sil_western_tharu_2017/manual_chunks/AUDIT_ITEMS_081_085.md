# Manual audit: items 81-85

The repository ingestion checklist, survey/comparative-table addendum, PDF
inspection policy, and strict rendered-page manual-review policy controlled
this block. Every one of the 80 conceptual cells was independently read by eye
from physical pp.46-47 / printed pp.41-42 at 400 dpi and rechecked in tight
900-dpi crops. PDF text, OCR, and the legacy CSV did not supply, complete,
normalize, infer, correct, or verify any transcription.

## Accounting

- Items: 81 `cabbage`, 82 `oil`, 83 `salt`, 84 `meat`, 85 `fat`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 77; explicit source blanks: 3; ambiguous: 0; illegible: 0.
- Expanded occurrences: 96 (90 target candidates; 6 controls).
- Source blanks: item 81/CCC, item 84/TkN, and item 85/CCC.
- Item 83 crosses the left/right column boundary; item 85 crosses physical
  pp.46-47 / printed pp.41-42.

Every form, repeated response, group label, qualifier, page/column coordinate,
and cell boundary was visually rechecked against the rendered source. Item 81
preserves the repeated DGC response and DDK's group-2 `ɡaɳʈʰɡobʰi`. Item 83
contains 30 occurrences: fourteen cells preserve responses from both groups 2
and 3, while HIN and SkP occur once. Item 84 retains RKB's literal `(small
piece)` qualifier on `buʈi`; the same form is unqualified under KkP.

Tight crops preserve item 82/DGC retroflex `ʈel`, the source's visibly dotted
`i` throughout item 84, item 84/KkP and RKB single retroflex `ʈ` in `buʈi`,
and item 85's nasal-vowel and retroflex distinctions. All strings are NFC. The
deterministic ledger SHA-256 is
`0def3ab32a53d2a7d3ae32d0ccd72dc98ab72851fbaa97760220918560b19ded`.

## Unresolved site identities

No lexical reading is ambiguous or illegible. Ten coordinates remain
medium-confidence site assignments because both locality lists print as `RNS`.
Every physical/printed page, item, site key, column, visible response
description, and candidate locality is enumerated in
`../unresolved_readings.tsv`.

## Post-entry reconciliation

Only after the independent ledger was frozen did the guarded importer compare
reviewed target occurrences with `20230530-tharu2.csv`. The block has 90 manual
and 89 legacy target occurrences: 76 agree exactly, leaving fourteen
manual-only and thirteen legacy-only multiset occurrences. Twelve paired
differences retain source dotted `i` in item 84 where the legacy file has
small-capital `ɪ`. Another pair retains source KkP `buʈi` where the legacy file
has `butːi`. The remaining manual-only occurrence is the source-visible
item-84/RKB `buʈi (small piece)`, which has no legacy counterpart. Every
difference was rechecked visually after the comparison. The legacy data were
never accepted as transcription evidence.

Cumulatively through item 85, 1,252/1,377 legacy target occurrences agree
exactly; the multiset retains 146 manual-only and 125 legacy-only occurrences.
Staging remains refused at 1,360/3,360 reviewed cells. Item 86 `fish`, physical
p.47 / printed p.42, is next.
