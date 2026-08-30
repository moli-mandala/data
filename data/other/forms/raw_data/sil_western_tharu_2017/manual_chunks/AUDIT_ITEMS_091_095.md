# Manual audit: items 91-95

The repository ingestion checklist, survey/comparative-table addendum, PDF
inspection policy, and strict rendered-page manual-review policy controlled
this block. Every one of the 80 conceptual cells was independently read by eye
from physical pp.47-48 / printed pp.42-43 at 400 dpi and rechecked in tight
900-dpi crops. PDF text, OCR, and the legacy CSV did not supply, complete,
normalize, infer, correct, or verify any transcription.

## Accounting

- Items: 91 `milk`, 92 `horns`, 93 `tail`, 94 `goat`, 95 `dog`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 80; source blanks: 0; ambiguous: 0; illegible: 0.
- Expanded occurrences: 85 (79 target candidates; 6 controls).
- Item 91 crosses physical pp.47-48 / printed pp.42-43.
- Item 94 crosses the left/right column boundary on physical p.48.

Every form, repeated response, group label, page/column coordinate, and cell
boundary was visually rechecked against the rendered source. Item 92/DDK
preserves group-1 `sĩŋ` and group-2 `kãʈa`; item 93/HIN preserves group-1
`pũtʃʰ` and group-2 `ɖum`. Item 95 retains repeated identical group-2/group-3
responses under DGC, DkR, and CCC. Tight crops preserve source dotted `i`,
nasal vowels, item 94's flap distinctions, and item 95/RKM retroflex `ʈ`.
All strings are NFC. The deterministic ledger SHA-256 is
`7232f7a53d12176664fea1b58ed5cf20206cea0d4cac000b431510229015bb33`.

## Unresolved site identities

No lexical reading is ambiguous or illegible. Ten coordinates remain
medium-confidence site assignments because both locality lists print as `RNS`.
Every physical/printed page, item, site key, column, visible response
description, and candidate locality is enumerated in
`../unresolved_readings.tsv`.

## Post-entry reconciliation

Only after the independent ledger was frozen did the guarded importer compare
reviewed target occurrences with `20230530-tharu2.csv`. The block has 79 manual
and 79 legacy target occurrences: 50 agree exactly, leaving twenty-nine
manual-only and twenty-nine legacy-only multiset occurrences. Twenty-six pairs
retain source `i` where the legacy file has `ɪ`; item 94/DGC also preserves
source flap `ɾ` where legacy has `ɽ`. Item 94/DDK independently preserves the
same source flap contrast, item 94/RKM preserves source `bʌkʌɾja`, and item
95/RKM preserves source retroflex `kuʈːa`. Every difference was rechecked
visually after the comparison. The legacy data were never accepted as
transcription evidence.

Cumulatively through item 95, 1,360/1,530 legacy target occurrences agree
exactly; the multiset retains 191 manual-only and 170 legacy-only occurrences.
Staging remains refused at 1,520/3,360 reviewed cells. Item 96 `snake`, physical
p.48 / printed p.43, is next.
