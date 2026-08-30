# Manual audit: items 71-75

The repository ingestion checklist, survey/comparative-table addendum, PDF
inspection policy, and strict rendered-page manual-review policy controlled
this block. Every one of the 80 conceptual cells was independently read by eye
from physical p.44 / printed p.39 at 400 dpi and rechecked in 900-dpi crops.
PDF text, OCR, and the legacy CSV did not supply, complete, normalize, infer,
correct, or verify any transcription.

## Accounting

- Items: 71 `rice`, 72 `potato`, 73 `eggplant`, 74 `groundnut`, 75 `chili`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 78; explicit source blanks: 2; ambiguous: 0; illegible: 0.
- Expanded occurrences: 81 (75 target candidates; 6 controls).
- Source blanks: item 71/CCC and item 73/CCC.
- Item 73 crosses the left/right column boundary; its Hindi control is on the
  left and all printed target responses are on the right.

Every form, repeated response, group label, and cell coordinate was visually
rechecked against the rendered source. Item 71 prints RNK twice in group 1;
both responses are preserved in the RNK conceptual cell. Item 74 prints HIN
twice in group 1, and item 75 prints DGC twice in group 1; those repeated-code
responses are likewise preserved as alternatives rather than discarded.
Tight crops retain item 73/HIN `bæ̃ŋɡʌn`, RKB's single `ʈ` in `bʌʈa`,
item 74's aspiration and nasal-vowel distinctions, and item 75's visibly
dotted `i`. All strings are NFC. The deterministic ledger SHA-256 is
`c0f5c32befbeae9006a247fd9b08e67459a3c49c2a1d7b9a6ef57daa8f8bbdb5`.

## Unresolved site identities

No lexical reading is ambiguous or illegible. Ten coordinates remain
medium-confidence site assignments because both locality lists print as `RNS`.
Every physical/printed page, item, site key, column, visible response
description, and candidate locality is enumerated in
`../unresolved_readings.tsv`.

## Post-entry reconciliation

Only after the independent ledger was frozen did the guarded importer compare
reviewed target occurrences with `20230530-tharu2.csv`. The block has 75 manual
and 75 legacy target occurrences: 59 agree exactly, leaving 16 paired multiset
differences. Fifteen differences are item-75 source dotted `i` versus legacy
small-capital `ɪ`; the remaining difference retains item-73/RKB `bʌʈa`
without the legacy file's added length mark. All differing cells were rechecked
against the 900-dpi crop. The legacy data were never accepted as transcription
evidence.

Cumulatively through item 75, 1,085/1,192 legacy target occurrences agree
exactly; the multiset retains 127 manual-only and 107 legacy-only occurrences.
Staging remains refused at 1,200/3,360 reviewed cells. Item 76 `turmeric`,
physical p.45 / printed p.40, is next.
