# Manual audit: items 106-110

The repository ingestion checklist, survey/comparative-table addendum, PDF
inspection policy, and strict rendered-page manual-review policy controlled
this block. Every one of the 80 conceptual cells was independently read by eye
from physical pp.50-51 / printed pp.45-46 at 400 dpi and rechecked in tight
900/1200-dpi crops. PDF text, OCR, and the legacy CSV did not supply, complete,
normalize, infer, correct, or verify any transcription.

## Accounting

- Items: 106 `mother`, 107 `older brother`, 108 `younger brother`, 109 `older sister`, 110 `younger sister`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 76; source blanks: 4; ambiguous: 0; illegible: 0.
- Expanded occurrences: 84 (77 target candidates; 7 controls).
- Item 107 crosses physical pp.50-51 / printed pp.45-46.
- Item 110 crosses the left/right column boundary on physical p.51.

Every form, repeated response, group label, page/column coordinate, and cell
boundary was visually rechecked against the rendered source. CCC is absent
from the complete item-106, item-107, item-109, and item-110 blocks, so those
coordinates are explicit source blanks. Repeated responses remain expanded:
item 106 retains two Hindi and two BNM forms, item 107 retains two Hindi and
two DkR forms, item 109/BNT retains its group-1/group-3 forms, and item 110
retains grouped RNS, DGC, and KkP responses. Tight crops preserve item 107's
`dʌda`/`dada` contrast, item 108/CCC `ʌbaⁱja`, item 109/RKB and SkP `dɪdi`,
and item 110's source dotted `i`.

The independent ledger was frozen before legacy comparison. During the
post-comparison source-image audit, targeted 1200-dpi rechecks resolved the
source-visible item-107 vowel contrast, item-108 superscript, and item-109
`ɪ` glyphs without accepting legacy strings as evidence. The final
deterministic ledger SHA-256 is
`ee8e8d41deb7060208b07c3f608099edb60bd05eb1f9f0f753ca35d269f6469a`.

## Unresolved site identities

No lexical reading is ambiguous or illegible. Ten coordinates remain
medium-confidence site assignments because both locality lists print as `RNS`.
Item 110 has one group-1 RNS row and two group-2 RNS rows; under the documented
occurrence-order rule, the sole group-1 and first group-2 responses are
provisionally assigned together to Sisaikhara, while the second group-2 row
maps to Sisana. Every physical/printed page, item, site key, column, visible
response description, and candidate locality is enumerated in
`../unresolved_readings.tsv`.

## Post-entry reconciliation

Only after the independent ledger was frozen did the guarded importer compare
reviewed target occurrences with `20230530-tharu2.csv`. The block has 77
manual and 77 legacy target occurrences: 73 agree exactly, leaving four
manual-only and four legacy-only multiset occurrences. The rendered source
retains item 110/DGC `bʌhʌnija`, DkR `vʌhʌnija`, RkM `bʌjinʌja`, and SkP
`bʌhni`, all with dotted `i` where the legacy file has `ɪ`. Every difference
was rechecked visually after the comparison. The legacy data were never
accepted as transcription evidence.

Cumulatively through item 110, 1,595/1,785 legacy target occurrences agree
exactly; the multiset retains 212 manual-only and 190 legacy-only occurrences.
Staging remains refused at 1,760/3,360 reviewed cells. Item 111 `son`, physical
p.51 / printed p.46, is next.
