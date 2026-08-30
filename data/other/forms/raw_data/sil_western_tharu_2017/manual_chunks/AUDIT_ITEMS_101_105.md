# Manual audit: items 101-105

The repository ingestion checklist, survey/comparative-table addendum, PDF
inspection policy, and strict rendered-page manual-review policy controlled
this block. Every one of the 80 conceptual cells was independently read by eye
from physical pp.49-50 / printed pp.44-45 at 400 dpi and rechecked in tight
900-dpi crops. PDF text, OCR, and the legacy CSV did not supply, complete,
normalize, infer, correct, or verify any transcription.

## Accounting

- Items: 101 `name`, 102 `man`, 103 `woman`, 104 `child`, 105 `father`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 78; source blanks: 2; ambiguous: 0; illegible: 0.
- Expanded occurrences: 111 (103 target candidates; 8 controls).
- Item 103 crosses physical pp.49-50 / printed pp.44-45.
- Item 105 crosses the left/right column boundary on physical p.50.

Every form, repeated response, group label, page/column coordinate, and cell
boundary was visually rechecked against the rendered source. CCC is absent
from the complete item-104 and item-105 blocks, so those coordinates are
explicit source blanks. Repeated group responses remain expanded: item 104 has
25 visible responses and item 105 has 32. Tight crops preserve item 101's
distinct `nãũ`, `naõ`, `não`, `nʌːu`, and `naũ` sequences; item 102/KkP
ordinary `g` in `log`; item 103/HIN ordinary `r` in `stri`; item 103/KkP
`meharu`; item 103/SkP `lʌdija`; and item 104's `lʌɽʌka` and `loɽa`.

The independent ledger was frozen before legacy comparison. During the
post-comparison source-image audit, a direct rendered-glyph comparison against
an earlier unambiguous source `ɽ` confirmed that item 104's descending flap is
retroflex; this decision came from the rendered pages, not from the legacy
string. The final deterministic ledger SHA-256 is
`a2911eb28f48141c7e73293f0068f2d96b7554bbbba943760c675c9c6639a5aa`.

## Unresolved site identities

No lexical reading is ambiguous or illegible. Ten coordinates remain
medium-confidence site assignments because both locality lists print as `RNS`.
Item 104 has two group-2 RNS rows and one unmatched group-3 RNS row; under the
documented occurrence-order rule, the first group-2 row and unmatched group-3
row are provisionally assigned together to Sisaikhara, while the second
group-2 row maps to Sisana. Every physical/printed page, item, site key,
column, visible response description, and candidate locality is enumerated in
`../unresolved_readings.tsv`.

## Post-entry reconciliation

Only after the independent ledger was frozen did the guarded importer compare
reviewed target occurrences with `20230530-tharu2.csv`. The block has 103
manual and 103 legacy target occurrences: 99 agree exactly, leaving four
manual-only and four legacy-only multiset occurrences. The rendered source
retains item 102/KkP `log` against legacy `loɡ`, item 103/KkP `meharu` against
legacy `mehaɾu`, item 103/SkP `lʌdija` against legacy `lɔɖɪja`, and item
104/RKB `walika` against legacy `walɪka`. Every difference was rechecked
visually after the comparison. The legacy data were never accepted as
transcription evidence.

Cumulatively through item 105, 1,522/1,708 legacy target occurrences agree
exactly; the multiset retains 208 manual-only and 186 legacy-only occurrences.
Staging remains refused at 1,680/3,360 reviewed cells. Item 106 `mother`,
physical p.50 / printed p.45, is next.
