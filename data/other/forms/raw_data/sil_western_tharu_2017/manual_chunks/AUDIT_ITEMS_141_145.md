# Manual audit: items 141-145

The repository ingestion checklist, survey/comparative-table addendum, PDF
inspection policy, and strict rendered-page manual-review policy controlled
this block. Every one of the 80 conceptual cells was independently read by eye
from physical pp.56-57 / printed pp.51-52 at 400 dpi and rechecked in tight
900/1200/1600-dpi crops, with targeted 2400-dpi crops for small glyphs. PDF
text, OCR, and the legacy CSV did not supply, complete, normalize, infer,
correct, or verify any transcription.

## Accounting

- Items: 141 `far`, 142 `big`, 143 `small`, 144 `heavy`, 145 `light`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 50; source blanks: 30; ambiguous: 0; illegible: 0.
- Expanded occurrences: 50 (45 target candidates; 5 controls).
- Item 141 crosses physical pp.56-57 / printed pp.51-52.
- Items 144 and 145 each print only the Hindi control response; all fifteen
  target-list cells in each complete item block are explicit source blanks.

Every form, group label, qualifier, page/column coordinate, and cell boundary
was visually rechecked against the rendered source. Item 143 retains nine
literal `(135)` qualifiers as source cross-references rather than lexical
forms or analytical assertions. Tight crops preserve item-141/TkN `duɾ`
against DKS `dʊɾ`, item-142/DGC `bʰaɽi` against the following `bʰaɾi`
forms, item-143/RKB `tʃʰoʈo`, DDK ordinary Latin `g` in `tʃʰuʈinʌg`, and
DkR `tʃʰoʈimoʈi`.

The independent ledger was frozen before legacy comparison at SHA-256
`efedd76ba6bc3d55fac5f3d0d910d3774883f5ed877e045cb4de4f995df76b21`.
During the post-comparison source-image audit, targeted 2400-dpi crops corrected
item-142/DGC to `bʰaɽi`, item-143/DDK to `tʃʰuʈinʌg`, and item-143/DkR to
`tʃʰoʈimoʈi`. These corrections came solely from the rendered source. The
final deterministic ledger SHA-256 is
`fe6cda4ad1bcfd3c2aff71bea8a2f304c7355aaffa9f9564b578f41bd5131b82`.

## Unresolved site identities

No lexical reading is ambiguous or illegible. Ten coordinates remain
medium-confidence site assignments because both locality lists print as `RNS`.
The source does not identify which occurrence belongs to Sisaikhara or Sisana;
responses are provisionally mapped in print order to metadata-row order. Items
144 and 145 have no printed RNS response, so both RNS metadata rows are blank.
Every coordinate and visible response description is enumerated in
`../unresolved_readings.tsv`.

## Post-entry reconciliation

Only after the independent ledger was frozen did the guarded importer compare
reviewed target occurrences with `20230530-tharu2.csv`. The block has 45 manual
and 45 legacy target occurrences: 42 agree exactly, leaving three manual-only
and three legacy-only multiset occurrences. The source-retained differences
are item-141/TkN `duɾ` versus legacy `dʊɾ`, item-143/RKB `tʃʰoʈo` versus
legacy `tʃʰoto`, and source ordinary `g` versus legacy IPA `ɡ` in
item-143/DDK. Every difference was rechecked visually after the comparison.
The legacy data were never accepted as transcription evidence.

Cumulatively through item 145, 2,009/2,317 legacy target occurrences agree
exactly; the multiset retains 325 manual-only and 308 legacy-only occurrences.
Staging remains refused at 2,320/3,360 reviewed cells. Item 146 `above`,
physical p.57 / printed p.52, is next.
