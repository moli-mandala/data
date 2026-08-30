# Manual audit: items 116-120

The repository ingestion checklist, survey/comparative-table addendum, PDF
inspection policy, and strict rendered-page manual-review policy controlled
this block. Every one of the 80 conceptual cells was independently read by eye
from physical pp.52-53 / printed pp.47-48 at 400 dpi and rechecked in tight
900/1200/1600-dpi crops. PDF text, OCR, and the legacy CSV did not supply,
complete, normalize, infer, correct, or verify any transcription.

## Accounting

- Items: 116 `girl`, 117 `day`, 118 `night`, 119 `morning`, 120 `noon`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 78; source blanks: 2; ambiguous: 0; illegible: 0.
- Expanded occurrences: 85 (79 target candidates; 6 controls).
- Item 119 crosses physical pp.52-53 / printed pp.47-48.

Every form, repeated response, group label, qualifier, page/column coordinate,
and cell boundary was visually rechecked against the rendered source. CCC is
absent from the complete item-116 and item-120 blocks, so those coordinates
are explicit source blanks. Repeated responses remain expanded: item 116
retains BNM, DDK, and DKS alternatives; item 119 retains two Hindi, DGC, DKS,
and DkR responses. Literal `(110)` and `(112)` cross-references are retained as
qualifiers rather than forms. Tight crops preserve item 116/SkP unaspirated
`tʃais`, the source's dotted `i`, item 119/BNM retroflex `ɖ`, and item 120's
`o`/`u`/`ʊ`, `ʌ`/`a`, and `ɔ` contrasts.

The independent ledger was frozen before legacy comparison at SHA-256
`2ac3a0ac1d838ee10bc66037e439038bc0560a424e6c7de54e7dcf254cce49a1`.
During the post-comparison source-image audit, targeted 1600-dpi crops showed
that item 116/SkP is unaspirated and item 119/HIN begins `sʊbʌh`; these
rendered-page readings, not legacy strings, supplied the corrections. The
final deterministic ledger SHA-256 is
`c39726bc41314af68d8ec9e7b9abbbcda539aaa7c0d0c14a39483c56c9da60f3`.

## Unresolved site identities

No lexical reading is ambiguous or illegible. Ten coordinates remain
medium-confidence site assignments because both locality lists print as `RNS`.
The source does not identify which occurrence belongs to Sisaikhara or Sisana;
the two occurrences are provisionally mapped in print order to metadata-row
order without inferring locality identity. Every physical/printed page, item,
site key, column, visible response description, and candidate locality is
enumerated in `../unresolved_readings.tsv`.

## Post-entry reconciliation

Only after the independent ledger was frozen did the guarded importer compare
reviewed target occurrences with `20230530-tharu2.csv`. The block has 79
manual and 79 legacy target occurrences: 59 agree exactly, leaving twenty
manual-only and twenty legacy-only multiset occurrences. Fourteen item-117
forms retain source dotted `i` against legacy `ɪ`. The other six differences
are item-116/KkP source `loɳɖia`, item-116/RNS and RkM source dotted `i`,
item-119/DkR source dotted `i`, and item-120/BNT and DkR source dotted `i`.
Every difference was rechecked visually after the comparison. The legacy data
were never accepted as transcription evidence.

Cumulatively through item 120, 1,718/1,943 legacy target occurrences agree
exactly; the multiset retains 247 manual-only and 225 legacy-only occurrences.
Staging remains refused at 1,920/3,360 reviewed cells. Item 121 `evening`,
physical p.53 / printed p.48, is next.
