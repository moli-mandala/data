# Manual audit: items 111-115

The repository ingestion checklist, survey/comparative-table addendum, PDF
inspection policy, and strict rendered-page manual-review policy controlled
this block. Every one of the 80 conceptual cells was independently read by eye
from physical pp.51-52 / printed pp.46-47 at 400 dpi and rechecked in tight
900/1200-dpi crops. PDF text, OCR, and the legacy CSV did not supply, complete,
normalize, infer, correct, or verify any transcription.

## Accounting

- Items: 111 `son`, 112 `daughter`, 113 `husband`, 114 `wife`, 115 `boy`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 75; source blanks: 5; ambiguous: 0; illegible: 0.
- Expanded occurrences: 86 (79 target candidates; 7 controls).
- Item 113 crosses physical pp.51-52 / printed pp.46-47.

Every form, repeated response, group label, qualifier, page/column coordinate,
and cell boundary was visually rechecked against the rendered source. CCC is
absent from every complete item block, so those five coordinates are explicit
source blanks. Repeated responses remain expanded, including two Hindi forms
in items 111 and 112, DGC's two daughter forms, KkP's three husband forms,
BNM's two wife forms, and all visible boy variants. Literal cross-references
`(102)`, `(103)`, `(104)`, and `(111)` are retained as qualifiers rather than
forms. Tight crops preserve the source retroflex flap `ɽ`, the `loɳɖ-`
variants, ordinary Latin `g` in item 113, ordinary `r` in item 114/KkP, and
source dotted `i` in the item-112 RNS and RkM forms.

The independent ledger was frozen before legacy comparison. During the
post-comparison source-image audit, targeted 1200-dpi rechecks confirmed the
retroflex flap and rounded-vowel readings without accepting legacy strings as
evidence. The final deterministic ledger SHA-256 is
`6046700b6d51780259c2d79af0189f2deb2204525a7bcd2f5a0af1ce1d66014c`.

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
manual and 79 legacy target occurrences: 64 agree exactly, leaving fifteen
manual-only and fifteen legacy-only multiset occurrences. The source retains
ordinary `g` rather than legacy IPA `ɡ` in nine item-113 occurrences and one
item-114 occurrence, item-114/KkP ordinary `r` rather than legacy `ɾ`, and
source dotted `i` rather than legacy `ɪ` in item-112/RNS and RkM. It also
retains item-111/KkP `loɳɖa` and item-112/KkP `loɳɖia` against legacy
`lɔɳɖa` and `lɔɳɖia`. Every difference was rechecked visually after the
comparison. The legacy data were never accepted as transcription evidence.

Cumulatively through item 115, 1,659/1,864 legacy target occurrences agree
exactly; the multiset retains 227 manual-only and 205 legacy-only occurrences.
Staging remains refused at 1,840/3,360 reviewed cells. Item 116 `girl`, physical
p.52 / printed p.47, is next.
