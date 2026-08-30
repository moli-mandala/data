# Manual audit: items 146-150

The repository ingestion checklist, survey/comparative-table addendum, PDF
inspection policy, and strict rendered-page manual-review policy controlled
this block. Every one of the 80 conceptual cells was independently read by eye
from physical pp.57-58 / printed pp.52-53 at 400 dpi and rechecked in tight
900/1200/1600-dpi crops, with targeted 2400-dpi crops for small glyphs. PDF
text, OCR, and the legacy CSV did not supply, complete, normalize, infer,
correct, or verify any transcription.

## Accounting

- Items: 146 `above`, 147 `below`, 148 `white`, 149 `black`, 150 `red`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 80; source blanks: 0; ambiguous: 0; illegible: 0.
- Expanded occurrences: 81 (76 target candidates; 5 controls).
- Item 148 crosses physical pp.57-58 / printed pp.52-53. BNM's group-2
  `seta` and group-4 `bʰuɾo` are retained as two responses in one conceptual
  cell rather than split or collapsed.

Every form, repeated response, group label, page/column coordinate, and cell
boundary was visually rechecked against the rendered source. Tight crops
preserve item 146's `u`/`ʊ` and length contrasts plus CCC `upːiɾi`,
item-147/DKS initial retroflex `ʈ`, item 148's DGC/DKS `ʊdʒːʌɾ`, DkR
`ʊɖːal`, DDK `uɖːaɾ`, and CCC ordinary Latin `g` in `goɾʌhʌɾ`, and item
149's source-dotted `i` in the six `kʌɾija` responses.

The independent ledger was frozen before legacy comparison at SHA-256
`2a244e3173f3762d3bb9f36fd1027117cbcae3176e543ac34b8f09f839b0ba6e`.
During the post-comparison source-image audit, targeted 2400-dpi crops corrected
item-147/DKS to `ʈʌɾe`, item-148/DkR to `ʊɖːal`, and item-148/DDK to
`uɖːaɾ`. These corrections came solely from the rendered source. The final
deterministic ledger SHA-256 is
`892993aac5fe0649f4d77036f4a403c2151bf744870aed876ffb94399f78dc22`.

## Unresolved site identities

No lexical reading is ambiguous or illegible. Ten coordinates remain
medium-confidence site assignments because both locality lists print as `RNS`.
The source does not identify which occurrence belongs to Sisaikhara or Sisana;
responses are provisionally mapped in print order to metadata-row order. Every
coordinate and visible response description is enumerated in
`../unresolved_readings.tsv`.

## Post-entry reconciliation

Only after the independent ledger was frozen did the guarded importer compare
reviewed target occurrences with `20230530-tharu2.csv`. The block has 76 manual
and 76 legacy target occurrences: 69 agree exactly, leaving seven manual-only
and seven legacy-only multiset occurrences. Six differences retain source
dotted `i` against legacy `ɪ` in item 149; the seventh retains source ordinary
Latin `g` against legacy IPA `ɡ` in item-148/CCC. Every difference was
rechecked visually after the comparison. The legacy data were never accepted
as transcription evidence.

Cumulatively through item 150, 2,078/2,393 legacy target occurrences agree
exactly; the multiset retains 332 manual-only and 315 legacy-only occurrences.
Staging remains refused at 2,400/3,360 reviewed cells. Item 151 `one`, physical
p.58 / printed p.53, is next.
