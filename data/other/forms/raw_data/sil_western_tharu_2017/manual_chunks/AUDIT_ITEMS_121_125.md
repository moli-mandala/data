# Manual audit: items 121-125

The repository ingestion checklist, survey/comparative-table addendum, PDF
inspection policy, and strict rendered-page manual-review policy controlled
this block. Every one of the 80 conceptual cells was independently read by eye
from physical pp.53-54 / printed pp.48-49 at 400 dpi and rechecked in tight
900/1200/1600-dpi crops. PDF text, OCR, and the legacy CSV did not supply,
complete, normalize, infer, correct, or verify any transcription.

## Accounting

- Items: 121 `evening`, 122 `yesterday`, 123 `today`, 124 `tomorrow`, 125 `week`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 77; source blanks: 3; ambiguous: 0; illegible: 0.
- Expanded occurrences: 81 (75 target candidates; 6 controls).
- Item 121 crosses the left/right column boundary on physical p.53.
- Item 124 crosses physical pp.53-54 / printed pp.48-49.

Every form, repeated response, group label, qualifier, page/column coordinate,
and cell boundary was visually rechecked against the rendered source. Item 125
has no BNT or DDK row and prints only one RNS response; these are three explicit
source blanks, with the unassignable second RNS locality retained separately.
Repeated responses remain expanded: item 121 retains two Hindi, BNM, and DKS
forms; item 125/RKB retains both group-1 and group-2 responses. Literal `(118)`,
`(122)`, and `(used most)` annotations are retained as qualifiers rather than
forms. Tight crops preserve nasal vowels and aspiration in item 121, item 123
`adʒʊ`, and item 125's `hʌptʌh` and `aʈʰʌdin`.

The independent ledger was frozen before legacy comparison. No source reading
changed during post-comparison rendered-image rechecks. The deterministic
ledger SHA-256 is
`2cdad8cc73e03630daa6283e3434d44fb755ec7c029557af48656b6d3091965e`.

## Unresolved site identities

No lexical reading is ambiguous or illegible. Ten coordinates remain
medium-confidence site assignments because both locality lists print as `RNS`.
Item 125 prints only one RNS response: under the documented occurrence-order
rule it is provisionally assigned to Sisaikhara, while Sisana remains an
explicit blank with no invented response. Every physical/printed page, item,
site key, column, visible response description, and candidate locality is
enumerated in `../unresolved_readings.tsv`.

## Post-entry reconciliation

Only after the independent ledger was frozen did the guarded importer compare
reviewed target occurrences with `20230530-tharu2.csv`. The block has 75
manual and 77 legacy target occurrences: 69 agree exactly, leaving six
manual-only and eight legacy-only multiset occurrences. The rendered source
retains item-121/DKS `sahidʒʊn`, item-123/CCC `adʒʊ`, item-124/DDK `kal`,
item-125/CCC `hʌpta`, DKS `hʌptʌh`, and RKB `aʈʰʌdin`. The final RKB
qualifier is source metadata; the legacy CSV has split `(used most)` into two
spurious lexical rows. Every difference was rechecked visually after the
comparison. The legacy data were never accepted as transcription evidence.

Cumulatively through item 125, 1,787/2,020 legacy target occurrences agree
exactly; the multiset retains 253 manual-only and 233 legacy-only occurrences.
Staging remains refused at 2,000/3,360 reviewed cells. Item 126 `month`,
physical p.54 / printed p.49, is next.
