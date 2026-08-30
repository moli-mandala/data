# Manual audit: items 131-135

The repository ingestion checklist, survey/comparative-table addendum, PDF
inspection policy, and strict rendered-page manual-review policy controlled
this block. Every one of the 80 conceptual cells was independently read by eye
from physical pp.55-56 / printed pp.50-51 at 400 dpi and rechecked in tight
900/1200/1600-dpi crops, with targeted 2400-dpi crops for small glyphs. PDF
text, OCR, and the legacy CSV did not supply, complete, normalize, infer,
correct, or verify any transcription.

## Accounting

- Items: 131 `bad`, 132 `wet`, 133 `dry`, 134 `long`, 135 `short`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 80; source blanks: 0; ambiguous: 0; illegible: 0.
- Expanded occurrences: 97 (87 target candidates; 10 controls).
- Item 133 crosses the left/right column boundary on physical p.55.
- Item 135 crosses physical pp.55-56 / printed pp.50-51.

Every form, repeated response, group label, qualifier, page/column coordinate,
and cell boundary was visually rechecked against the rendered source. Item 131
retains four Hindi, two CCC, and two KkP responses; KkP's `(person)` and
`(object)` annotations remain qualifiers. Item 132 retains two Hindi and BNM
responses; item 134/DDK retains both responses; item 135 retains all repeated
group-1/group-3 forms rather than collapsing them. Tight crops preserve dotted
`i`, ordinary Latin `g`, item-131/SkP `tʃʰɪʈɔn`, item-134/DDK `dʰẽɖ`, and
item 135's aspiration and retroflex-stop contrasts.

The independent ledger was frozen before legacy comparison at SHA-256
`7cf9ccdf4199c915407a9480ab5f568e030c0edc546e8fd7c0e18f1582cbc8e7`.
During the post-comparison source-image audit, targeted 2400-dpi crops corrected
item-131/SkP to `tʃʰɪʈɔn`, DkR to `gʌndhʌjʌna`, and KkP's second form to
`mælʌha`. These corrections came solely from the rendered source. The final
deterministic ledger SHA-256 is
`07feb89c2b5f69391deda31fc3fd25ff564da54770ff36adf4d01b8117d3168d`.

## Unresolved site identities

No lexical reading is ambiguous or illegible. Ten coordinates remain
medium-confidence site assignments because both locality lists print as `RNS`.
The source does not identify which occurrence belongs to Sisaikhara or Sisana;
responses are provisionally mapped in print order to metadata-row order. Every
physical/printed page, item, site key, column, visible response description,
and candidate locality is enumerated in `../unresolved_readings.tsv`.

## Post-entry reconciliation

Only after the independent ledger was frozen did the guarded importer compare
reviewed target occurrences with `20230530-tharu2.csv`. The block has 87
manual and 89 legacy target occurrences: 60 agree exactly, leaving twenty-seven
manual-only and twenty-nine legacy-only multiset occurrences. Source-retained
differences principally preserve ordinary `g`, dotted `i`, `u` against legacy
`ʊ`, and the printed alveolar/retroflex contrasts. The two excess legacy rows
are the item-131/KkP `(person)` and `(object)` qualifiers misparsed as forms in
the legacy CSV. Every difference was rechecked visually after the comparison.
The legacy data were never accepted as transcription evidence.

Cumulatively through item 135, 1,915/2,188 legacy target occurrences agree
exactly; the multiset retains 291 manual-only and 273 legacy-only occurrences.
Staging remains refused at 2,160/3,360 reviewed cells. Item 136 `hot`, physical
p.56 / printed p.51, is next.
