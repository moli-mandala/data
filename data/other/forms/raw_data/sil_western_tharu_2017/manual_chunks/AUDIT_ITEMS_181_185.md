# Manual audit: items 181-185

The repository ingestion checklist, survey/comparative-table addendum, PDF
inspection policy, and strict rendered-page manual-review policy controlled
this block. Every one of the 80 conceptual cells was independently read by eye
from physical pp.63-64 / printed pp.58-59 and rechecked in tight 900/1800-dpi
rendered-page crops. PDF text, OCR, and the legacy CSV did not supply, complete,
normalize, infer, correct, or verify any transcription.

## Accounting

- Items: 181 `all`, 182 `eat!; he ate`, 183 `bite!; he bit`, 184 `he is/was
  hungry`, 185 `drink!; he drank`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 79; source blanks: 1; ambiguous: 0; illegible: 0.
- Expanded occurrences: 135 (125 target candidates; 10 controls).
- Item 182 crosses physical pp.63-64 / printed pp.58-59; item 185 crosses the
  left/right column boundary on physical p.64 / printed p.59.
- CCC is absent from the complete printed item-184 block and is retained as an
  explicit source blank.

Every form, group label, page/column coordinate, and cell boundary was visually
rechecked against the rendered source. Literal ellipses and periods attached to
incomplete response fragments are described in `Source_Qualifier`; they are
not invented lexical forms. Tight crops preserve item 181's ordinary `r`/flap
`ɾ` contrasts and `(176)` references; item 182's plain `t` in DDK, ordinary
`d` in DKS, and page-spanning responses; item 183's retroflex series except for
the visibly plain `t` in CCC `kʌtʌi`; item 184's ordinary `r` in RkM
`bʰukʰorʌ` and second RNS `bʰukʰorʌhʊ`; and item 185's
`i`/`ɪ`, `u`/`ʊ`, and nasalization contrasts.

The independent ledger was frozen before legacy comparison at SHA-256
`5f6dbee9c643bb87c8e35bc73aca778c1f38f106cfa30263b7a56d852b2f20da`.
The post-comparison audit returned to the rendered source, not the legacy
strings, and corrected three glyph classifications: item 183/CCC `kʌʈʌi` to
source-visible `kʌtʌi`, item 184/RkM `bʰukʰoɾʌ` to `bʰukʰorʌ`,
and item 184/RNS occurrence 2 `bʰukʰoɾʌhʊ` to `bʰukʰorʌhʊ`.
The final deterministic ledger SHA-256 is
`4c2112f505ca1ce5e74d5f28c99b4b3fd55971984f994fcea733e2169a3d683b`.

## Unresolved site identities

No lexical reading is ambiguous or illegible. Ten coordinates remain
medium-confidence site assignments because both locality lists print as `RNS`.
The source does not identify which occurrence belongs to Sisaikhara or Sisana;
responses are provisionally mapped in within-item print order to metadata-row
order. Every coordinate and visible response description is enumerated in
`../unresolved_readings.tsv`.

## Post-entry reconciliation

Only after the independent ledger was frozen did the guarded importer compare
reviewed target occurrences with `20230530-tharu2.csv`. Sixty-six of 125 manual
target occurrences agree exactly with 66 of 125 legacy occurrences. The 59
manual-only and 59 legacy-only differences retain the rendered source's
plain/retroflex stop and ordinary-r/flap contrasts, vowels, source qualifiers,
response boundaries, and punctuation treatment; the legacy data were never
accepted as transcription evidence.

Cumulatively through item 185, 2,544/2,993 legacy target occurrences agree
exactly; the multiset retains 464 manual-only and 449 legacy-only occurrences.
Staging remains refused at 2,960/3,360 reviewed cells. Item 186 `sleep!; he
slept`, physical p.64 / printed p.59, is next.
