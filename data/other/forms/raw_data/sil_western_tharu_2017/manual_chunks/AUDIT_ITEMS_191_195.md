# Manual audit: items 191-195

The repository ingestion checklist, survey/comparative-table addendum, PDF
inspection policy, and strict rendered-page manual-review policy controlled
this block. Every one of the 80 conceptual cells was independently read by eye
from physical pp.65-66 / printed pp.60-61 and rechecked in tight 900/1800-dpi
crops, with targeted 3600-dpi crops for flap and vowel distinctions. PDF text,
OCR, and the legacy CSV did not supply, complete, normalize, infer, correct, or
verify any transcription.

## Accounting

- Items: 191 `it burns; it burned`, 192 `he dies; he died`, 193 `kill!; he
  killed`, 194 `it flies; it flew`, 195 `walk!; he walked`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 79; source blanks: 1; ambiguous: 0; illegible: 0.
- Expanded occurrences: 145 (135 target candidates; 10 controls).
- Item 191 crosses the left/right column boundary on physical p.65 / printed
  p.60; item 194 crosses physical pp.65-66 / printed pp.60-61.
- CCC is absent from the complete item-193 block and is retained as an explicit
  source blank.

Every form, group label, page/column coordinate, and cell boundary was visually
rechecked against the rendered source. Repeated DGC responses in item 191,
RKB/DDK/DGC/CCC responses in item 195, and every RNS occurrence remain expanded
under the printed group labels. Literal ellipses and periods are recorded as
qualifiers rather than lexical characters; item-193 `(192)` cross-references
remain source qualifiers. Tight crops preserve ordinary `r`, alveolar `ɾ`,
retroflex `ɽ`, and retroflex stops as distinct; source-visible ordinary `g`
against IPA `ɡ`; nasalization; repeated `j`; and vowel contrasts. Item 194/TkN
remains the single unsegmented printed response `ʊɽɾʌhihæ̃ʊɽʌt`, and
item 194/RNS occurrence 2 preserves length and small-capital `ɪ` in
`uɾːɪhæ̃ / uɾːɪrʌhẽ`.

The independent ledger was frozen before legacy comparison at SHA-256
`4ddc7140ef709dcaa8dc79bcceb91dee6141bfcadf5dd3b5ad7e3b0197ac21de`.
The post-comparison audit returned only to targeted 3600-dpi source crops and
corrected 33 retained form readings. These comprise the missing item-192/TkN
flap; item 194's source-visible retroflex/alveolar/ordinary-r contrasts, RNS
length and `ɪ`, and TkN sequence; and item 195's ordinary-r series, DkR
retroflex flaps, DGC initial flap, and CCC vowel. The final deterministic ledger
SHA-256 is
`98f82183485dc07d96323fbcb805b7c9134fcb6638c132f3cf0f55da15c3ac3e`.

## Unresolved site identities

No lexical reading is ambiguous or illegible. Ten coordinates remain
medium-confidence site assignments because both locality lists print as `RNS`.
The source does not identify which occurrence belongs to Sisaikhara or Sisana;
responses are provisionally mapped in within-item print order to metadata-row
order. Every coordinate and visible response description is enumerated in
`../unresolved_readings.tsv`.

## Post-entry reconciliation

Only after the independent ledger was frozen did the guarded importer compare
reviewed target occurrences with `20230530-tharu2.csv`. Thirty-one of 135 manual
target occurrences agree exactly with 31 of 135 legacy occurrences. The 104
manual-only and 104 legacy-only differences retain source punctuation handling,
ordinary/flap contrasts, `i`/`ɪ`, `u`/`ʊ`, `g`/`ɡ`, aspiration,
nasalization, repeated `j`, qualifiers, and unsegmented response evidence; the
legacy data were never accepted as transcription evidence.

Cumulatively through item 195, 2,626/3,258 legacy target occurrences agree
exactly; the multiset retains 647 manual-only and 632 legacy-only occurrences.
Staging remains refused at 3,120/3,360 reviewed cells. Item 196 `run!; he ran`,
physical p.66 / printed p.61, is next.
