# Manual audit: items 186-190

The repository ingestion checklist, survey/comparative-table addendum, PDF
inspection policy, and strict rendered-page manual-review policy controlled
this block. Every one of the 80 conceptual cells was independently read by eye
from physical pp.64-65 / printed pp.59-60 and rechecked in tight 900/1800-dpi
rendered-page crops. PDF text, OCR, and the legacy CSV did not supply, complete,
normalize, infer, correct, or verify any transcription.

## Accounting

- Items: 186 `he is/was thirsty`, 187 `he sleeps; he slept`, 188 `lie down!;
  he lay down`, 189 `sit down; he sat do`, 190 `give!; he gave`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 78; source blanks: 2; ambiguous: 0; illegible: 0.
- Expanded occurrences: 140 (130 target candidates; 10 controls).
- Item 188 crosses physical pp.64-65 / printed pp.59-60.
- CCC is absent from the complete item-186 block; DkR is absent from the
  complete item-188 block. Both coordinates are explicit source blanks.

Every form, group label, page/column coordinate, and cell boundary was visually
rechecked against the rendered source. Literal ellipses and periods in item 186
are described in `Source_Qualifier`, not retained as lexical characters. The
ledger preserves item-187 group-1/group-2 responses, item-188 group-2/group-4
CCC alternatives and literal `(187)` references, item-189's two separately
printed CCC responses, and all imperative/declarative pairs. Tight crops retain
source-visible ordinary `g` against IPA `ɡ`, plain `t` against retroflex `ʈ`,
alveolar flap `ɾ` against retroflex flap `ɽ`, nasalized vowels, repeated `j`,
and item-188/RkM's single unsegmented printed form
`ledʒdʒaːleʈõɾʌhõ`.

The independent ledger was frozen before legacy comparison at SHA-256
`512802839521b4b609eb1bc1f1d39682e518ad82f818d967536cb90c6f3c3b47`.
The post-comparison audit returned only to the rendered cells and corrected
four retained readings: both item-188/SkP alternatives changed their first
flap from `ɾ` to the visibly retroflex `ɽ`, while item-190/RNS occurrence 2
changed `dejdæ / dejdʌi` to source-visible `dejʌde / dejʌdɪ`. The final
deterministic ledger SHA-256 is
`0b5ae95cadb76c9cd107b1e5a92f816651b86833ae38d1b572929f516c1e3186`.

## Unresolved site identities

No lexical reading is ambiguous or illegible. Ten coordinates remain
medium-confidence site assignments because both locality lists print as `RNS`.
The source does not identify which occurrence belongs to Sisaikhara or Sisana;
responses are provisionally mapped in within-item print order to metadata-row
order. Every coordinate and visible response description is enumerated in
`../unresolved_readings.tsv`.

## Post-entry reconciliation

Only after the independent ledger was frozen did the guarded importer compare
reviewed target occurrences with `20230530-tharu2.csv`. Fifty-one of 130 manual
target occurrences agree exactly with 51 of 130 legacy occurrences. The 79
manual-only and 79 legacy-only differences retain source punctuation handling,
plain/retroflex stops and flaps, `i`/`ɪ`, `u`/`ʊ`, `g`/`ɡ`, nasalization,
repeated `j`, qualifiers, and the single unsegmented RkM response; the legacy
data were never accepted as transcription evidence.

Cumulatively through item 190, 2,595/3,123 legacy target occurrences agree
exactly; the multiset retains 543 manual-only and 528 legacy-only occurrences.
Staging remains refused at 3,040/3,360 reviewed cells. Item 191 `it burns; it
burned`, physical p.65 / printed p.60, is next.
