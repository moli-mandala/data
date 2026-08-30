# Manual audit: items 196-200

The repository ingestion checklist, survey/comparative-table addendum, PDF
inspection policy, and strict rendered-page manual-review policy controlled
this block. Every one of the 80 conceptual cells was independently read by eye
from physical pp.66-67 / printed pp.61-62 and rechecked in tight 900/1800-dpi
crops. PDF text, OCR, and the legacy TSV did not supply, complete, normalize,
infer, correct, or verify any transcription.

## Accounting

- Items: 196 `run!; he ran`, 197 `go!; he went`, 198 `come!; he came`,
  199 `speak!; he spoke`, 200 `he hears; he heard`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 78; source blanks: 2; ambiguous: 0; illegible: 0.
- Expanded occurrences: 144 (134 target candidates; 10 controls).
- Item 196 crosses the left/right column boundary on physical p.66 / printed
  p.61; item 200 crosses physical pp.66-67 / printed pp.61-62.
- CCC is absent from the complete item-196 and item-199 blocks. Both are
  retained as explicit source blanks.

Every form, group label, page/column coordinate, and cell boundary was visually
rechecked against the rendered source. Repeated RKB and RNS responses in item
196 and every RNS response in item 197 remain expanded under their printed
group labels. Literal ellipses and `(195)`, `(196)`, and `(past)` are qualifiers,
not lexical characters. Item 197/DKS's literal colon response separator and
item 198/BNM's parenthesized prefix are also retained as source qualifiers.
Tight crops preserve ordinary `r` against alveolar `ɾ` and retroflex `ɽ`,
ordinary `g` against IPA `ɡ`, dotted `i` against small-capital `ɪ`, length,
nasalization, repeated consonants, and vowel contrasts.

The independent ledger was frozen before legacy comparison at SHA-256
`431001c022db3b3dd88787c7212efe8793e067eb5feb6db1cd3e6a890a253ce1`.
The post-comparison audit returned only to the 1800-dpi source crops and
corrected one reading: item 196/TkN's second response visibly contains repeated
ordinary `r`, `dɔrrʌho`. The final deterministic ledger SHA-256 is
`0980e0652127db0d8894a65dbd39b2db8fa74786b70f589dc78bdbf552ca77f1`.

## Unresolved site identities

No lexical reading is ambiguous or illegible. Ten coordinates remain
medium-confidence site assignments because both locality lists print as `RNS`.
Within each comparison group, occurrence order is provisionally mapped to
metadata-row order; an unmatched third occurrence and singleton group response
remain with provisional metadata row 1. This preserves every response without
claiming a locality identity. Every coordinate and visible-response description
is enumerated in `../unresolved_readings.tsv`.

## Post-entry reconciliation

Only after the independent ledger was frozen did the guarded importer compare
reviewed target occurrences with `20230530-tharu2.csv`. Forty-seven of 134
manual target occurrences agree exactly with 47 of 134 legacy occurrences.
The 87 manual-only and 87 legacy-only differences retain source punctuation
handling, ordinary/flap contrasts, `i`/`ɪ`, `g`/`ɡ`, qualifiers, and repeated
consonant evidence; the legacy data were never accepted as transcription
evidence.

Cumulatively through item 200, 2,673/3,392 legacy target occurrences agree
exactly; the multiset retains 734 manual-only and 719 legacy-only occurrences.
Staging remains refused at 3,200/3,360 reviewed cells. Item 201 `he sees; he
saw`, physical p.67 / printed p.62, is next.
