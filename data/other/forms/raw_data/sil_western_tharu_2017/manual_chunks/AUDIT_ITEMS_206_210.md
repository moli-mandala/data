# Manual audit: items 206-210

The repository ingestion checklist, survey/comparative-table addendum, PDF
inspection policy, and strict rendered-page manual-review policy controlled
this final block. Every one of the 80 conceptual cells was independently read
by eye from physical p.68 / printed p.63 and rechecked in tight 900/1800-dpi
crops, with a targeted 3600-dpi check for item 209/DKS. PDF text, OCR, and the
legacy TSV did not supply, complete, normalize, infer, correct, or verify any
transcription.

## Accounting

- Items: 206 `she`, 207 `we (inc.)`, 208 `we (exc.)`, 209 `you (2nd pl.)`,
  210 `they`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 62; source blanks: 18; ambiguous: 0; illegible: 0.
- Expanded occurrences: 69 (65 target candidates; 4 controls).
- Item 206 prints only its heading and no response rows, yielding 16 explicit
  source blanks. CCC is absent from the complete item-208 and item-209 blocks,
  yielding two further explicit blanks.

Every form, group label, page/column coordinate, and cell boundary was visually
rechecked against the rendered source. Repeated RNS, DDK, KkP, and DGC responses
remain expanded under their printed group labels. Item 208's `(202)`/`(207)`,
item 209's `(203)`/`(206)`, and item 210's `(174)`/`(205)` cross-references are
source qualifiers, not lexical forms. Tight crops preserve ordinary `r` against
alveolar `ɾ`, ordinary `g` against IPA `ɡ`, retroflex `ʈ` against plain `t`,
`u` against `ʊ`, nasalization, aspiration, and printed vowel contrasts.

The independent ledger was frozen before legacy comparison at SHA-256
`4f278d8c640c0b7f7fdf0629ebcb211103d79df406f68076ae3ddb4e9d694855`.
The post-comparison audit returned only to a targeted 3600-dpi source crop and
corrected one reading: item 209/DKS is visibly `ʈureh`, with retroflex `ʈ`,
plain `u`, and ordinary `r`. The final deterministic ledger SHA-256 is
`ae528792b27750002fffdfa67fc4ff542c8a345ac3178dc6e8be6cac0373253c`.

## Unresolved site identities

No lexical reading is ambiguous or illegible. Ten final coordinates remain
medium-confidence site assignments because both locality lists print as `RNS`.
The two item-206 coordinates are explicit blanks; all attested RNS responses
are provisionally mapped in within-item and within-group occurrence order to
metadata-row order without claiming locality identity. Every coordinate and
visible-response description is enumerated in `../unresolved_readings.tsv`.

## Post-entry reconciliation

Only after the independent ledger was frozen did the guarded importer compare
reviewed target occurrences with `20230530-tharu2.csv`. Forty-six of 65 manual
target occurrences agree exactly with 46 of 65 legacy occurrences. The 19
manual-only and 19 legacy-only differences retain rendered-source ordinary
`r`, ordinary `g`, and vowel evidence; the legacy data were never accepted as
transcription evidence.

Cumulatively through the final item 210, 2,794/3,548 legacy target occurrences
agree exactly; the multiset retains 766 manual-only and 754 legacy-only
occurrences. All 3,360/3,360 conceptual cells now have manual decisions, no
lexical ambiguity or illegibility remains, and the source-local staging guard
passes. Shared integration, build, graph review, and browser QA remain outside
this source-local lane.
