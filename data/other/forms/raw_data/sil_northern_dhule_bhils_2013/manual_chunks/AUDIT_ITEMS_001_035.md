# Manual visual audit: Appendix C items 1–35

This immutable chunk covers physical PDF pages 91–97 (printed pages 83–89),
items 1–35, and all thirteen list columns. All **455 cells** were inspected at
300 dpi and keyed directly from the rendered scan. OCR did not seed, supply,
or verify any of these lexical transcriptions. A separate later locator pass
was used only to identify printed item-number boundaries on unreviewed pages;
its lexical output was ignored and is not retained in this package.

## Counts

- 436 attested cells
- 17 confirmed source blanks
- 2 ambiguous cells, retained but excluded from any future staging until
  resolved
- 0 illegible cells
- 455 total manually reviewed cells = 35 prompts × 13 lists
- 420 reviewed target cells: 403 attested, 15 blank, 2 ambiguous
- 35 reviewed Toranmal comparison cells: 33 attested, 2 blank, 0 ambiguous

The 17 explicit blanks are:

- physical 93 / printed 85, item 11 breast: KEL, AMO, BHU, TOR
- physical 95 / printed 87, item 21 heart: MUN, AST, MAN, BHU, AML, KAN,
  SHA, TOR
- physical 96 / printed 88, item 27 roof: AST, MAN, BHU, KAN, SHA

The two unresolved readings are:

- physical 92 / printed 84, item 10 tongue, KEL (left): `1 dʒibh̃`.
  Independent 900-dpi re-review confirms a final h-like character with a
  superscript tilde, but its exact intended scope/value remains unclear. The
  reading stays unresolved and is not silently regularized.
- physical 97 / printed 89, item 31 mortar, MUN (left): `2 khaŋʌɖo?`.
  Independent 900-dpi re-review confirms the response and the source-printed
  question mark. It remains unresolved because the uncertainty is the
  source's own.

## Diplomatic transcription decisions

- Source similarity-group numbers and comma-separated alternatives remain in
  `Manual_Transcription`; they are evidence, not cognacy judgments.
- The scan's underbar-like dental mark is encoded with combining dental bridge
  `U+032A`, and the scan's length dot is encoded as IPA length mark `ː`.
- Retroflex letters, nasalization, aspiration, alternatives, and the printed
  question mark are retained. NFC is required for every field.
- A blank is represented by an empty transcription plus `Review_Status=blank`;
  no dash or invented value is substituted.

The authoritative data are
`items_001_035_hand_keyed.tsv`; the neighboring Python file is a reproducible,
reviewable emitter for that TSV. Neither file contains an OCR column.

The full-ledger pagination was also visually checked at the appendix endpoint.
Most leaves contain five prompts, but physical p. 100 contains four (46–49),
p. 130 contains five (195–199), pp. 131–132 each contain five (200–204 and
205–209), and p. 133
contains item 210 alone. The base ledger and importer encode these irregular
leaves explicitly rather than extrapolating a five-prompts-per-page formula.
