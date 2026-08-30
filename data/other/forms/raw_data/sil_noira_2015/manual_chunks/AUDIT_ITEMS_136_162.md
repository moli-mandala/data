# Manual review audit — Noira items 136–162

## Scope and method

- Source: ESR 2015-012, Appendix A3.
- Items: 136–162 inclusive; 27 prompts × 17 ordered lists = 459 conceptual cells.
- Source coordinates: physical PDF pp. 62–67 / printed pp. 56–61, with the physical page and left/right column recorded per item/site row.
- Each cell was visually inspected and independently hand-keyed from 400-dpi rendered page images. Difficult retroflex, nasalized, affricate, and vowel clusters were re-rendered and checked at 900 dpi.
- OCR and PDF text were not used to seed, supply, or verify any lexical reading. The ledger has no OCR field and every row declares `hand-keyed-from-rendered-source; OCR-not-copied`.
- All fields are NFC-normalized. Printed numeric cognate/similarity labels are retained separately from the diplomatic phonetic transcription. Multiple responses are preserved in source order.

## Accounting

- Manually reviewed conceptual cells: 459.
- Attested cells: 459 (378 target, 81 controls).
- Explicit source blanks: 0.
- Ambiguous cells: 0.
- Illegible or clipped cells: 0.
- Unresolved readings: 0.
- Expanded printed responses: 492 (402 target candidates, 90 controls).

## High-resolution review notes

The 900-dpi pass resolved the potentially confusable printed glyphs without an unresolved reading. Representative diplomatic decisions include item 136 `MAR` `uʂɳʌ | gʌrʌm`; item 137 `NTO` `heɭo`; item 148 `HIN` `səɸeɖ`; item 149 `NTE` `kẽɳdʒe`; item 154 `NCH` `tʒjʌr`; and item 161 `BMU` `igjaʌ | igjara` versus `DBA` `igijʌre`. These are visual decisions from the rendered source, not normalized reconstructions.

## Validation

`items_136_162_hand_keyed.tsv` is regenerated only from the literal hand-entered decisions in `hand_keyed_items_136_162.py`. The guarded source-local importer accepts 459 rows and stages 402 target response candidates. Focused tests check exact coverage, disjoint item/site keys, NFC, OCR-blind schema, mandatory declaration, page continuations, response counts, high-resolution decisions, cumulative accounting, and rejection of OCR-bearing or undeclared ledgers.
