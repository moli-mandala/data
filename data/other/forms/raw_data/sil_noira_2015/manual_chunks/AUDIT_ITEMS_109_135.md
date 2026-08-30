# Manual review audit — Noira items 109–135

## Scope and method

- Source: ESR 2015-012, Appendix A3.
- Items: 109–135 inclusive; 27 prompts × 17 ordered lists = 459 conceptual cells.
- Source coordinates: physical PDF pp. 56–62 / printed pp. 50–56, with the physical page and left/right column recorded per item/site row.
- Each cell was visually inspected and independently hand-keyed from 400-dpi rendered page images. Difficult glyphs on physical pp. 57, 59, and 61 were re-rendered and checked at 900 dpi.
- OCR and PDF text were not used to seed, supply, or verify any lexical reading. The ledger has no OCR field and every row declares `hand-keyed-from-rendered-source; OCR-not-copied`.
- All fields are NFC-normalized. Printed numeric cognate/similarity labels are retained separately from the diplomatic phonetic transcription. Multiple responses are preserved in source order.

## Accounting

- Manually reviewed conceptual cells: 459.
- Attested cells: 457 (376 target, 81 controls).
- Explicit source blanks: 2 (both target).
- Ambiguous cells: 0.
- Illegible or clipped cells: 0.
- Unresolved readings: 0.
- Expanded printed responses: 580 (457 target candidates, 123 controls).

## Explicit blanks

1. Physical PDF p. 58 / printed p. 52 / item 115 “boy” / `DBA` Dungra Bhili-Ambadungar / left column: source prints `0 no entry`.
2. Physical PDF p. 58 / printed p. 52 / item 116 “girl” / `DBA` Dungra Bhili-Ambadungar / left column: source prints `0 no entry`.

These rows have empty manual transcription and cognate-label fields, `Review_Status=source_blank`, and are excluded from staging.

## Validation

`items_109_135_hand_keyed.tsv` is regenerated only from the literal hand-entered decisions in `hand_keyed_items_109_135.py`. The guarded source-local importer accepts 459 rows and stages 457 target response candidates. Focused tests check exact coverage, disjoint item/site keys, NFC, OCR-blind schema, mandatory declaration, blanks, page continuations, response counts, cumulative accounting, and rejection of OCR-bearing or undeclared ledgers.
