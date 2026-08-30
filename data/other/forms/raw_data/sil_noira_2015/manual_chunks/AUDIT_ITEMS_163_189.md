# Manual review audit — Noira items 163–189

- Scope: 27 prompts × 17 sites = 459 conceptual cells.
- Source coordinates: physical PDF pp. 67–74 / printed pp. 61–68.
- Method: every cell hand-keyed from 400-dpi rendered page images; difficult glyphs visually rechecked on 900-dpi renders. PDF text/OCR was not read, copied, transformed, or accepted as transcription.
- Declaration on every ledger row: `hand-keyed-from-rendered-source; OCR-not-copied`.
- Ledger SHA-256: `e462c7cea69be36036982fe82f7f8eca0121747cb6125f6437354db5bd4161f4`.

## Accounting

- Reviewed cells: 459.
- Attested cells: 453.
- Source blanks: 6.
- Expanded responses: 607 (504 target; 103 Gujarati/Marati/Hindi controls).
- Ambiguous cells: 0.
- Illegible cells: 0.
- Unresolved readings: none.

## Source-explicit blanks

All six entries print `0 no entry`; none was inferred from whitespace.

| Physical PDF page | Printed page | Item | Site | Column |
|---:|---:|---:|---|---|
| 69 | 63 | 172 | DBA | right |
| 69 | 63 | 173 | DBA | right |
| 70 | 64 | 174 | DBA | left |
| 70 | 64 | 177 | KTA | right |
| 70 | 64 | 177 | NTE | right |
| 70 | 64 | 177 | NJA | right |

## Diplomatic decisions and high-resolution rechecks

- Physical p. 68 / printed p. 62, item 165 DBA: `kɔrɔ` confirmed at 900 dpi.
- Physical p. 69 / printed p. 63, item 170 NGO: `kɛhinʌhɔi`; NTO: line-wrapped `kolakh-dzat̪iɳ`; MAR: line-wrapped `konʈiɖ-prʌkʌrtse`.
- Physical p. 70 / printed p. 64, item 178 BMU: source-printed `puʈiio` retained.
- Physical p. 71 / printed p. 65, item 179 DBA: `t̪huɽɔ` confirmed.
- Physical p. 71 / printed p. 65, item 183 BMU: a printed backslash between `sauwio` and `sau` is treated as the source's response separator; the lexical forms are retained separately and the editorial decision is explicit here.
- Physical p. 72 / printed p. 66, item 184 NAS: `pukh̪lagi` confirmed.
- Identical source-printed alternatives with different numeric group labels remain duplicated in source order (not silently collapsed), notably items 181 and 188.
- Line-wrapped words and clauses are joined without inserting spaces at typographic hyphenation; source lexical hyphens are retained.

## Guard and tests

- The guarded importer accepted the ledger: 459 reviewed cells, 453 attested cells, 6 source blanks, and 504 target form candidates.
- Seven cumulative focused-test modules passed: 35 tests.
- The tests require NFC, an OCR-blind schema, the exact reviewer declaration, unique item/site keys, explicit physical/printed page and column coordinates, disjoint cumulative chunks, and manifest agreement.
