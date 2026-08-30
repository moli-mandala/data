# Noira 2015 manual audit: items 28–54

## Scope and method

- Source: ESR 2015-012, Appendix A3.
- Reviewed range: items 28–54, physical PDF pp. 38–44 / printed pp. 32–38.
- Sites per item: 17; conceptual cells: 27 × 17 = 459.
- Every cell was read directly from 400-dpi rendered page images and visually
  checked at adequate zoom. PDF text/OCR was not read, copied, transformed, or
  used to seed any accepted transcription.
- The ledger is OCR-blind and every row declares
  `hand-keyed-from-rendered-source; OCR-not-copied`.
- Multiple printed responses remain in source order and are expanded
  deterministically on ` | ` by the guarded importer. Printed cognate labels
  are retained one-for-one in `Source_Cognate_Labels`.

## Accounting

| Scope | Reviewed cells | Attested cells | Source blanks | Expanded responses |
|---|---:|---:|---:|---:|
| 14 target lists | 378 | 378 | 0 | 471 |
| 3 control lists | 81 | 81 | 0 | 127 |
| Total | 459 | 459 | 0 | 598 |

- Ambiguous cells: 0.
- Illegible/clipped cells: 0.
- Unresolved readings: none.
- Exclusions from target staging: all 81 Gujarati, Marati, and Hindi control
  cells (127 printed responses); retained in the audit ledger.

## Source typography retained

Printed hyphens that coincide with line wrapping were preserved rather than
silently interpreted: item 42 TKO `tʃʌnnigo-gedʒ`; item 50 GUJ
`meigɦɖ-ɦanuʃa`, MAR `indrʌɖ-ɦanuʂʌ`, and HIN `indrəɖ-ɦənuʂ`.
This is a diplomatic transcription decision, not an unresolved reading.

## Reproduction and validation

```sh
python3 manual_chunks/hand_keyed_items_028_054.py
python3 import_noira_2015.py \
  --ledger manual_chunks/items_028_054_hand_keyed.tsv
python3 -m pytest -q manual_chunks/test_items_028_054_hand_keyed.py
```

Generated ledger SHA-256:
`57c27027fdd4972c1b93db6fb09fef88699aa9e5bdb3dd9e6d64a1df05104b11`.
