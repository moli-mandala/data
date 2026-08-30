# Noira 2015 manual audit: items 55–81

## Scope and method

- Source: ESR 2015-012, Appendix A3.
- Reviewed range: items 55–81, physical PDF pp. 44–50 / printed pp. 38–44.
- Sites per item: 17; conceptual cells: 27 × 17 = 459.
- Every cell was read directly from 400-dpi rendered page images and visually
  checked at adequate zoom. PDF text/OCR was not read, copied, transformed, or
  used to seed or verify any lexical transcription.
- The ledger is OCR-blind and every row declares
  `hand-keyed-from-rendered-source; OCR-not-copied`.
- Multiple printed responses remain in source order and are expanded
  deterministically on ` | ` by the guarded importer. Printed cognate labels
  are retained one-for-one in `Source_Cognate_Labels`.

## Accounting

| Scope | Reviewed cells | Attested cells | Source blanks | Expanded responses |
|---|---:|---:|---:|---:|
| 14 target lists | 378 | 378 | 0 | 463 |
| 3 control lists | 81 | 81 | 0 | 111 |
| Total | 459 | 459 | 0 | 574 |

- Ambiguous cells: 0.
- Illegible/clipped cells: 0.
- Unresolved readings: none.
- Exclusions from target staging: all 81 Gujarati, Marati, and Hindi control
  cells (111 printed responses); retained in the audit ledger.

## Diplomatic transcription decisions

- Item 61 MAR source `dzhʌɖʌ` retains the printed `dzh` sequence rather than
  being normalized to an affricate character.
- Item 66 NCH/NPN and related responses retain the unusual printed sequence
  `phʌlvɔ`.
- Apparent line-wrapping hyphens are preserved in item 74 NTO
  `buimun-gjanɖana` and MAR `bwɦimu-gaʈʃa`.
- Item 75 source cedilla forms (`miriçɔ`, `mirçe`) and item 76 KTA `ajjɖ` are
  retained exactly as printed.

These are visually resolved source readings, not guessed normalizations or
unresolved cells.

## Reproduction and validation

```sh
python3 manual_chunks/hand_keyed_items_055_081.py
python3 import_noira_2015.py \
  --ledger manual_chunks/items_055_081_hand_keyed.tsv
python3 -m pytest -q manual_chunks/test_items_055_081_hand_keyed.py
```

Generated ledger SHA-256:
`30a6087035d3c00a682f2e90c473da7aca46e42f4b97f5367b4132f7bda68e98`.
