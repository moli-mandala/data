# Noira 2015 manual audit: items 82–108

## Scope and method

- Source: ESR 2015-012, Appendix A3.
- Reviewed range: items 82–108, physical PDF pp. 50–56 / printed pp. 44–50.
- Sites per item: 17; conceptual cells: 27 × 17 = 459.
- Every cell was read directly from 400-dpi rendered page images and visually
  checked at adequate zoom. Difficult cells on physical pp. 54 and 56 were
  independently re-rendered and inspected at 800 dpi. PDF text/OCR was not
  read, copied, transformed, or used to seed or verify any lexical reading.
- The ledger is OCR-blind and every row declares
  `hand-keyed-from-rendered-source; OCR-not-copied`.
- Multiple printed responses remain in source order and are expanded
  deterministically on ` | ` by the guarded importer. Printed cognate labels
  are retained one-for-one in `Source_Cognate_Labels`.

## Accounting

| Scope | Reviewed cells | Attested cells | Source blanks | Expanded responses |
|---|---:|---:|---:|---:|
| 14 target lists | 378 | 378 | 0 | 454 |
| 3 control lists | 81 | 81 | 0 | 103 |
| Total | 459 | 459 | 0 | 557 |

- Ambiguous cells: 0.
- Illegible/clipped cells: 0.
- Unresolved readings: none.
- Exclusions from target staging: all 81 Gujarati, Marati, and Hindi control
  cells (103 printed responses); retained in the audit ledger.

## Diplomatic transcription decisions

- Item 97 NAS source `bhodʒijo` retains the printed `bh` sequence rather than
  normalizing it to an IPA aspiration character.
- Item 98 preserves the printed dental diacritic in HIN `mət̪tʃhər` and all
  repeated numeric cognate labels in the multi-response cells.
- Item 100 preserves apparent printed line-wrap compounds as hyphenated forms,
  including NTE `dʒagli-malja`, TKO `dʒʌgʌ-limʌlʌj`, and NJA
  `dʒʌgʌ-limalai`.
- Item 102 DBA `maɦũʔʊ` and KNA `t̪holja` were rechecked at 800 dpi; the
  glottal stop and dental diacritic are visibly printed and are not inferences.
- Items 107–108 preserve source spaces and hyphens, including HIN item 108
  `tʃhoʈəb-ɦai`, rather than silently regularizing compounds.

These are visually resolved source readings, not guessed normalizations or
unresolved cells.

## Reproduction and validation

```sh
python3 manual_chunks/hand_keyed_items_082_108.py
python3 import_noira_2015.py \
  --ledger manual_chunks/items_082_108_hand_keyed.tsv
python3 -m pytest -q manual_chunks/test_items_082_108_hand_keyed.py
```

Generated ledger SHA-256:
`ff1d6f982dcca3a1b59400c1064a913f1fedbbe824af1d35f43d0f17a061d9fe`.
