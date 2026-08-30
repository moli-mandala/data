# Deferred shared integration for Varenkamp 2024 Ho

Do not apply these proposals until the source-local importer completes and its
focused tests pass.

## Bibliography

```bibtex
@techreport{varenkamp2024ho,
  author = {Varenkamp, Bryan},
  title = {A Study of Ho Dialects},
  year = {2024},
  number = {2024-009},
  institution = {SIL International},
  series = {Journal of Language Survey Reports},
  url = {https://www.sil.org/resources/archives/100299},
  note = {Survey fieldwork conducted in 1989}
}
```

The existing language row `ho,Ho,hooo1248,23.96,87.12,Munda,,` is the parent
language. Proposed dialect rows are the fourteen target codes and locality
labels in `list_registry.tsv`; the source code should remain the stable dialect
identifier and the printed locality the display label. No rows are proposed
for HO1-HO3 or the ten comparison controls from this source.

Route the 2,900 rows in `staged_forms.csv` through a preservation profile that
keeps the reviewed NFC diplomatic Unicode transcription and strips only source
similarity-group labels. The source-local `symbol_inventory.tsv` is the coverage
contract. Do not infer cognacy from similarity numbers. Two ambiguous target
cells, 38 target blanks, all 630 republished-Ho cells, and all 2,100 non-Ho
comparison cells remain excluded.

Deferred commands:

```sh
python3 data/other/forms/raw_data/sil_ho_2024/import_ho.py --verify-pdf --stage
pytest -q tests/test_sil_ho_2024.py
make all
pytest -q
```
