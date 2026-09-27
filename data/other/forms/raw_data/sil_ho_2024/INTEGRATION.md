# Deferred shared integration for Varenkamp 2024 Ho

## Shared integration update — 14 September 2026

The frozen staging is now adapted into `data/other/forms/20260914-sil-ho.csv`: 2,900 rows under canonical language `ho`, with shared dialects, bibliography, formatted references and explicit profile routing. Read-only shared-CLDF verification on 21 September 2026 confirms every installed key, persistent ID, transcription layer, citation, dialect tag and source-defined variant edge. No database was rebuilt. A new full build, repository-wide suite and global retrospective audit remain unclaimed; see the current shared review for scope and evidence. See the [shared review](../../../../../source_checklists/20260914-manual-surveys-review.md) for counts, corrections, validation and remaining gates.

The source manifests and staged files remain frozen as extraction evidence. The
following text records that earlier source-local stage; its pending shared gates
are superseded only to the extent documented in the shared review.

## Frozen source-local documentation

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
