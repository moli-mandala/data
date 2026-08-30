# Deferred shared integration

The source-local package is complete. Root should integrate these artifacts
without adding duplicate source/language/dialect identities.

## Reference

Reuse the existing key `canvin2025` in `cldf/sources.bib`; enrich that entry to:

```bibtex
@article{canvin2025,
  author       = {Canvin, Maggie and Joseph, Nidhin and Manoj, M. R.},
  title        = {A Sociolinguistic Survey of the Mavilan Tulu Language in Northern Kerala},
  journal      = {Journal of Language Survey Reports},
  number       = {2025-005},
  year         = {2025},
  publisher    = {SIL Global},
  url          = {https://www.sil.org/resources/archives/111012},
  included     = {Appendix A.2: all 615 attested target forms; three comparison lists audited but excluded},
  provenance   = {data/tmp/pdfs/JLSR2025-005.pdf; SHA-256 d7675b86c9f083eb2389d268078325643979db4a520da776246cf8ecb5fdc629; source-local ledger data/other/forms/raw_data/sil_mavilan_tulu_2025},
  ocr          = {No accepted transcription is OCR-derived; all 1248 cells visually hand-reviewed from rendered pages},
  jambu_editor = {Aryaman Arora and OpenAI Codex}
}
```

## Language, dialect, and profile routing

No new language or dialect rows are needed. Reuse base language `markodi` and
the existing registered dialect IDs:

- MTP → `markodi_pannithadam`
- MTV → `markodi_vannarkadav`
- MTE → `markodi_ennappara`

Reuse `conversion/markodi.txt` and the existing `markodi` routing. A greedy
full-form check finds every one of the 615 staged forms covered; no profile
addition is proposed. `profile_inventory.tsv` is the preservation inventory.

## Form reconciliation

`staged_forms.tsv` contains 615 target forms and `staged_audit.tsv` contains all
1,248 source cells. The target count equals the existing
`data/other/forms/20260723-markodi.csv` count, but the manual source strings are
not identical: 556 exact matches and 59 visually rechecked literal-source
differences. Replace/reconcile by stable item + site identity; do not append a
second 615-row source.

The source asserts no etymological assignments, so source-local staged
`Parameter_ID` is blank. Preserve the independently curated assignments in
`data/other/forms/raw_data/markodi_etyma.csv` by joining them to item/gloss +
site after selecting the manual source transcription. Do not treat survey
similarity percentages as cognacy.

Source-local accounting is exact: 615 staged target forms; 615 attested control
cells excluded; 18 literal source blanks excluded (9 target, 9 control); 0
ambiguous/illegible; no items 209–210.

## Consolidated commands after shared edits

```bash
python3 data/other/forms/raw_data/sil_mavilan_tulu_2025/import_mavilan_tulu_2025.py --stage
python3 -m pytest -q data/other/forms/raw_data/sil_mavilan_tulu_2025/test_mavilan_tulu_2025.py
make all
python3 -m pytest -q
```

After the full build, review the Markodi graph, verify representative forms from
each of MTP/MTV/MTE in the browser, and confirm source links/notes expose the
printed item/site/page coordinates. These shared gates remain deferred here.
