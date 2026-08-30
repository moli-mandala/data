# SIL JLSR 2023-002 Eastern Gujari survey

Source package for Hugoniot, Polster, Ahmad, and Rajan's *A Sociolinguistic
Profile of Eastern Gujari*. The survey/comparative-table addendum of
`SOURCE_INGESTION_CHECKLIST.md` is active. The OCR-heavy addendum is not
applicable: Appendix B is born-digital Unicode. Its text layer was used only as
an extraction scaffold; every printed cell was visually checked against the
rendered canonical PDF.

## Canonical source and scope

- SIL archive: <https://www.sil.org/resources/archives/95899>
- Publisher PDF: `JLSR2023_002.pdf`, 9,149,165 bytes, 121 pages
- PDF SHA-256: `41352b2db97dbd059a1bc229a8ed370fed700c1726f3886a580cba586137475e`
- Included: Appendix B, physical PDF pp. 41--76; lexical matrix pp. 42--76
  (printed pp. 34--68)
- Matrix: 210 prompts x 15 lists = 3,150 conceptual cells

The 2023 publication reports 1996 fieldwork and a report originally dated
1997. The PDF is not redistributed. The SIL archive and PDF were accessed on
2026-08-28.

Eight Indian lists are new targets: Udhampur, Jammu, Chamba, Rampur,
Nalagarh, Dehra Dun, Kotdwara, and Haldwani (1,680 cells). Six Pakistan Gujari
lists are reprinted from *Sociolinguistic Survey of Northern Pakistan*, volume
3, and already exist under `SSNP-gojri-CHT`, `-SSW`, `-GLT`, `-KGH`, `-NAK`,
and `-CAK` (1,260 audit-only cells). Urdu is a 210-cell comparison control.
The primary SSNP forms are neither duplicated nor overwritten.

## Extraction and visual review

`extract_eastern_gujari.py` verifies the PDF hash/page count and deterministically
extracts `extraction_scaffold.tsv`. `reviewed_transcription.tsv` is the import
authority. All 35 lexical pages and all 3,150 cells were manually inspected at
180 dpi. No table cell is handwritten or image-only, and no OCR was used.
`finalize_review.py` records the already-completed visual decisions; it does
not perform recognition.

There are 33 explicit no-entry cells: 25 targets and eight controls/reprints.
These include all eight Indian lists for items 11, 23, and 24, plus Nalagarh
item 177. PDF p. 45/item 23/Urdu literally prefixes its no-entry notation with
`ʊrdu`; the full printed cell is preserved in the ledger and classified as a
blank. No cell is clipped, illegible, ambiguous, or unresolved.

Similarity-group digits remain notes only. A slash is expanded only when it
introduces another numbered response, so phrase-internal spaces remain intact.
The wrapped Nalagarh response at PDF p. 71/item 176 is one cell with three
responses. PDF p. 63/item 127/Jammu prints `bʌɾo` twice under groups 3 and 4;
both occurrences remain in the audit and the exact duplicate is installed once.

## Installation and results

The importer emits 1,753 forms from 1,655 attested target cells (1,754 printed
target alternatives minus one exact duplicate), preserving NFC source IPA in
both `Form` and `Phonemic`. All 3,150 cells remain in the audit. Similarity
numbers are never promoted to cognacy, etymology, borrowing, or graph claims.
`conversion/sil-eastern-gujari.txt` covers every installed grapheme.

## Reproduction

From the data-repository root, with the PDF cached in the outer workspace:

```sh
python data/other/forms/raw_data/sil_eastern_gujari_2023/extract_eastern_gujari.py \
  ../tmp/pdfs/eastern_gujari/JLSR2023_002.pdf \
  --output data/other/forms/raw_data/sil_eastern_gujari_2023/extraction_scaffold.tsv \
  --review-template data/other/forms/raw_data/sil_eastern_gujari_2023/reviewed_transcription.tsv
python data/other/forms/raw_data/sil_eastern_gujari_2023/finalize_review.py
python data/other/forms/raw_data/sil_eastern_gujari_2023/import_eastern_gujari.py --install
UV_CACHE_DIR=/tmp/uv-cache uv run --with pytest --with segments --with pypdf \
  pytest -q tests/test_sil_eastern_gujari_2023.py
```

Shared bibliography/dialect/profile routing, the full build and test suite,
generated CLDF, and browser QA are deferred to the coordinating task. Exact
proposed integration is in `INTEGRATION.md`.

