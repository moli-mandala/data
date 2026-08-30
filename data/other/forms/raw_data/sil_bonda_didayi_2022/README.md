# SIL Bonda/Didayi survey wordlists (JLSR 2022-004)

This package ingests Appendix B of Mathew and Chamberlain's 1997 survey,
published by SIL International in 2022. The pinned canonical PDF is the SIL
archive copy (record 92608), SHA-256
`bb0548b4324224260b9618786dfd3aa40377138d0fbf4ae14c796df82f6190ce`.
The PDF is not redistributed here.

Appendix B is PDF pages 21–50 (printed 16–45): 210 prompts × thirteen lists =
2,730 conceptual cells. Nine Bonda/Didayi lists are targets (1,890 cells); four
Gutob/Parenga/Rona Desiya/Oriya lists are comparison controls (840 cells).
All 30 rendered pages and every cell were visually inspected. The appendix is
born-digital Unicode; no OCR was used and no handwriting/image-only IPA occurs.

`extract_pdf.py` pins the PDF checksum and recreates `extracted_cells.tsv`.
`import_bonda_didayi.py --install` recreates the installed forms, exhaustive
audit, and manifest. The importer strips lexical-similarity group numbers from
forms but retains them as descriptive notes only; it does not assert cognacy.
Comma-separated source responses are installed as variants, while spaces and
hyphens inside phrases are preserved. Four source-disqualified prompts (11,
23, 24, 70) and every explicit no-entry/dash response are not installed.

One cell is unresolved only because the source physically omits it: PDF 45,
printed 40, item 174 `those`, Orapadar U. Didayi. No form is guessed. The audit
also preserves the printed label defects `Kaluguda U.` at item 174 and
`Orapadar U. Diday` at item 208. A single broken embedded-text glyph at item 50,
Chitrakonda was manually read from the rendered page as `bɾihumhaiʒã`; its audit
status records the visual correction.

Rebuild locally:

```sh
UV_CACHE_DIR=/tmp/uv-cache uv run --with pdfplumber python \
  data/other/forms/raw_data/sil_bonda_didayi_2022/extract_pdf.py
python3 data/other/forms/raw_data/sil_bonda_didayi_2022/import_bonda_didayi.py --install
UV_CACHE_DIR=/tmp/uv-cache uv run --with pytest --with segments pytest -q \
  tests/test_sil_bonda_didayi_2022.py
```

Shared registry/build changes and browser QA are intentionally deferred to the
consolidating agent; see `INTEGRATION.md`.
