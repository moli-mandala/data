# SIL ESR 2018-010 Nilgiri Irula

Ingestion of Appendix B, the 187-item comparative word lists in Sylvia Ernest,
Clare O'Leary and Juliana Kelsall's *A Sociolinguistic Survey of Nilgiri Irula*.
The survey was carried out in 1992-1993, completed as an unpublished report in
1993, and published unchanged as SIL Electronic Survey Report 2018-010.

This source uses both the survey-wordlist and OCR-heavy addenda of
`SOURCE_INGESTION_CHECKLIST.md`.  The 11 Irula sites are target lects and must be
registered as dialects beneath `Irula`.  Coimbatore Tamil and the MAD, Kannada,
Badaga, Alu Kurumba, Betta Kurumba and Jenu Kurumba comparison rows are controls;
they remain accounted for in the audit and are not installed from this source.

## Extraction

The publisher PDF's Appendix B has no text layer.  Each of its 24 pages is a
67-74 dpi embedded raster image with three typewritten columns.  The tiny source
image is preserved by SHA-256 and every installed form retains its PDF/printed
page, column, item and site locator.  `extract_ocr.py` renders and crops the
columns and makes a reproducible Tesseract pass.  OCR is a structural and
base-letter scaffold only: it does not reliably recover the report's IPA
diacritics.

The authoritative transcription is reviewed against enlarged source crops.
The clean Unicode IPA from the overlapping 2015 Palakkad Kunjapana word list is
kept as review evidence where applicable, but never substituted for what the
1992-1993 list prints.  Any source mark that cannot be resolved from its pixels,
parallel Irula attestations, and the report's own transcription inventory is
retained as a typed transcription uncertainty in the per-record audit.

## Source identity

- SIL archive: 76656
- Publisher file: `silesr2018_010.pdf` (3.5 MB, 69 PDF pages)
- SHA-256: `2e5a4ef0f4c941437d09a1c8fa49ba01d4fe79e0915ad9248ac7b83280fb4c62`
- Appendix B: PDF pages 30-53, printed pages 25-48, 187 prompts
- Source date: fieldwork 1992-1993; report completed 1993; publication 2018

## Installed artifacts

`import_irula.py` verifies the OCR/review topology and emits 2,054 recoverable
Irula forms to `data/other/forms/20260828-sil-nilgiri-irula.csv`.  Its complete
3,417-row audit retains all 15 target gaps, 1,319 comparison-list records, and
29 classified layout fragments.  The accompanying manifest records source
identity, hashes, counts, transcription policy, and typed uncertainty totals.

The source's lexical-similarity groups are descriptive metadata only.  This
ingest creates no cognate, etymological, borrowing, or derivational edges.
