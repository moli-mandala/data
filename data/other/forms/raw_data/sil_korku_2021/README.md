# SIL Korku survey (JLSR 2021-040)

This package extracts Appendix F of James Stahl's *A Sociolinguistic Survey of
the Korku [kfq] Language Area*. The research was conducted in 1985 and
published by SIL International as **Journal of Language Survey Reports
2021-040**.

- Canonical record: <https://www.sil.org/resources/archives/90546>
- Canonical PDF: <https://www.sil.org/system/files/reapdata/77/13/79/77137924173691528074907383931327079270/JLSR2021_040.pdf>
- PDF SHA-256: `d17426da3788d66c95f05824483941e7d5468e154c66d43c6354262fda00190d`
- PDF extent: 102 pages

The report's Appendix F introduction (PDF 43, printed 38) says that **nine**
locality lists are reproduced, correcting the earlier discovery-tracker
description of five Korku lists plus Nihali. The appendix contains 210 item
slots for eight Korku localities and one Jammat Jalgaon Nihali comparison list:

- PDF 46–56: Chikli Ruma / Khanapur Ruma
- PDF 57–67: Bagdara Ruma / Warsari Ruma
- PDF 68–78: Moragao Bouriya / Lahi Bouriya
- PDF 79–89: Amdhana Mawasi / Khamalpur Bondoy
- PDF 90–100: Jammat Jalgaon Nihali comparison control

The phonetic chart and wordlists (PDF 44–100) are image-only. `extract_ocr.py`
verifies the pinned PDF, extracts and enhances its embedded scans, and
reproduces `tesseract_raw.txt`. OCR is evidence only. It was never accepted as
a transcription. The checked scaffold was produced with Tesseract 5.5.2
(English model, page-segmentation modes 4 and 6), pypdf 6.15.0, and Pillow
11.3.0; the script records the exact preprocessing operations.

`manual_review_data.py` is the authoritative ledger. Every one of the 1,890
source cells was visually reviewed against the scan: 1,680 target Korku cells
and 210 Nihali comparison cells. The ledger records 218 ruled blanks (216 in
target lists, two in the control). One target cell is unreadable and excluded:
PDF 83 / printed 78, item 93 “tail”, Amdhana Mawasi. The scan shows only faint,
clipped marks, so no form is guessed. All other faint, clipped, questioned, or
collector-marked readings remain explicitly flagged in the audit with their
page/item/site coordinates.

`import_korku.py --install` expands the ledger into:

- `data/other/forms/20260828-sil-korku.csv`: 1,521 rows from 1,463 attested
  target cells; slash alternatives become separate stable rows.
- `data/other/forms/raw_data/20260828-sil-korku-audit.csv`: all 1,890 cells,
  including 217 missing target cells and 210 excluded Nihali controls.
- `data/other/forms/raw_data/20260828-sil-korku-manifest.json`: source/OCR/
  manual-ledger hashes, topology, counts, and policy.

The survey's reported lexical-similarity percentages are not historical
cognacy judgments. Installed rows therefore have no cognacy or etymology links.
The source notes that five survey workers used varying transcription systems;
the forms preserve the manually read source distinctions, with transparent
broad-IPA conversions and uncertainty notes rather than silent guesses.

The copyrighted PDF and page images are not checked in. The package contains
only extracted lexical facts, review evidence, and the reproducible code and
metadata needed for scholarly verification.

Focused validation:

```sh
python data/other/forms/raw_data/sil_korku_2021/import_korku.py --install
UV_CACHE_DIR=/tmp/uv-cache uv run --with pytest --with segments pytest -q tests/test_sil_korku_2021.py
```

Shared registry/build integration is deferred to the coordinating task and is
specified exactly in `INTEGRATION.md`.
