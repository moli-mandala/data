# SIL Koya survey (JLSR 2021-029)

This package extracts Appendix E of *A Sociolinguistic Survey of Koya
Dialects*. The survey was conducted in 1985–1986 and published by SIL
International as **Journal of Language Survey Reports 2021-029**.

Canonical record: <https://www.sil.org/resources/archives/88873>

Canonical PDF: <https://www.sil.org/system/files/reapdata/10/79/91/107991682663957550005886358528752432250/JLSR2021_029.pdf>

PDF SHA-256: `a6541e0d2397849ce7c36961b3849f3b2c1f1c267036cfa1a3f6025796e14e7d`

The PDF is 124 pages. Ordinary text extraction works for the report, but the
phonetic alphabet and wordlists on PDF pages 80–123 (printed pages 75–118) are
raster images. `extract_ocr.py` verifies the pinned PDF, extracts and enhances
the embedded scans, and can reproduce `tesseract_raw.txt`. OCR is retained only
as comparison evidence. It was not used to install an unreviewed form.

`manual_review_data.py` is the authoritative cell ledger. Every one of the
1,840 cells actually printed in the seven target and two control lists was
visually inspected against the enhanced source image. It also records 50
conceptual list slots omitted by the report: Malakanagiri items 61–80 and
Jaganathapuram, Chintoor, and Podia items 201–210. Empty strings are confirmed
ruled blanks or identified omissions, never OCR failures.

`import_koya.py --install` expands the ledger to:

- `data/other/forms/20260828-sil-koya.csv`: 1,438 rows representing 1,401
  attested target cells; source slash alternatives are separate stable rows.
- `data/other/forms/raw_data/20260828-sil-koya-audit.csv`: all 1,890 conceptual
  site × item slots, including 420 excluded Telugu/Oriya controls and 69 missing
  target slots.
- `data/other/forms/raw_data/20260828-sil-koya-manifest.json`: pinned source,
  counts, review policy, and uncertainty summary.

The source gives lexical similarity group numbers, not etymological claims.
All installed forms therefore remain unlinked. The western wordlists overlap
the later Beine/Rama digitization; those values are recorded only in the audit
as alternate evidence after manual source-image checking.

SIL's publication page states a fair-use policy for scholarly research and
instruction. This package checks in extracted lexical facts and audit metadata,
not the copyrighted PDF or page images.

Run the focused checks with:

```sh
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_sil_koya_2021.py
```

Shared registry/build integration is intentionally described in
`INTEGRATION.md` and deferred to the coordinating agent.

