# SIL Kullu survey wordlists (JLSR 2021-009)

This package reproducibly ingests Appendix C of Frank Blair's *A
Sociolinguistic Profile of Kullu District, Himachal Pradesh* (SIL,
JLSR 2021-009). The source is the canonical SIL archive record
<https://www.sil.org/resources/archives/88003>; its 126-page PDF is pinned by
SHA-256 `720a97198254160bfa88a9557b33955b2814878e346901ff399cacc53d5c4fdd`.
The PDF itself is not checked in because SIL permits scholarly fair use but
prohibits republication. `extract_kullu.py` verifies the downloaded file before
extracting page images and deterministic cell crops.

## Source topology and review

Appendix C contains 198 numbered prompts (notwithstanding the report's
“200-item wordlist” description) and 16 Pahari survey sites: 3,168 target
cells. There is no Hindi response column. On the first wordlist sheet, `Hindi`
labels the otherwise blank item-number gutter; the audit records this layout
fact separately and excludes it from the lexical topology.

Every one of the 3,168 handwritten/image-only cells was inspected manually
against the full-resolution source page and deterministic crop. The
authoritative ledger is `manual_pages.tsv`: 2,753 cells contain transcriptions
and 415 are explicitly confirmed blank. Two readings retain reviewer
ambiguity flags and three retain source question marks. No cell is transcribed
from OCR alone. `vision_raw.tsv`, `vision_cells_raw.tsv`, and the `Raw_OCR`
fields of `transcription.tsv` are retained only as non-authoritative comparison
evidence.

The diplomatic normalization uses Unicode IPA: stress `ˈ`, length `ː`,
aspiration `ʰ`, source underlining as the corresponding dental or retroflex
segment where legible, and NFC throughout. Source parenthetical qualifiers are
preserved. Slash-separated source alternatives are retained in the ledger and
split into stable installed variants by the importer. Small handwritten
lexical-similarity group numbers are excluded from forms and are not treated as
historical cognacy. Cognateset and etymology fields therefore remain blank.

## Reproduction

Download the canonical PDF to
`../tmp/pdfs/kullu/JLSR2021_009.pdf` relative to the workspace root, then run:

```sh
UV_CACHE_DIR=/tmp/uv-cache uv run --with pypdf --with pillow \
  python data/other/forms/raw_data/sil_kullu_2021/extract_kullu.py \
  --output-dir /tmp/kullu-extract \
  --scaffold /tmp/kullu-extract/transcription.tsv
python data/other/forms/raw_data/sil_kullu_2021/import_kullu.py --install
UV_CACHE_DIR=/tmp/uv-cache uv run --with pytest --with segments pytest -q tests/test_sil_kullu_2021.py
```

The importer writes 2,963 installed rows after splitting slash alternatives,
a 3,169-row audit (3,168 lexical cells plus one layout record), and a manifest.
It asserts the complete site/item sequence, manual review state, explicit blank
accounting, and NFC normalization. `INTEGRATION.md` contains the exact shared
registry and routing changes intentionally deferred during parallel ingestion.

## Files

- `manual_pages.tsv`: authoritative hand transcription/review ledger.
- `prompts.tsv`: the 198 English elicitation prompts.
- `transcription.tsv`: cell topology plus OCR comparison scaffold; never authoritative.
- `extract_kullu.py`: verified page/cell extraction.
- `vision_ocr.swift`, `merge_*_scaffold.py`, `vision*_raw.tsv`: reproducible OCR evidence path.
- `import_kullu.py`: deterministic installer and audit/manifest builder.
- `../../20260828-sil-kullu.csv`: installed source rows.
- `../20260828-sil-kullu-audit.csv`: cell-level audit.
- `../20260828-sil-kullu-manifest.json`: pinned provenance and review counts.
