# SIL JLSR 2022-015 Bagheli survey

Source package for Binoy Koshy's *A Sociolinguistic Study of Bagheli Speakers
in Madhya Pradesh*. The package installs Appendix B.4's Bagheli survey lects and
audits the neighbouring Standard Hindi comparison list. It activates the
survey/comparative-table and OCR-heavy addenda of
`SOURCE_INGESTION_CHECKLIST.md`.

## Canonical source and scope

- SIL archive: <https://www.sil.org/resources/archives/94596>
- Publisher PDF: `JLSR2022_015.pdf`, 161 pages
- PDF SHA-256: `d1424f317dc12fe01d99d33abd917201575487f4de44529678ecce1c282a4627`
- Included: Appendix B.4, physical PDF pages 59--81, printed pages 50--72
- Table: nominally 210 prompts and 19 lists (18 Bagheli lects plus Standard
  Hindi); the printed table wholly omits prompts 23 and 24
- Bibliography cross-check: Glottolog reference 664025,
  `hh:wsoc:Koshy:Bagheli`, classified as wordlist/sociolinguistics

The report is © 2022 SIL International. Its printed fair-use policy permits
copies for scholarly research and instruction, but prohibits republication and
commercial use without written consent. The PDF is therefore not checked in.
This package preserves bibliographic metadata, reproducible hashes, OCR
comparison evidence, and extracted linguistic facts. It was accessed on
2026-08-28.

The existing Jambu Bagheli material is the Lexibank LSI wordlist attached to
`bagheli_lakshman`. This package is a distinct SIL survey with stable site and
page/item provenance; it supplements rather than replaces the LSI source.

## Image extraction and manual transcription

Appendix B.4 has no usable text layer: each page contains a lossless embedded
table image. `extract_ocr.py` verifies the PDF hash, extracts the 23 images,
crops their four columns, and can reproduce `image_manifest.tsv` and the 92
block `tesseract_scaffold.txt`. The OCR is deliberately non-authoritative.

`manual_transcription.txt` is the sole lexical authority. Every printed response
line and every absent site/item cell was inspected against the enlarged source
image, column by column, across all 23 pages. No form is copied from or installed
on the authority of OCR. Compact bracketed site sets are expanded only after the
form and every printed code have been visually checked. Source IPA is retained
diplomatically and normalized only to NFC; parenthetical qualifiers and
footnote markers are split into audit/notes fields.

The manual ledger contains 2,284 attested response lines, expanding to 6,111
response occurrences. It also contains two explicit `by name` non-lexical
directives (24 response occurrences) and two legible but unassigned response
lines. The importer adds the remaining visually confirmed absent cells, yielding
exactly 3,990 reviewed conceptual cells (210 x 19): 3,933 with at least one
lexical form, 10 with only a `by name` response, and 47 genuinely blank. Because
a site may have multiple printed alternatives, response occurrences outnumber
conceptual cells.

## Installation policy and results

All 18 survey lects attach to the existing base language `bagheli_lakshman` and
are proposed as source-local dialects. Standard Hindi is a comparison control
and is retained only in the audit. Similarity-group numbers are descriptive
metadata, not cognate or etymology claims.

`import_bagheli.py --install` deterministically emits:

- 5,828 Bagheli form occurrences to
  `data/other/forms/20260828-sil-bagheli.csv`;
- 6,184 audit rows, including 283 Hindi control occurrences, 47 confirmed blank
  conceptual cells, 24 `by name` non-lexical occurrences, and two unassigned
  source lines;
- `20260828-sil-bagheli-manifest.json`, pinning sources, artifacts, decisions,
  and counts.

Item 173 site `m` prints the comma-separated alternatives `je,e`. The raw
source-local CSV preserves that one printed response as one row; the shared
form parser correctly expands it to separate compiled forms `je` and `e`, so
the expected post-build count is 5,829 rather than 5,828.

`conversion/sil-bagheli.txt` covers every installed grapheme. It maps the
source's affricates, retroflexes, vowel length, aspiration, and dental mark to
Jambu display transcription while the exact manually transcribed IPA remains in
`Phonemic`/`Original` after shared integration.

## Unresolved and editorial decisions

- Item 191 `berəʈɛ` and item 195 `reŋeʈe` are legible, but their source lines
  print no bracketed site code. They are excluded with high transcription
  confidence and unresolved assignment.
- Item 189 site `a` prints `bejtʰe?`. The legible form is installed without the
  punctuation; the source's question mark is retained as a source-uncertainty
  qualifier.
- Item 121 prints uppercase `L` in `[LSahms]`, although Appendix B defines only
  lowercase site `l`. It is interpreted as `l`, with medium site-code confidence
  on that one expanded cell.
- Superscript `1` after item 73 `bēgen`, item 102 `menə`, and item 194
  `uɖəʈʰ he` (plus the two item 189 forms `bet`, `betʰ`) is retained as a source
  footnote marker in notes, not treated as a segment or tone.
- The report inconsistently labels code `t` as Thurua/Dabhaura/Mahdeiya in its
  surrounding material. The wordlist site is registered as Mahdeiya
  (Singraulihi), and the inconsistency remains explicit in its locality note.

## Reproduction

With the canonical PDF cached at
`../tmp/pdfs/bagheli/JLSR2022_015.pdf` relative to the outer workspace:

```sh
python data/other/forms/raw_data/sil_bagheli_2022/extract_ocr.py
python data/other/forms/raw_data/sil_bagheli_2022/import_bagheli.py --install
UV_CACHE_DIR=/tmp/uv-cache uv run --with pytest --with segments pytest -q tests/test_sil_bagheli_2022.py
```

The source package intentionally defers the shared bibliography/dialect/profile
routing edits, `make all`, full pytest, and browser rebuild to the coordinating
task. Exact proposed integration is in `INTEGRATION.md`.
