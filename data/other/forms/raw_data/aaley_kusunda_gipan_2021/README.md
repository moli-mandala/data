# Aaley 2021 *Kusunda Gipan* glossary

This package installs the final Kusunda--Nepali glossary in Uday Raj Aaley's
2021 teaching book *Kusunda Gipan*, printed pages 48--50 (physical PDF pages
54--56).

## Scope and rights

- Source census: all 160 printed table rows.
- Installed census: 161 distinct attestations. The two slash groups (`घै/गहि`
  “wound” and `मुङ/मोङ` “king”) are split and linked as sound variants. The
  second exact `पाङजाङ` “five” row is retained in the 162-row audit but not
  installed twice.
- The Language Commission publication is all-rights-reserved and its PDF is
  not checked in. Its SHA-256 is pinned in `manifest.json`.
- The glossary is the book's canonical deduplicated lexical inventory.
  Pedagogical sentences, exercises, and repeated lesson vocabulary are not
  additional dictionary entries and are outside this glossary ingest.

## Extraction and transcription

The three ruled tables were extracted structurally with `pdfplumber`. Both the
raw Preeti glyph codes and decoded Unicode Devanagari are frozen in
`snapshot/glossary.tsv`. The open Preeti 1.0.1 converter was used as an
extraction aid only and is not vendored; all three rendered glossary pages were
checked. One converter-produced doubled virama in printed `फ्याक्सम` was
corrected in the frozen Unicode ledger while its raw glyph string remains.

Printed Kusunda Devanagari is retained in `Native`. `Form`/`Phonemic` is an
explicitly graphemic romanization, not narrow IPA. Following the book's
six-vowel chart, Devanagari `अ` is romanized `ə` and `आ` as `a`; final inherent
schwa is suppressed. The audit retains each Nepali gloss verbatim, including
source spelling errors, while `Gloss` supplies a conservative English
translation.

## Reproduction

From `data/`:

```sh
python3 data/other/forms/raw_data/aaley_kusunda_gipan_2021/import_gipan.py
python3 data/other/forms/raw_data/aaley_kusunda_gipan_2021/import_gipan.py --install
```

Passing `--pdf PATH` additionally verifies the PDF checksum, 56-page census,
and the glossary table counts of 70, 74, and 16 rows. Normal offline rebuilds
use only the pinned ledger.
