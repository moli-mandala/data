# SIL ESR 2019-005 Mudhili Gadaba

This directory contains the reproducible extraction for *A Sociolinguistic Survey of the
Mudhili Gadaba People of Andhra Pradesh* (Adimathara, Faith, Mathew and Vunnamatla 2019), SIL
Electronic Survey Report 2019-005. The publisher PDF is not redistributed; its expected SHA-256
is recorded in `20260828-sil-mudhili-gadaba-manifest.json`.

## Scope and results

Appendix A.3, printed pages 15–32 (PDF pages 19–36), contains a 210-item comparative wordlist.
Seven village lists are Mudhili Gadaba targets: Bobbilivalasa, Gogaduvalasa, Panukuvalasa,
Reyavanivalasa, Kothavalasa, Suregadivalasa and Chinachipuruvalasa. The Srikakulam Telugu list is
a comparison control and is deliberately excluded.

The checked source transcription contains 1,760 response records. The import installs 1,538
Mudhili Gadaba forms. Its audit also accounts for eight target cells printed `No Entry`, 214 Telugu
control responses, and five prompts printed `DISQUALIFIED` (11, 23, 32, 70 and 188). The source
makes no etymological claims, so every installed row is intentionally graph-neutral.

Artifacts:

- `transcription_pass2_quarter.tsv`: authoritative, source-facing transcription of all 210 items;
- `tesseract_raw.txt`, `ocr_scaffold.json`: structural OCR and its parsed scaffold;
- `import_gadaba.py`: deterministic importer and completeness assertions;
- `../20260828-sil-mudhili-gadaba-audit.csv`: complete per-record decisions;
- `../20260828-sil-mudhili-gadaba-manifest.json`: source/transcription hashes and pinned counts;
- `../../20260828-sil-mudhili-gadaba.csv`: installed rich-schema forms.

Run `make sil-gadaba` from the data repository to rebuild the installed file and audit without the
publisher PDF.

## Extraction and transcription

Appendix A.3 is an image-only 168-dpi scan. `crop_columns.py` renders the 18 wordlist pages and
cuts their three printed columns; Tesseract supplies only layout and ASCII scaffolding. Every
installed form was read visually from quarter-column crops (about nine printed lines per review
image). The earlier half-column pass is retained only as an independent, superseded reading.

The source distinguishes plain coronals from dental and retroflex coronals with very small raster
marks. A detached box below the letter is transcribed as dental `t̪ d̪ n̪`; an attached descending
tail is retroflex `ʈ ɖ ɳ`; no mark is plain `t d n`. Source vowel length, nasalisation, and
superscript vowel marks are preserved in `Phonemic`. Parenthetical number annotations and two
printed question marks are also preserved there and in notes, but removed from `Form` before its
explicit `sil-gadaba` display-profile conversion. Typed `source-raster-*` flags keep difficult
symbol classes filterable; they are review metadata, not claims that every flagged form is wrong.

## Source caveats and location policy

The report says the wordlists are field transcriptions that were not subjected to thorough
phonological analysis. They were gathered in several phases from 1995–1998 by different
elicitors, and the authors note that a second mother-tongue speaker did not check every list.
Lexical-similarity group numbers are therefore retained only as source notes and never converted
into historical-cognacy edges.

The report supplies villages and historical mandal assignments but no coordinates. Each survey
site is a registered dialect of canonical `Gadaba` (`mudh1235`) with a documented quality-C
mandal-centre approximation. In particular, the import preserves the report's 1996 assignment of
Panukuvalasa to Salur even though modern locality references are inconsistent.
