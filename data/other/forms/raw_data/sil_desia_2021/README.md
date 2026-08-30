# SIL Desia survey (JLSR 2021-056)

This package installs all nineteen Desia/Kotia-Adivasi Oriya comparative
wordlists in Gangadhar Behera's *A Sociolinguistic Survey among Desia-Speaking
People Groups in South Orissa, India*. The survey was created in 2007 and
published as SIL *Journal of Language Survey Reports* 2021-056.

## Canonical source and representations

- SIL archive record: <https://www.sil.org/resources/archives/91960>
- SIL publication record: <https://www.sil.org/resources/publications/entry/91960>
- canonical PDF: <https://www.sil.org/system/files/reapdata/54/86/16/54861697763004359352591752899754568865/JLSR2021_056.pdf>
- pinned archived retrieval: <https://web.archive.org/web/20240617131527id_/https://www.sil.org/system/files/reapdata/54/86/16/54861697763004359352591752899754568865/JLSR2021_056.pdf>
- SHA-256: `04de0004c1375955c1adbeb8941b187aa4fc88f484ee00e9bc69655813e6690b`
- 3,737,879 bytes; 158 pages

The SIL archive/publication records do not state a distinct reuse licence. The
package therefore preserves the pinned source identity and extracted lexical
facts but does not redistribute the PDF.

The checked source is `tmp/pdfs/desia-2021-056/source.pdf`; it is not
redistributed here. Appendix B.5 occupies physical PDF pages 80–127 (printed
pages 71–118).

The appendix is typeset, not handwritten or raster-only, and has an embedded
Unicode text layer. `text_layer_scaffold.txt` is only a locating/transcription
scaffold and is not read by the importer. `manual_review.tsv` is authoritative:
all 4,696 printed response lines were compared directly with the 48 rendered
pages, and `page_review.tsv` records that complete page-by-page visual pass.
No OCR output was used or needed.

The positional text extractor misplaced zero-width dental and nasalization
marks on 542 response lines. These were corrected from direct glyph inspection
and the PDF content-stream character order; `glyph_order_corrections.tsv`
records every raw scaffold/manual pair and exact locator. For example, the
positional scaffold `tu̪ i` is visibly/source-logically `t̪ui`, and `ɐk̃ ɪ` is
`ɐ̃kɪ`. The unmodified positional extraction remains in `Source_Text` and
`text_layer_scaffold.txt`; only the visually confirmed form is installed.

## Scope and accounting

Appendix B.5 has 210 prompts and nineteen Desia lists, one for each surveyed
community/site. All nineteen are target varieties under Jambu's existing
`AdivasiOriya` base language; there are no external comparison controls.

- 3,990 conceptual site/item cells manually accounted;
- 4,696 printed response lines visually reviewed;
- 4,658 attested response lines;
- 38 explicit `no entry` cells (all sites for items 23 `urine` and 24 `feces`);
- 4,655 installed forms after three literally identical duplicate readings
  printed under two different similarity-group labels are merged;
- zero clipped, illegible, ambiguous, or unresolved readings.

One attested response has a genuinely blank similarity-group column: PDF p.103
(printed p.94), item 109 `older sister`, Ghumar `apa`. The blank is preserved
and is not inferred. Three cells repeat one identical form under groups 1 and 2:
item 113/Souraguda and item 138/Konda Maliguda and Patta Maliguda. Each yields
one installed form whose Notes and audit preserve both group labels.

The source's group numbers are Wordsurv lexical-similarity judgments, not
etymologies or cognate sets. They occur only in Notes/audit; all installed
`Parameter_ID`, `Cognateset`, and `Etymology` fields are blank.

The report explicitly labels the response column `T-IPA`. Consequently the raw
CSV preserves that source transcription in both raw `Form` and `Phonemic`;
after the shared profile route is integrated, compiled `Original`/`Phonemic`
retain the source IPA while compiled `Form` receives the house transcription.
Representative profile decisions are `tʃ → c`, `dʒ → j`, dental `̪ → ∅`,
retroflex `ɖ ɳ ɭ ɽ → ḍ ṇ ḷ ṛ`, source colon `: → ː`, and long nasal vowels such
as `ãː → ā̃`. Aspiration and superscript vowels are preserved.

Appendix B.4, Table 1, and B.5 disagree about several village/block labels.
`metadata_discrepancies.tsv` records every discrepancy used in editorial
decisions. In particular, B.5's `Dame side` is preserved verbatim and linked
conservatively to the Dom list; it is never silently corrected to “Dom side.”

## Reproduction and checks

```sh
sh data/other/forms/raw_data/sil_desia_2021/render_wordlist.sh
UV_CACHE_DIR=/tmp/uv-cache uv run --with pdfplumber python data/other/forms/raw_data/sil_desia_2021/extract_scaffold.py
# Do not overwrite manual_review.tsv without repeating the full visual review.
python3 data/other/forms/raw_data/sil_desia_2021/import_desia.py --verify-pdf --install
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_sil_desia_2021.py
```

Shared bibliography/dialect registration, profile routing, full build, full
test suite, graph checks, and browser QA are intentionally deferred and
specified exactly in `INTEGRATION.md`.
