# JLSR 2021-020 Ahirani: source inspection

Checked 2026-08-28 under the survey-wordlist/comparative-table and OCR-heavy-source
gates in `data/SOURCE_INGESTION_CHECKLIST.md`.  The source reports two 210-item
Ahirani wordlists and a Marathi comparison, but publishes only aggregate similarity
counts and percentages.  No lexical response is available to ingest, so the extraction,
sound-profile, language/dialect, CLDF-build, and browser-entry gates are genuinely
inapplicable rather than deferred.

## Pinned source

- Blair, Frank, with researchers P. Keder and L. Sanni. 2021 [research and
  manuscript dated 1987]. *A Preliminary Investigation into Ahirani
  Bilingualism*. Journal of Language Survey Reports 2021-020.
- Official SIL archive record: <https://www.sil.org/resources/archives/88614>.
- Official publisher PDF:
  <https://www.sil.org/system/files/reapdata/64/58/79/64587984906500660600369098330426075882/JLSR2021_020.pdf>.
- Inspected workspace file: `tmp/pdfs/ahirani/JLSR2021_020.pdf`.
- SHA-256:
  `0cdf41417a022eca443026ac71e2b03d550ba815c6a0f079d08048ecdbeefb45`.
- File size: 449,914 bytes; fifteen physical pages (cover, title, publication
  information, abstract, contents, preface, nine numbered report pages).
- Rights: the publication page says copyright © 2021 SIL International and gives
  the JLSR scholarly-research/instruction fair-use policy.  The PDF is not
  redistributed in the source package.

## Complete-publication audit

The born-digital text layer was extracted from all fifteen pages, and every rendered
page was inspected visually.  The complete contents end with References and contain
no appendix.  Section 4.1 says that the researchers elicited two 210-item Ahirani
wordlists, from Akkalkuva taluka and Dhule taluka, and compared them with Marathi.
Only the following aggregate results are printed:

| Comparison | Similarity | Similar items / denominator |
|---|---:|---:|
| Marathi–Akkalkuva | 66% | 138/210 |
| Marathi–Dhule | 63% | 128/202 |
| Akkalkuva–Dhule | 90% | 181/202 |

No prompt, response form, phonetic transcription, item-level judgement, or list
metadata beyond the two taluka labels is published.  The official SIL site search for
“Ahirani wordlist” returned no separate data-appendix record on the inspection date.

## Editorial decision and checklist disposition

- Installed lexical rows: **0**.
- Exclusions: the source-reported **two 210-item Ahirani lists and Marathi
  comparison are unavailable at item level**, so there are no lexical cells that can
  be installed or individually audited.  The six aggregate numbers above are
  sociolinguistic evidence, not lexical forms.
- Unresolved readings: **0**; the aggregate tables are legible and the report contains
  no IPA.
- Transcription decision: none required.  OCR was not needed; the text layer and all
  fifteen page renders agree on the publication topology and table values.
- Metadata/reference/dialect/profile decision: do not add a lexical-source
  bibliography entry, pseudo-dialect rows, or a sound profile for a source that
  contributes no forms.  The pinned archive/PDF metadata remains in this inspection
  record and the survey census.
- Validation: complete fifteen-page text and visual audit, including the contents,
  both lexical-similarity tables, and final references page.  There can be no focused
  importer test, CLDF row, graph node, or representative browser entry for this
  source.

This closes JLSR 2021-020 as an inspected aggregate-only survey report; it is not an
unfinished OCR candidate.  If the underlying 1987 field sheets are later published,
they would constitute a new source requiring their own ingestion package.
