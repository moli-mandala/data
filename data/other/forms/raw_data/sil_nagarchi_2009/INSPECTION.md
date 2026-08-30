# ESR 2009-010 Nagarchi: source inspection

Checked 2026-08-28 under the survey-wordlist/comparative-table and OCR-heavy-source
gates in `data/SOURCE_INGESTION_CHECKLIST.md`.  The complete official publication
reports an unsuccessful attempt to collect Nagarchi wordlists and contains no lexical
data to ingest.  The extraction, sound-profile, language/dialect, CLDF-build, and
browser-entry gates are therefore genuinely inapplicable rather than deferred.

## Pinned source

- George, Symon, Thangmualian Valte, and Eldose K. Mathai. 2009. *A
  Sociolinguistic Survey Among the Nagarchi Community of Central India*.
  Electronic Survey Reports 2009-010.
- Official SIL archive record: <https://www.sil.org/resources/archives/9044>.
- Official publisher PDF:
  <https://www.sil.org/system/files/reapdata/16/82/66/168266457676336418888975638813476176422/silesr2009_010.pdf>.
- Inspected workspace file: `tmp/pdfs/nagarchi/silesr2009_010.pdf`.
- SHA-256:
  `12ecc9f2e7014f1b35c19ca90e8ed0178fe6f5dfb2c7dcbf80609b46e3995580`.
- File size: 96,621 bytes; five physical and numbered pages.
- Rights: the title page says copyright © 2009 by the three authors and SIL
  International, all rights reserved.  The PDF is not redistributed in the
  source package.

## Complete-publication audit

The born-digital text layer was extracted from all five pages, and every rendered
page was inspected visually.  The contents are exhaustive:

1. Purpose and goals
2. Geography
3. People
4. Languages
5. Findings based on survey fieldwork
6. Conclusions
7. References

There is no appendix, prompt list, phonetic transcription, lexical response,
lexical-similarity matrix, or denominator.  The only table-like visual is a survey-area
map on page 3.  The abstract and section 1 identify “attempted wordlist collection” as
a research tool.  Crucially, section 5 on page 4 states that the researchers attempted
to collect Nagarchi wordlists in many areas, but could not do so because even elderly
Nagarchi consultants were unable to provide them.  Thus there is no unpublished-looking
table hidden by a bad text layer: the source explicitly says that the collection failed.

## Editorial decision and checklist disposition

- Installed lexical rows: **0**.
- Exclusions: **0 lexical cells**, because no wordlist responses were collected or
  published.
- Unresolved readings: **0**; no IPA or other lexical transcription appears.
- Transcription decision: none required.  OCR was not needed; the text layer and
  full-page renders agree on the publication topology.
- Metadata/reference/dialect/profile decision: do not add a lexical-source
  bibliography entry, pseudo-dialect rows, or a sound profile for a source that
  contributes no forms.  The pinned archive/PDF metadata remains in this inspection
  record and the survey census.
- Validation: complete five-page text and visual audit.  There can be no focused
  importer test, CLDF row, graph node, or representative browser entry for this
  source.

This closes ESR 2009-010 as an inspected failed-wordlist survey, not an unfinished
OCR candidate.
