# ESR 2008-018 Pardhan: source inspection

Checked 2026-08-28 under the survey-wordlist/comparative-table and OCR-heavy-source
gates in `data/SOURCE_INGESTION_CHECKLIST.md`.  The complete official publication
contains no lexical data to ingest, so the extraction, sound-profile, language/dialect,
CLDF-build, and browser-entry gates are genuinely inapplicable rather than deferred.

## Pinned source

- Valte, Thangmualian, Eldose K. Mathai, and Symon George. 2008. *A
  Sociolinguistic Survey Among the Pardhan Community of Central India*. SIL
  Electronic Survey Report 2008-018.
- Official SIL archive record: <https://www.sil.org/resources/archives/9039>.
- Official publisher PDF:
  <https://www.sil.org/system/files/reapdata/12/33/65/123365113542484887465191886041356894943/silesr2008_018.pdf>.
- Inspected workspace file: `tmp/pdfs/pardhan/silesr2008_018.pdf`.
- SHA-256:
  `ff72d156424ce3b33602f6698a812bc1630d7dfe61a237b48981b9419c966f3e`.
- File size: 414,926 bytes.  The PDF has five physical pages: one unnumbered
  title/copyright page followed by four numbered report pages.
- Rights: the title page says copyright © 2008 by the three authors and SIL
  International, all rights reserved.  The PDF is not redistributed in the
  source package.

## Complete-publication audit

The born-digital text layer was extracted from all five pages, and every rendered
page was inspected visually.  The numbered report is exactly the four pages stated
by the archive record.  Its contents are exhaustive:

1. Purpose and goals
2. Geography
3. People
4. Languages
5. Previous research
6. Findings based on survey fieldwork
7. Conclusions
8. References

There is no appendix, wordlist, elicitation table, phonetic transcription, lexical
item, lexical-similarity matrix, or denominator.  The only table-like visual is a
map of the Pardhan area on numbered page 2.  Section 1 explicitly identifies the
research tools as informal interviews, questionnaires, and observations.  Section
6 says that questionnaires and interviews were administered in several villages in
Seoni, Balaghat, and Mandla districts; it neither says that wordlists were collected
nor publishes any responses.  The report concludes that respondents called their
mother tongue Gondi rather than a distinct Pardhan language.

## Editorial decision and checklist disposition

- Installed lexical rows: **0**.
- Exclusions: **0 lexical cells**, because no lexical cells are published.
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

This closes ESR 2008-018 as an inspected no-form survey summary; it is not an
unfinished OCR candidate.
