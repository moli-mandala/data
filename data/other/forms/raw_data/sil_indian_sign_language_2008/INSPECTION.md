# ESR 2008-006 Indian Sign Language: source inspection

Checked 2026-08-29 under the survey-wordlist/comparative-table and OCR-heavy-source
gates in `data/SOURCE_INGESTION_CHECKLIST.md`. The complete report contains lexical
analysis, but it does not publish any lexical sign form. The extraction, sound-profile,
language/dialect, graph, importer, and browser-entry gates are therefore genuinely
inapplicable rather than unfinished transcription work.

## Pinned source

- Jane E. Johnson and Russell J. Johnson. 2008. *Assessment of Regional Language
  Varieties in Indian Sign Language*. SIL Electronic Survey Report 2008-006,
  April 2008.
- Official SIL archive record: <https://www.sil.org/resources/archives/9033>.
- Current publisher PDF locator:
  <https://www.sil.org/system/files/reapdata/95/53/47/95534743464081586996546262623353152613/silesr2008_006.pdf>.
- The current endpoint returned Cloudflare HTTP 403. The exact older publisher PDF
  was acquired from the Internet Archive capture at timestamp `20170516235047` of
  `http://www-01.sil.org:80/silesr/2008/silesr2008-006.pdf`.
- SHA-256:
  `00ae89e7fcfee81dd46c6895f338dd989ed9dd7ebe99cf8604c946eaf18a426f`.
- File size: 1,024,681 bytes. `pdfinfo` reports PDF 1.6, 121 US-Letter pages,
  and no encryption. The copyrighted PDF is not redistributed in this package.

## Complete-publication and lexical-locus audit

The text layer was extracted from all 121 pages and all 121 pages were rendered at
180 dpi. The contents and the report's lexical methods/results identify two places
where item-level lexical material is published; every page of both loci was visually
checked against the renders:

1. Table 2 on physical pages 11-12 prints the final 245-item elicitation instrument.
   It contains numbered English glosses only. Superscripts identify picture cues and
   items classified as iconic; they are not sign transcriptions.
2. Appendix G on physical pages 58-63 repeats the English glosses and prints ten
   pairwise similarity judgments for Delhi, Kolkata, Chennai, Mumbai, and Hyderabad,
   plus a per-item mean. Values such as `0`, `0.25`, `0.5`, `0.75`, and `1` encode
   analyst judgments; dashes are missing comparisons. They do not encode signs.

The methods say that elicitation was videotaped and representative city wordlists
were visually compared for movement, hand shape, location, orientation, and
non-manual features. The publication does **not** include or link those recordings,
individual sign stills, diagrams keyed to items, SignWriting, HamNoSys, Stokoe
notation, movement-hold feature strings, or any other recoverable form
representation. Figure 2 on physical page 12 is a generic silhouette of the
elicitation setup, not a lexical sign.

## Editorial decision and checklist disposition

- Installed lexical rows: **0**.
- Excluded analysis material: **245 prompt-only rows**, **2,450 pairwise judgment
  slots**, and **245 mean slots**. None is a lexical form.
- Unresolved published readings: **0**. The underlying recorded signs are absent
  from the publication, not illegible cells that could be manually transcribed.
- Transcription decision: do not coerce English prompts or similarity numbers into
  Jambu's spoken-form field, and do not reconstruct signs from secondary sources.
- Metadata/reference/dialect/profile decision: retain the exact source identity and
  checksum in this inspection and the census, but create no pseudo-variety, lexical
  reference row, or sound profile for a source contributing no forms.
- Validation: 121-page PDF integrity check, complete text extraction/rendering, and
  direct visual inspection of physical pages 11-12 and 58-63. A focused importer,
  graph node, CLDF row, and representative browser entry are inapplicable. The
  repository-wide build remains deferred until the active lexical packages finish.

This closes ESR 2008-006 as an inspected lexical-analysis report with no published
lexical forms, not as an OCR or representation backlog item.
