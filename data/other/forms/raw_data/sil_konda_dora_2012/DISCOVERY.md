# ESR 2012-016 Konda Dora: acquisition and extraction topology

Discovery/acquisition record created 2026-08-28 with
`data/SOURCE_INGESTION_CHECKLIST.md` active.  Applicable addenda are “Survey
wordlists or comparative tables” and “OCR-heavy source.”  This is **not** a
completed ingestion: Appendix 9.5 is queued for exhaustive manual transcription
and review.

## Pinned source

- Blair, Frank, and Jacob George, with researchers Susan George and Stephen
  Watters. 2012 [created 1987]. *Multilingualism Among the Konda Dora*. SIL
  Electronic Survey Reports 2012-016.
- Official SIL archive record: <https://www.sil.org/resources/archives/49120>.
- Official publisher PDF:
  <https://www.sil.org/system/files/reapdata/38/76/91/38769117428458388974018399323322688545/silesr2012_016.pdf>.
- Acquired workspace file: `tmp/pdfs/konda_dora/silesr2012_016.pdf`.
- SHA-256:
  `6e0a3e5522a45752938f8279753d07b4e29d7b76ca73e88f71c4e283dfd0f533`.
- File size: 31,978,201 bytes; 106 physical PDF pages; archive extent 101
  numbered report pages.
- Rights: official SIL publication; the PDF is not redistributed in this
  source package.

## Appendix 9.5 topology

Appendix 9.5, “Word Lists,” occupies physical PDF pages 88–106 (printed pages
83–101).  Physical page 88 gives methodology and source metadata.  The lexical
tables contain four lists:

| Role | List | Location | Physical pages | Printed pages |
|---|---|---|---:|---:|
| target | Koraput Konda (Kubi) | Pansawalsa, Potangi, Koraput District, Orissa | 89–97 | 84–92 |
| target | Visakh Konda (Kubi) | Lakshmipuram, Paderu, Visakh District, Andhra Pradesh | 89–97 | 84–92 |
| comparison control | Telugu | Visakh District, Andhra Pradesh | 98–106 | 93–101 |
| comparison control | Adivasi Oriya (Kotia Oriya) | Visakh District, Andhra Pradesh | 98–106 | 93–101 |

The main text calls these 210-item lists.  The printed appendix contains items
1–210 plus supplemental `211 chest`, `212 liver`, a second source-numbered
`212 foot`, and `213 which?`: **214 printed prompt rows × 4 lists = 856
conceptual cells**, including **428 target cells** and **428 comparison cells**.
The duplicate item number 212 is visibly printed and must be represented with
distinct source-local prompt identities rather than silently renumbered.

The source prefixes Dravidian forms with lexical-similarity group numbers; these
are synchronic survey judgements, not historical-etymological claims.  Verbs may
print three slash-delimited forms (third-person past, imperative, infinitive), and
other cells may contain two elicited lexical alternatives.  Parentheses explicitly
mark a source-side uncertainty about phonetic or morphological status.

## Transcription requirement

The appendix is an image scan of typewritten phonetic forms.  The supplied OCR
layer is useful for locating pages and rows but visibly confuses glyphs, combining
marks, group digits, and punctuation.  Every installed form must be manually
transcribed from the rendered page and visually verified cell by cell.  The
eventual package must account for all 856 conceptual cells, split only the
source-defined slash alternatives/forms with explicit provenance, preserve blanks
and parentheses, list every ambiguous or illegible reading by physical/printed
page, prompt, and list, exclude the two control lists from installed target forms,
and keep OCR strictly as a comparison scaffold.

The exact source is acquired and the extraction geometry is resolved; manual
transcription, importer/profile work, shared metadata, audit, full build, and
browser QA remain pending.
