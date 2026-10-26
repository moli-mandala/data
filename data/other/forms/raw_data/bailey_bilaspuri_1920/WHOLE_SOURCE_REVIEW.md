# Whole Bilaspuri recovery — review handoff

The source-ingestion checklist is active, with dictionary/glossary, survey/table
and OCR-heavy addenda. The old 288-row glossary remains untouched. It does not
establish whole-source completion.

The pinned Bailey scan's printed pages 231–244 have now all been visually read
once. `whole-grammar-first-reading-20260927.tsv` records 257 groups (including
abbreviated paradigms, complete examples and three sentence-commentary groups).
`whole-sentences-first-reading-20260927.tsv` records all 22 printed sentences.
`whole-context-controls-first-reading-20260927.tsv` accounts for 30 shared
geographic, bibliographic, comparative and neighboring-lect groups. These are
provisional readings, not approved canonical forms. Each source unit needs a
second image reading before a proposal is frozen.

The existing embedded OCR is only a locating aid. Page 233's comparison table
and essentially all of page 236's grammatical tables were missing from it.
`inventory_whole_source.py` verifies the PDF hash and regenerates the 18-page
inventory for printed 231–248. The 245–248 glossary has its prior full visual
transcription and independent audit; preserve those 288 rows and their keys.

## Specific second-reading checks

- Distinguish breve, macron and stacked nasal marks by the original glyph;
  do not transfer marks between repeated occurrences merely by analogy.
- Check all future forms for the source's dotted nasal and underdotted r.
  Early first readings on p239 may lose these distinctions relative to p233.
- Check all stative endings against the source. The first-reading transcription
  currently uses provisional `-ādā`; the first vowel length/identity needs
  comparison across p233 and pp240–244.
- Resolve the 22 sentence rows' individual glyph and segmentation notes. Their
  translations are complete printed literal English, not token alignments.
- Keep hyphenated paradigm endings literal. Do not synthesize unattested full
  inflections or turn “etc.” into generated rows. Preserve the headword context
  and grammatical cell labels when mapping an ending.
- Sasi and Panjabi comparison columns and neighboring Bilaspuri/Handuri sections
  are explicitly attributed controls for this package, not standard Bilaspuri.
  Retain their readings and source context in the audit.
- Page 241 prints “fut. plur.” in a past/passive description. Preserve the
  printed label in evidence and record any editorial interpretation separately.

## Still required

Independent source review, a complete reviewed derivative and full proposal,
all 288 old rows/keys preserved, source-profile coverage, language/tag/reference
checks, focused importer tests, and a fresh independent output audit. The
whole-source package is not complete until these pass. Full compiled-data and
browser-database builds remain deferred under the user's current instruction.
No publication is authorized for this resumed ingestion stage.
