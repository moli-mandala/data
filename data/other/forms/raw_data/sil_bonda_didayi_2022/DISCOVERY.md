# JLSR 2022-004 Bonda/Didayi: acquisition and extraction topology

Discovery/acquisition record created 2026-08-28 with
`data/SOURCE_INGESTION_CHECKLIST.md` active.  The applicable addendum is “Survey
wordlists or comparative tables.” Appendix B has now been exhaustively extracted
and visually reviewed; see `README.md`, `CHECKLIST.md`, and the cell-level audit.

## Pinned source

- Mathew, Chacko, and Bradford Chamberlain, with researcher Faith Adimathara.
  2022 [created 1997]. *The Bonda and the Didayi from Malkangiri District,
  Orissa: A Preliminary Study*. Journal of Language Survey Reports 2022-004.
- Official SIL archive record: <https://www.sil.org/resources/archives/92608>.
- Official publisher PDF:
  <https://www.sil.org/system/files/reapdata/12/03/42/120342618681437246994104987693028749512/JLSR2022_004.pdf>.
- Acquired workspace file: `tmp/pdfs/bonda_didayi/JLSR2022_004.pdf`.
- SHA-256:
  `bb0548b4324224260b9618786dfd3aa40377138d0fbf4ae14c796df82f6190ce`.
- File size: 930,964 bytes; 64 physical PDF pages.
- Rights: the JLSR publication page supplies its scholarly-research/instruction
  fair-use policy.  The PDF is not redistributed in this source package.

## Appendix B topology

Appendix B, “Wordlists,” occupies physical PDF pages 21–50 (printed pages
16–45).  It prints 210 prompts for thirteen locality/language lists:

### Target lists

| Language | Variety | Village/town |
|---|---|---|
| Bonda | Upper | Dumripada |
| Bonda | Lower | Kadamguda |
| Bonda | Lower | Kendhuguda |
| Bonda | Lower | Rasabeda |
| Didayi | Lower | Chitrakonda |
| Didayi | Lower | Oringi |
| Didayi | Upper | Biapada |
| Didayi | Upper | Orapadar |
| Didayi | Upper | Kaluguda |

### Comparison lists

| Language/variety | Village/town |
|---|---|
| Gutob Gadaba | Tikrapada |
| Parenga Parja | Kinumun |
| Rona Desiya | Malenga |
| Oriya | Cuttack |

The complete matrix therefore has **13 × 210 = 2,730 conceptual cells**:
**1,890 target cells** and **840 comparison cells**. Prompts 11 `breast`, 23
`urine`, 24 `feces`, and 70 `millet` are printed `DISQUALIFIED`, accounting for
52 structural cells. The source also physically omits the Orapadar row at item
174; the complete audit accounts for that absence and every printed `no entry`,
dash, and lexical alternative.

The numbers prefixed to forms are the report's lexical-similarity groups and must
remain notes rather than historical-etymological claims.  Commas may separate
multiple source forms for one prompt, and predicate prompts distinguish paired
elicitation meanings such as imperative/past forms.  These distinctions must be
preserved rather than split mechanically by punctuation without prompt context.

## Transcription and review requirement

The appendix is born-digital, typeset Unicode IPA with a usable text layer; OCR is
neither required nor appropriate.  Nevertheless, every extracted cell and every
installed form must be compared visually with the rendered source page.  The
eventual package must account for all 2,730 conceptual cells, distinguish variants
from phrase-internal punctuation, preserve diacritics and source uncertainty,
report all blanks/no-entry/disqualified cells, list every ambiguous or clipped
reading by physical/printed page, prompt, and list, and exclude the four comparison
lists from installed target forms while retaining them in the audit.

The source-local extraction, visual review, importer, profile, audit, manifest,
and focused tests are complete. Shared metadata/routing, the full build, and
browser QA remain deferred to the consolidating agent as documented in
`INTEGRATION.md`.
