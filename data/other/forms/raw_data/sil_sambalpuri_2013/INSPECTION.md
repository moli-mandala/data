# ESR 2013-015 Sambalpuri: source inspection

## Decision

**No lexical forms are published in this report; there are no source records
to ingest.** Do not create an installed forms CSV, importer, sound profile,
language/dialect registry rows, or bibliography integration solely from this
PDF. The report describes wordlists collected during fieldwork but publishes
only site metadata and pairwise lexical-similarity percentages.

This inspection applies the survey-wordlist/comparative-table addendum of
`SOURCE_INGESTION_CHECKLIST.md`. The OCR-heavy addendum is inapplicable: the PDF
has a usable born-digital text layer, and there are no image-only lexical cells.

## Canonical source and inspected artifact

- Eldose K. Mathai and Juliana Kelsall. 2013. *Sambalpuri of Orissa,
  India: A brief sociolinguistic survey*. SIL Electronic Survey Reports
  2013-015, September 2013.
- Historical SIL series locator:
  <http://www.sil.org/silesr/abstract.asp?ref=2013-015>.
- Inspected file: workspace `tmp/pdfs/sambalpuri/source.pdf`.
- PDF SHA-256:
  `45bd0559ded2aedfd4c969b962c3b0f4f433125a98a31ad883576f25b509a080`.
- File size: 196,207 bytes; 10 PDF pages; metadata creation date 7 October
  2013; title and authors agree with the title page.
- Copyright page (PDF p. 2) states © 2013 Eldose K. Mathai, Juliana Kelsall,
  and SIL International, “All rights reserved.” No redistribution licence is
  asserted here.
- Glottolog reference `549905` classifies it as `comparative;socling` and cites
  the same SIL series locator. This classification does not imply that the
  underlying forms are printed.

## Exact evidence

The complete ten-page PDF was inspected through its text layer. The material
pages establishing scope were also rendered and visually checked.

1. **Contents, PDF p. 3 / printed p. 2.** The complete contents run from the
   abstract through sections 1-4 and then `References`. There is no appendix,
   wordlist-data section, or supplementary table of forms.
2. **Procedures, PDF p. 6 / printed p. 5, §2.2.** The authors state that a
   210-item basic-vocabulary wordlist was used, elicited in Hindi, and
   transcribed in IPA. They describe grouping phonetically similar words and
   reducing the comparisons to lexical-similarity percentages. Footnote 1 says
   that fewer than 210 items were sometimes compared when a response could not
   be obtained or was inappropriate. The prompt list, responses, IPA
   transcriptions, actual per-pair denominators, and similarity-group decisions
   are not printed.
3. **Site selection, PDF p. 6 / printed p. 5, Table 1.** The report names four
   Sambalpuri collection sites but supplies no forms:

   | Village | Tehsil | District |
   |---|---|---|
   | Kabarapally | Sambalpur | Sambalpur |
   | Bichhuvan | Bhatli | Bargarh |
   | Jharsuguda | Jharsuguda | Jharsuguda |
   | Balangir | Balangir | Balangir |

4. **Results, PDF p. 7 / printed p. 6, Table 2.** This is a triangular matrix
   containing only ten pairwise percentages for the four Sambalpuri lists and
   one Standard Oriya comparison list:

   | Pair | Percentage |
   |---|---:|
   | Sambalpur - Bargarh | 95 |
   | Sambalpur - Jharsuguda | 90 |
   | Bargarh - Jharsuguda | 92 |
   | Sambalpur - Balangir | 91 |
   | Bargarh - Balangir | 91 |
   | Jharsuguda - Balangir | 90 |
   | Sambalpur - Standard Oriya | 76 |
   | Bargarh - Standard Oriya | 75 |
   | Jharsuguda - Standard Oriya | 75 |
   | Balangir - Standard Oriya | 76 |

5. **Summary, PDF p. 9 / printed p. 8, §4.1.** The authors repeat only the
   ranges: 90-95 percent among the four Sambalpuri lists and 75-76 percent
   against Standard Oriya.
6. **Report terminus, PDF p. 10 / printed p. 9.** The final page is
   `References`; no lexical appendix follows it.

## Accounting and exclusions

- Fieldwork reportedly produced four Sambalpuri 210-item wordlists and one
  Standard Oriya comparison list, a nominal design of 1,050 site × prompt
  slots. The source explicitly warns that actual compared counts can be lower,
  but does not publish those denominators.
- Published lexical-form cells: **0**.
- Published form-level blanks, ambiguous readings, clipped cells, or
  illegible cells: **0**, because no form table is printed.
- Published comparative results: **10 percentage cells**, fully enumerated
  above. They are aggregate analysis, not lexical attestations and not evidence
  from which forms or cognacy edges can be reconstructed.
- Target sites: Kabarapally, Bichhuvan, Jharsuguda, and Balangir. Standard
  Oriya is a comparison control. These mappings are documentary only and do
  not justify registry changes without published lexical rows.
- Manual IPA cells inspected/transcribed: **0**. There are no printed IPA
  response cells; no OCR or guessed reconstruction was attempted.
- A separate wordlist publication or archival data appendix would be a new
  candidate source. It should be pinned and ingested independently if located.

## Proposed survey-census update

Replace the ESR 2013-015 row in
`data/other/forms/raw_data/sil_survey_sources.md` with:

```markdown
| ESR 2013-015 Sambalpuri | reports four 210-item Sambalpuri collections plus one Standard Oriya comparison, but publishes only site metadata and ten lexical-similarity percentages; no prompt list, IPA responses, denominators, or lexical appendix | **inspected; no lexical rows to ingest** (complete 10-page official PDF; §2.2 and Tables 1-2 on printed pp. 5-6; report ends with References on printed p. 9) |
```

The shared census file is intentionally not edited in this source-local task;
the coordinating agent can apply the replacement once parallel inspections
are consolidated.
