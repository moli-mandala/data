# ESR 2012-015 Kurumba: acquisition and extraction topology

Discovery/acquisition record created 2026-08-28 with
`data/SOURCE_INGESTION_CHECKLIST.md` active.  Applicable addenda are “Survey
wordlists or comparative tables” and “OCR-heavy source.”  This is **not** a
completed ingestion: the appendix is queued for exhaustive manual transcription
and review.

## Pinned source

- Blair, Frank, with R. Prabhu, K. Chitrarasu, B. B. Rajah, J. Rajaiah,
  Carolyn Rensch, and Cal Rensch. 2012 [fieldwork 1984–1985; report written
  1986/87]. *A Sociolinguistic Profile of Kurumba Dialects*. SIL Electronic
  Survey Reports 2012-015.
- Official SIL archive record: <https://www.sil.org/resources/archives/50805>.
- Official publisher PDF:
  <https://www.sil.org/system/files/reapdata/12/35/31/123531168727431837602494229591948707669/silesr2012_015.pdf>.
- Acquired workspace file: `tmp/pdfs/kurumba_2012/silesr2012_015.pdf`.
- SHA-256:
  `250dc3d83661227caa66bf16e390e51c2dcb7186fa435252541ed13bbfcd9137`.
- File size: 128,831,439 bytes; 436 physical PDF pages; archive extent
  `ii, 431 pages`.
- Rights: official SIL publication; the PDF is not redistributed in this
  source package.

## Appendix C topology

Appendix C, “Word Lists,” occupies physical PDF pages 214–436 (printed pages
209–431).  Physical pages 214–216 give the list metadata.  Physical pages
217–436 (printed pages 212–431) contain 550 numbered prompts in twenty-two
25-item blocks.  The first eighteen lists are printed as nine two-list column
pairs; the final Betta Kurumba list occupies a single column.  This yields
**19 × 550 = 10,450 conceptual response cells** before blanks and variants are
audited.

| Physical pages | Printed pages | Left list | Right list |
|---:|---:|---|---|
| 217–238 | 212–233 | Standard Tamil, Madras | Standard Kannada, Bangalore |
| 239–260 | 234–255 | Belavarthy Kurumba | Pudukkottai Kurumba |
| 261–282 | 256–277 | Kotagiri Alu Kurumba | Badaga, Arvenu/Kotagiri |
| 283–304 | 278–299 | Kolar Kurubas | Chitradurga Kurubas |
| 305–326 | 300–321 | Buringi Kurumba | Madapalli Kurumba |
| 327–348 | 322–343 | Kurumbatheru Kannada | Thangiyadikuppam Kurumba |
| 349–370 | 344–365 | Beerajjanur Kurumba | Karmadai Kurumba |
| 371–392 | 366–387 | Karmadai Vakkaliga | Kurumbapalayam Kurumba |
| 393–414 | 388–409 | Kalangal Kurumba | Masinagudi Jennu Kurumba |
| 415–436 | 410–431 | Maddur Colony Betta Kurumba | — |

The inventory therefore contains fifteen Kurumba/Kuruba-labelled survey lists
and four obvious Tamil, Kannada, Badaga, and Vakkaliga comparison lists.  Final
target/control classification, especially for the Kolar and Chitradurga Kuruba
comparanda, must follow the report's analysis and the checklist's standing policy
rather than this preliminary label count.

## Transcription requirement

The PDF is a scan.  Its supplied OCR layer is a useful locator, but it visibly
confuses phonetic glyphs (`MAn`, `tfekke`, `BAN6`, and similar output) and cannot
be used as lexical data.  Every installed form must be manually transcribed from
the rendered page and visually verified cell by cell.  The eventual package must
account for all 10,450 conceptual cells, preserve blanks and uncertainty, report
every ambiguous or illegible cell with physical/printed page, item, and list, and
keep OCR strictly as a comparison scaffold.

The exact source is acquired and the extraction geometry is resolved; manual
transcription, importer/profile work, shared metadata, audit, full build, and
browser QA remain pending.
