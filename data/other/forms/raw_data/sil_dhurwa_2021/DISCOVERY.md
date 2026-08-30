# Source discovery and topology — JLSR 2021-034 Dhurwa

## Canonical source and acquisition

- Official SIL archive record: <https://www.sil.org/resources/archives/89899>.
- Canonical SIL attachment: <https://www.sil.org/system/files/reapdata/13/49/80/134980156912388403723813021095818766285/JLSR2021_034.pdf>.
- The live attachment returned a 5,770-byte Cloudflare HTML challenge to command-line retrieval on 2026-08-28.
- The Wayback CDX index records an exact `200 application/pdf` capture at timestamp `20240706204652`, digest `SHVOK7OX2RSSZBLFRRPZOI6QXO7KBJEM`.
- Pinned copy: `tmp/pdfs/dhurwa_2021/JLSR2021_034.pdf`, acquired from the `id_` replay of that exact canonical URL.
- Pinned file: 649,838 bytes, 24 physical pages, SHA-256 `92965cbf77b88685a3f46e59053ce027b4a600037c8043d518c477ac7eac341e`.

The PDF metadata title and authors match the official record. SIL lists the attachment as 634.61 KB and the work as 24 pages. The report's printed fair-use notice permits scholarly research and educational copies but prohibits republication or commercial use without written permission.

## Corrected Appendix B topology

Direct visual inspection corrects the earlier census estimate of 210 prompts and physical pp. 17–22:

| Physical PDF page | Printed page | Items | Cells |
|---|---:|---:|---:|
| 17 | 12 | 1–41 | 205 |
| 18 | 13 | 42–82 | 205 |
| 19 | 14 | 83–124 | 210 |
| 20 | 15 | 125–167 | 215 |
| 21 | 16 | 168–200 | 165 |

Appendix B therefore contains exactly **200 prompts × 5 response columns = 1,000 conceptual cells**. Physical p. 22 / printed p. 17 begins Appendix C, not the wordlist. This agrees with the report's own methodology, which calls it a 200-item wordlist.

## List identity

The first four response headers are visibly `Tiriya`, `Nethanar`, `Dharba`, and `Kukanar`; these are also the report's four named Dhurwa dialects. A fifth response column is printed throughout Appendix B, but its header cell is blank on physical pp. 17–21. The surrounding prose describes four Dhurwa dialects and does not authoritatively identify this fifth list.

The package therefore assigns only the neutral source-local code `U5` and display label `Unlabeled fifth printed column`. It is fully transcribed and audited but excluded from staging until independent source evidence establishes its identity. No language or locality is inferred from the forms.

## Extraction policy

The appendix is born-digital, but every accepted transcription is hand-keyed from rendered page images under the user's stricter manual-review rule. PDF text extraction was used only for report prose and topology outside the lexical appendix; it was not used to seed, supply, or verify any lexical reading. The exhaustive review covers all 1,000 cells on physical pp. 17–21 / printed pp. 12–16, items 1–200: 995 attested, five explicit blanks, 1,008 expanded responses, and no ambiguous or illegible transcription cells.
