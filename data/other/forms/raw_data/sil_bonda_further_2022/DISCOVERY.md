# JLSR 2022-005 Bonda: acquisition and extraction topology

Discovery record created and corrected 2026-08-28 with
`data/SOURCE_INGESTION_CHECKLIST.md` active. The applicable addendum is
“Survey wordlists or comparative tables,” together with the OCR/manual-review
addendum and the user's stricter visual-review rule. This is a guarded,
source-local partial ingestion; the shared census tracker records checkpoint
progress, but shared registries, staging, and build outputs remain unchanged.

## Canonical source

- Chacko Mathew. 2022 [survey conducted and report created 2002]. *The Bonda:
  Further Sociolinguistic Survey*. Journal of Language Survey Reports
  2022-005.
- Official SIL archive record: <https://www.sil.org/resources/archives/92609>.
- Official publisher PDF:
  <https://www.sil.org/system/files/reapdata/14/59/18/145918519553516788025406164230420247696/JLSR2022_005.pdf>.
- The archive describes the attachment as 1.19 MB and the report as 74 pages.
- Pinned official bytes: `tmp/pdfs/bonda_further_2022/JLSR2022_005.pdf`,
  downloaded from the canonical publisher attachment in the official browser
  PDF viewer after the command-line endpoint returned a Cloudflare challenge.
- Exact file: 1,247,227 bytes, 74 physical pages, PDF 1.6, SHA-256
  `9c4457aa6e73906b34e8c69e790e9d205a9b95cfc2a94ccae054bcb1537dfcfa`.

## Appendix A topology

Appendix A, “Wordlists,” occupies physical PDF pages 15–47 (printed pages
10–42). The first response page begins with items 1–5; physical p. 46 ends at
item 208 and physical p. 47 contains items 209–210. It prints 210 numbered
prompts against eleven lists, for **2,310 conceptual cells**.

The earlier 15-list census was a layout error. Three comparison labels wrap
onto a second source line: `Tikrapada` + `Gadaba`, `Kinumun` + `Parenga Parja`,
and `Malenga Rona` + `Desiya` each name one list and have only one response.
They are not six independent lists.

The three Upper Bonda lists staged as the target scope are:

- Podeiguda (`UB`),
- Bondapada (`UB`), and
- Dumripada (`UB`).

They contribute **630 target cells**. The remaining eight lists are
comparison/control material: Kadamguda Lower Bonda, Kendhuguda Lower Bonda,
Rasabeda Lower Bonda, Tikrapada Gadaba, Biapada Upper Didayi, Kinumun Parenga
Parja, Malenga Rona Desiya, and Oriya (Cuttack). These **1,680 comparison
cells** remain audit-only and must not duplicate or overwrite the JLSR
2022-004 installation.

Section 2.1 supplies the authoritative cross-source relationship. Podeiguda
and Bondapada were newly collected for the 2002 survey. Dumripada is the 1997
Upper Bonda wordlist checked in the later survey and explicitly said to
replace the old Dumripada wordlist. The eight comparison lists are the same
wordlists shown in the previous report. Similarity group decisions were
re-evaluated. Consequently the package stages the three current Upper Bonda
lists source-locally, but integration must reconcile Dumripada as a checked
replacement and keep the eight republished lists audit-only.

At least item 11 is printed `DISQUALIFIED` across the table. Every such prompt,
true blank, and source qualification must be counted cell by cell rather than
treated as a form.

## Representation and review requirement

The appendix is born-digital and has an extractable text layer, but the text
layer is not transcription evidence. Under the user's stricter rule, every
lexical response is hand-keyed from the rendered page and visually verified;
PDF extraction is used only to locate pages, rows, and report prose. It may not
supply, normalize, or verify an accepted reading. All blanks, disqualified
prompts, ambiguities, clipped cells, and illegible readings are recorded with
physical/printed page, item, and list coordinates.

The appendix bounds, all eleven list labels, first page, final page, and
representative middle pages were visually inspected. The source-local package
now contains OCR-blind manual checkpoints for physical pp. 15–40 / printed
pp. 10–35 / items 1–165. Exhaustive remaining-page review, complete Dumripada replacement
reconciliation, shared metadata/routing, builds, and browser QA remain pending.
