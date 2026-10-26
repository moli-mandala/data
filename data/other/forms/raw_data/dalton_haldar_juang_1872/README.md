# Haldar's Juanga vocabulary in Dalton 1872

## Whole-source recovery installed locally, 2026-09-27

The current source CSV contains **226 rows**, preserving all 196 earlier keys
and their order. Coverage includes the Juang chapter (printed152–158), all361
comparison/separate-list cells (235–242), and explicit grammatical markers.
All19 historical transcription holds are recovered; 153 printed blanks remain
excluded. The373-unit audit also accounts for four chapter attestations, five
documentary/control terms, and three grammatical markers. Thirteen retained
cells have corrected literal readings, ten rows retain typed uncertainty, and
nine source-local variant links preserve printed alternatives. No borrowing or
ancestry links were inferred. The existing canonical language is `ju`; chapter
prose does not inherit the Haldar-table dialect tag.

Original accents and marked n are preserved. The display profile retains
conservative existing consonant mappings while preserving accents; no separate
phonemic representation is invented. Source question marks and `(s.)` labels
are residual notes, with uncertainty tagged where applicable. `Tumba` has a
qualified contextual definition and no inferred botanical species.

Independent frozen-output review passed **0/20 material errors** across22 rows,
plus five exclusion controls and three explicit grammatical markers. Evidence,
the exact freeze, installation hashes, and scoped bibliography reconciliation
are in `whole-source-recovery/`. Thirteen focused tests pass from the documented
`data/` working directory, including actual installed-parser conversion.

Use `install_whole_source.py --install` to reproduce the approved source-stage
installation. The historical writer is guarded against overwriting it; its
`build()` remains available to compare the old inventory. The source CSV, YAML,
profile, audit and bibliography are installed. Full CLDF, compiled identity,
graph and reference checks, full suite, database refresh and browser QA remain
deferred. **This is not a claim of full ingestion completion.**

## Historical table-only installation (superseded)

Edward Tuite Dalton's *Descriptive Ethnology of Bengal* (1872), [public-domain scan](https://commons.wikimedia.org/wiki/File:Descriptive_Ethnology_of_Bengal.pdf), SHA256 `38dda30a06f7c56df34311328c36d3fdf8cb9835a5420aea4453fbb1b76008b1`, credits Babu Rakhal Das Haldar with compiling its comparative tables. The scan does not identify individual Juanga speakers or elicitation sites. The source spelling remains in `Original`, the sound profile is conservative rather than asserted IPA, and dialect coordinates are blank. Dalton's 1866 Appendix F is a probable earlier partial witness of overlapping Juang material, not a second independent citation. Samuells's explicitly collected 1856 list is materially different. No automatic etymology or borrowing edge is inferred from the comparisons or `(s.)` annotations.

This source-stage install covers **all Juanga target cells** in Haldar's comparative table (printed pp. 235–241, PDF pp. 250–256) and the separate Juang vocabulary (printed pp. 241–242, PDF pp. 256–257), ending before Group VIII. The comparison contains 227 target cells: 35, 42, 42, 37, 37, 25, and nine on successive pages. The separate list contains 134 entries: 39 on p. 241 and 95 on p. 242. `reviewed_inventory.tsv` retains the old p. 236 pilot, `comparative-extension.tsv` covers its other 185 target cells, and `separate-list-inventory.tsv` covers every separate-list head. Together they generate `audit.jsonl` with **361 cell decisions: 189 accepted, 19 typed transcription holds, and 153 printed blanks**. Accepted cells produce **196 installed form rows**, preserving the original 31 p. 236 pilot entry keys. Alternatives printed in one cell receive distinct keys. The eight adjacent comparison-language columns are controls, not Juanga attestations; p. 236 has an explicit 336-cell presence mask and control sample. The comparison's grammatical/sentence prompts are individually inventoried, including blank Juanga cells.

The original scan controls transcription. Rendered page images under `images/` and raw Tesseract sequencing aid under `ocr/` support review; OCR never overrides the printed glyph. A second reviewer checked all 30 accepted cells in the comparison extension, corrected p. 235 row 15 `ainya` to printed `ainyá`, and found zero remaining material errors. The separate-list audit sampled 20 entries across sections with zero errors; a subsequent enlarged-image pass reduced its holds from 18 to four. Its remaining holds are p. 241 One and p. 242 Bad, Give, and White. The p. 236 pilot had a 16-cell verbatim control sample and all 42 target cells visually reviewed. No uncertain form is silently completed from the similar separate list.

The historical `import_source.py` generated the earlier CSV from these three inventories. It is superseded by the guarded whole-source installer above. Representative app entries cannot be inspected until an authorized database refresh.
