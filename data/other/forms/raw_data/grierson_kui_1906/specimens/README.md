# Complete specimen first reading

This directory accounts for all four printed specimens (40–43), pp.464–471, as a **source-local first reading**, not an installed or independently accepted ingestion.

`p464-first-reading.tsv` through `p471-first-reading.tsv` retain every physical interlinear form and its aligned printed English gloss. Page469 contains the free translation of specimen42, not another interlinear block. Page465 also contains the complete free translation of specimen40. The surrounding linguistic commentary on465/469/470 is separately owned by the grammar/prose inventory, not omitted from the whole chapter.

`prepare_first_reading.py` emits 890 physical atoms (884 interlinear and six title atoms), 108 physical lines, 77 provisional source-punctuation expression spans, and two explicit physical-line continuations. Counts by specimen are 124/328/104/334. The script asserts exact form/gloss cardinality, unique physical keys, and exactly one contiguous expression-span assignment per non-title atom. Expression spans preserve the full literal interlinear text; they are not invented translations or syntactic analyses. They require review before export, including the two printed line-break joins. No word meanings are manufactured from the free translations.

All eight higher-resolution IA original images are URL/hash-pinned in `alternate-source-manifest.json`. The printed464 folio and successive original specimen headings/content were visually verified; the alternate leaf mapping is printed page +21. Original DjVu witnesses remain available for comparison. The unreadable raw DjVu text for464 is retained as unaccepted extraction evidence.

The earlier low-resolution464 draft is preserved in `p464-low-resolution-draft.tsv`. Its macron/underdot errors were corrected against the clearer original, with the initial correction batch recorded in `p464-alternate-self-review.json`; the final TSV is authoritative for this first-reading freeze. These self-corrections do not replace independent review.

The approved source-lect mapping assigns specimen40 to Gumsur/Bod,41–42 to Khondmals, and43 to the historical Kui/Kalahandi route. The modern-language-identification conflict for Kalahandi is explicit in every corresponding record; coordinates are not inferred. Specimen40 retains the printed Smith1876 attribution and Grierson’s translation caveat.

Thirty-two atoms carry typed first-reading uncertainty: the Smith nasal-upper-mark class (ñ provisional; straight/slanted/curved source marks are not harmonized), p470 `ēohē` versus `ēchē`, p471 damaged `sibki-mānēñju`, and p471 `mũtēñju` versus a possible breve. The parent independently inspected the giving-is crop and supported provisional **sibki**, while explicitly leaving the damaged medial cluster unresolved. General volume notation does not settle the Smith nasal-letter mark, as recorded separately in `../volume-notation-review-20260927.json`. Literal stacked `sūḍiē̃` preserves the plain i and the macron-plus-tilde on e; no IPA inference is made.

All canonical CSV/YAML/registry/profile files remain untouched. Independent complete reading, grammatical metadata, expression alignment, integrated source metadata/profile/parser/identity checks, and source-stage installation are pending. Full builds and browser QA remain outside this authorized source-only stage.
