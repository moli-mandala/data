# Source-ingestion checklist status — JLSR 2021-034 Dhurwa

## Source and topology

- [x] Official archive record and canonical PDF URL identified.
- [x] Exact canonical-URL Wayback capture acquired and pinned with checksum, bytes, and page count.
- [x] Rights/fair-use notice, authorship, edition, and data-collection date recorded.
- [x] All available representations inspected for topology; lexical forms remain visually hand-keyed.
- [x] Appendix B boundary and item/page topology established: physical pp. 17–21, 200 × 5 = 1,000 cells.
- [x] Four printed headers identified without inferring the blank fifth header.

## Exhaustive manual review and source-local staging

- [x] Physical pp. 17–21 / printed pp. 12–16 / items 1–200 exhaustively reviewed in five disjoint chunks.
- [x] All 1,000 cells have explicit page/item/column coordinates and manual declarations.
- [x] Five source-explicit blanks, thirteen multiple-response expansions, and all fifth-column responses accounted for.
- [x] No OCR/PDF lexical readings used; no OCR fields in the ledger.
- [x] Ambiguous/illegible/unresolved transcription cells: zero.
- [x] Source-local importer rejects OCR-bearing or incomplete ledgers.
- [x] Source-local exhaustive audit, 809-row target staging, list registry, complete-source profile, reproducible manifest, documentation, and focused tests present.

## Deferred shared integration gates

- [ ] Resolve the fifth response column only if authoritative evidence is found; otherwise retain it audit-only.
- [x] Reinventory and validate the conversion profile against every complete-source staged form.
- [ ] Apply shared BibTeX, language/dialect, and profile-routing changes proposed in `INTEGRATION.md`.
- [ ] Run the consolidated build, compiled-CLDF checks, full pytest, graph validation, and browser QA.

The source-local extraction and audit are exhaustive. Full ingestion remains incomplete until the deferred shared integration/build/QA gates pass. Shared registries and generated outputs remain untouched.
