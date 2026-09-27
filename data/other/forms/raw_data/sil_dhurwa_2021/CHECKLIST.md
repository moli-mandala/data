# Source-ingestion checklist status — JLSR 2021-034 Dhurwa

## Shared integration update — 14 September 2026

The frozen staging is now adapted into `data/other/forms/20260914-sil-dhurwa.csv`: 809 rows under canonical language `Parji`, with shared dialects, bibliography, formatted references and explicit profile routing. Read-only shared-CLDF verification on 21 September 2026 confirms every installed key, persistent ID, transcription layer, citation, dialect tag and source-defined variant edge. No database was rebuilt. A new full build, repository-wide suite and global retrospective audit remain unclaimed; see the current shared review for scope and evidence. See the [shared review](../../../../../source_checklists/20260914-manual-surveys-review.md) for counts, corrections, validation and remaining gates.

The source manifests and staged files remain frozen as extraction evidence. The
following text records that earlier source-local stage; its pending shared gates
are superseded only to the extent documented in the shared review.

## Frozen source-local documentation

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
