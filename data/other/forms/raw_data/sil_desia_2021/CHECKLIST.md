# Source-ingestion checklist review

Applicable addendum: **survey wordlists/comparative tables**. The OCR-heavy
addendum is inapplicable because Appendix B.5 is fully typeset with an embedded
Unicode text layer and contains no handwritten or image-only IPA. Its central
editorial rule is nevertheless honored: extracted text is a scaffold only,
and every installed response was visually compared with the rendered source.

- [x] Canonical SIL archive/publication records and direct PDF URL identified.
- [x] PDF pinned by canonical URL, archived retrieval, byte size, page count, and SHA-256.
- [x] PDF metadata, embedded Unicode text, and all rendered appendix pages inspected.
- [x] Appendix B.4 site metadata, Table 1, and all Appendix B.5 wordlist pages inspected.
- [x] All 4,696 printed response lines visually checked cell by cell.
- [x] All 3,990 conceptual target cells accounted for, including 38 explicit blanks.
- [x] No OCR-derived or unreviewed text-layer reading installed.
- [x] Exact Unicode diacritics retained with NFC normalization only.
- [x] All 542 positional text-layer combining-mark misattachments corrected from rendered glyphs/content-stream order and retained in a row-level comparison ledger.
- [x] Identical duplicate readings within a site/item merged without loss of group metadata.
- [x] Similarity groups retained as non-etymological notes only.
- [x] Blank similarity-group field preserved without guessing.
- [x] Metadata discrepancies preserved and documented without silent harmonization.
- [x] Source-local importer, audit, manifest, page review, unresolved ledger, and profile added.
- [x] Full visual review exceeds the seeded 20-record audit gate (4,696/4,696 checked; zero material errors); first/last pages, column and page transitions, multi-form cells, blanks, and rare symbols inspected deliberately.
- [x] Focused source tests pass.
- [ ] Shared bibliography/dialect rows and profile routing: deferred to coordinating task; exact proposal in `INTEGRATION.md`.
- [ ] Full `make all`, full pytest, graph checks, and browser QA: deliberately deferred to coordinating task.

Review summary: 210 prompts × 19 target lists; 3,990 conceptual cells; 4,658
attested and 38 explicit `no entry`; 4,655 installed forms; zero controls;
zero clipped, illegible, ambiguous, or unresolved readings. One blank source
group and three cross-group exact duplicates are preserved explicitly.
