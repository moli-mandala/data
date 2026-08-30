# Source-ingestion checklist review

Applicable addenda: **survey wordlists/comparative tables** and **OCR-heavy
sources** (the OCR gate is conservative even though Appendix B.5 is typeset and
has a Unicode text layer).

- [x] Canonical SIL record and direct PDF URL identified.
- [x] PDF pinned by URL, archived retrieval, byte size, page count, and SHA-256.
- [x] PDF metadata, embedded text, rendered page images, and OCR representation inspected.
- [x] Appendix B.4 site metadata and all Appendix B.5 wordlist pages inspected.
- [x] All 2,900 printed response lines and code brackets visually checked.
- [x] All 5,250 conceptual cells accounted for, including controls and blanks.
- [x] No OCR-derived reading installed; OCR is comparison evidence only.
- [x] Exact Unicode diacritics retained with NFC normalization only.
- [x] Slash alternative expanded; identical repeats within one site/item deduplicated.
- [x] Similarity groups retained as non-etymological notes only.
- [x] Two unidentified source codes preserved and excluded without guessing.
- [x] Source-local importer, audit, manifest, page review, unresolved ledger, and profile added.
- [x] Focused source tests pass.
- [x] Shared bibliography and Kodaku language/eighteen dialect rows integrated.
- [x] Shared profile routing, census status, audit registry, and metadata checks integrated.
- [ ] Full `make all`, full pytest, graph checks, and browser QA: deliberately deferred.

Review summary: 210 prompts × 25 lists; 18 targets and seven controls; 5,250
cells audited; 5,183 attested cells; 67 blank/unlisted cells; 4,458 installed
target rows; zero ambiguous/illegible installed readings; two unresolved
unidentified site-code assignments (`u`, `n`).
