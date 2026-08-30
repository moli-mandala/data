# Ho 2024 source-ingestion checklist

The mandatory source-ingestion checklist is active. Applicable addenda:
survey wordlists/comparative tables and OCR-heavy sources.

## Completed source-local gates

- Canonical report acquired and pinned by SHA-256, byte size, and page count.
- Appendix D.1 metadata, D.2 alphabet, and D.3 topology identified.
- All 27 source rows classified: 14 new targets, 3 republished controls, and
  10 non-Ho controls.
- Immutable base ledger contains exactly 210 x 27 = 5,670 unique item/site
  cells with physical page, printed page, column, gloss, and OCR-evidence fields.
- PDF pages 72-119 (items 1-144; 3,888 cells) were manually transcribed and
  visually checked. Counts: 3,595 attested, 292 blank, 1 ambiguous, 0 illegible.
- The admissible OCR-blind helper chunk adds PDF pages 120-127 (items 145-168;
  648 cells): 619 attested, 28 blank, 1 ambiguous, 0 illegible. Effective totals
  are therefore 4,536 reviewed, 4,214 attested, 320 blank, 2 ambiguous, and
  1,134 unreviewed.
- A second admissible OCR-blind chunk adds PDF pages 128-135 (items 169-192;
  648 cells): 595 attested, 52 blank, 1 ambiguous, 0 illegible. Effective totals
  are 5,184 reviewed, 4,809 attested, 372 blank, 3 ambiguous, and 486 unreviewed.
- The final admissible OCR-blind chunk adds PDF pages 136-141 (items 193-210;
  486 cells): 461 attested, 25 blank, 0 ambiguous, 0 illegible. Final totals are
  5,670 reviewed, 5,270 attested, 397 blank, 3 ambiguous, and 0 unreviewed.
- Dry build accounts for all 5,670 audit rows and stages 2,900 attested target
  response cells. It excludes 2,730 republished/non-Ho control cells, 38 target
  blanks, and 2 ambiguous target cells.
- The three current unresolved readings are fully localized in
  `unresolved_readings.tsv`.
- Source-local importer is OCR-blind for form staging and guarded against
  incomplete review, chunk overlap, unknown/duplicate keys, coordinate drift,
  invalid statuses, missing manual provenance, and non-NFC text.
- Target/control routing is explicit in `list_registry.tsv`; only the fourteen
  new 1989 Ho lists can stage.

## Completed extraction/audit gates

- A previously generated OCR-derived chunk was rejected and contributes no
  evidence. All accepted chunks are literal OCR-blind hand-keyed decision data.
- Final unresolved ledger, staged forms/audit, manifest counts, source-symbol
  inventory, and focused tests are complete.

## Deferred consolidated integration gates

- Shared bibliography, language/dialect registries, source routing, and sound
  profile integration are deferred to the root integration pass.
- Consolidated `make all`, full pytest, generated CLDF files, graph validation,
  and browser QA have not been run or edited here.
