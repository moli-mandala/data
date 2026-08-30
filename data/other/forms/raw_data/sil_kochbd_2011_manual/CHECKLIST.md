# Checklist — ESR 2011-023 Koch Bangladesh manual recovery

Active addenda: survey wordlists/comparative tables; OCR-heavy/manual visual
review.

## Primary extraction and freeze

- [x] Pin the exact 91-page primary PDF and SHA-256.
- [x] Render and visually inspect physical pages 43–62 / printed pages 42–61.
- [x] Hand-transcribe all items 1–307 directly from page images.
- [x] Use high-resolution crops for small marks; do not let OCR, PDF text,
  legacy glyphs, or installed forms supply or verify a reading.
- [x] Expand printed site groups mechanically and freeze all 2,149 conceptual
  cells / 2,159 expanded rows with page, item, site, group, and evidence hashes.
- [x] Preserve distinct variants, repeated-group relations, source spacing,
  explicit blanks, globally not-used rows, and visible ambiguous bases.
- [x] Exclude all 226 ambiguous expanded rows without inferring any modifier;
  preserve the mixed item-241/site-`r` coordinate exactly.
- [x] Complete all 307 items with no pending page or item.

## Source-local post-freeze package

- [x] Snapshot and checksum the legacy installed forms, audit, and base profile.
- [x] Reconcile 2,159 frozen rows exhaustively against 2,208 expanded legacy
  rows, including duplicate aliases and 21 spurious not-used collisions.
- [x] Preserve matching legacy source `Entry_Key` identity; assign unused
  suffixes only to newly recovered occurrences.
- [x] Stage 1,017 resolved Koch target rows with 1,017 unique keys.
- [x] Keep 772 resolved control rows, 226 ambiguous rows, 25 blanks, and 119
  not-used rows audit-only.
- [x] Resolve all seven site/code identities from report metadata and preserve
  source spelling variants; leave unprinted exact coordinates blank.
- [x] Record exact reference metadata, author order, archive locator, source
  scope, and PDF hash.
- [x] Freeze the 44-codepoint inventory and exact source-local preservation
  profile; confirm zero unresolved profile mappings.
- [x] Record exclusion and non-etymology policies.
- [x] Generate deterministic hashes and pass source-local focused tests.

## Deferred shared gates

- [x] Replace the legacy shared source CSV with `staged_forms.csv`.
- [x] Update the shared dialect/site registry from `site_metadata.tsv`, removing
  invented coordinates rather than carrying them forward.
- [x] Replace the shared bibliographic entry from `reference_metadata.json`.
- [x] Route the exact base-profile snapshot explicitly in the shared build.
- [x] Update the shared ingestion checklist and integration manifest. Global
  checklist regeneration remains deferred until concurrent Garo work finishes.
- [ ] Run the consolidated CLDF build and opaque form-identity reconciliation.
- [ ] Run graph validation and the full test suite.
- [ ] Rebuild the browser database and perform source/language/form browser QA.
- [ ] Commit only after the shared gates are reviewed.

Shared source-specific installation is complete. No consolidated build, global
source-audit regeneration, browser database, common census document, Garo file,
or commit is changed by this integration stage.
