# Source-ingestion checklist audit

Applicable addenda: **Survey wordlists/comparative tables** and **OCR-heavy**.

## Source-local gates passed

- [x] Canonical archive/PDF metadata and exact SHA-256/bytes/pages pinned.
- [x] Exact 210 × 13 topology and target/control roles established.
- [x] All 2,730 cells hand-keyed/reviewed from renders; OCR was never lexical
  authority; exact manual declaration is present on every final cell.
- [x] Cell coordinates, source glosses, similarity labels, punctuation,
  alternatives, blanks, ambiguity, confidence, and review method preserved.
- [x] NFC, unique-key, full-coverage, coordinate, method, OCR-field, status,
  target/control, and source-pin guards implemented.
- [x] Exhaustive audit: 2,703 attested + 24 blank + 3 ambiguous + 0 illegible.
- [x] Target-only staging: 2,497 forms; 21 target blanks and 2 target ambiguous
  cells excluded; all 210 Toranmal control cells excluded.
- [x] Stable entry keys, exact citation locators, base-language routes, and
  language-qualified dialect tags emitted in source-local staging.
- [x] Source-local profile proposal and complete input-symbol inventory tested.
- [x] Bibliography/language/dialect/profile/routing and overlap plan recorded in
  `INTEGRATION.md`; focused tests pass.
- [x] Frozen hashes recorded and independently reverified before reconciliation:
  2,730-row topology, six-chunk manual bundle, 2,730-row staged audit, 2,497-row
  target output, unresolved ledger, list registry, and conversion profile.
- [x] All 234 retained source renders/crops have exact file hashes and a frozen
  tree hash in `render_hashes.tsv` / `preintegration_manifest.json`.
- [x] Exact post-freeze reconciliation accounts for 630 Noira and 840 Bareli
  republication cells without using later forms to supply or verify Dhule IPA.
- [x] All 2,497 immutable Entry_Keys and citation locators regenerate exactly;
  the incomplete-stage refusal and OCR-blind guards remain active.

## Shared source-specific integration passed

- [x] Applied canonical BibTeX, two base-language rows, eight new dialect rows,
  and reuse four existing dialect rows.
- [x] Preserved the frozen post-freeze reconciliation for four republished
  2018 Pauri lists and the related 2015 lists; later publication forms do not
  supply or verify the 2013 readings.
- [x] Installed the exact 2,497-row dated target CSV and exact audited
  source-specific profile; added immutable source-key routing.
- [x] Focused source, registry, reference and profile assertions pass.

## Deferred consolidated gates

- [ ] Run `make all`, compiled-CLDF/opaque-identity assertions, full pytest,
  generated-diff review, `errors.txt` review, and global audit regeneration.
- [ ] Browser refresh/QA is not requested and remains deferred.

The source-local and shared source-specific package is complete. The overall ingest is not complete
until the explicitly deferred consolidated gates pass.

The integration-ready contract is frozen in `preintegration_manifest.json`.
It installs all 2,497 independently audited target rows, excludes 21 target
blanks, two target ambiguities, and all 210 Toranmal control cells, reuses four
existing Bareli dialect IDs, adds eight source-locality dialect IDs, and maps
to two new (`Vasavi`, `Noiri`) plus two existing (`PauriBareli`,
`RathwiBareli`) base languages. The old bibliography-only key `bhildhule`
previously cited no forms and has been retired in favour of the canonical
source key `watters2013northerndhule`. The exact shared installation contract
is recorded in `shared_integration_manifest.json`.
