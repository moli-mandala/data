# Source-ingestion checklist status

Active addenda: **Survey wordlists/comparative tables** and **OCR-heavy source**.

- [x] Canonical SIL source, exact PDF, checksum, size, page count, rights state,
  included appendix, and exclusions recorded.
- [x] All available representations inspected; corrupt OCR classified as
  locator-only and separated from manual authority.
- [x] Nineteen-list / 550-prompt / 220-page topology fixed before extraction.
- [x] Stable list, page, prompt, and 10,450 cell identifiers generated.
- [x] Target/control and canonical base-language mappings recorded source-locally.
- [x] Reproducible renderer and non-overwriting OCR scaffold generator provided.
- [x] Importer refuses pending or OCR-only forms and stages only manually
  reviewed attestations.
- [x] Manually inspect and hand-transcribe all 10,450 cells from rendered pages.
- [x] Manually verify all 550 printed prompt glosses.
- [x] Complete page ledger; reconcile attested/blank/ambiguous/illegible totals.
- [x] Record every unresolved cell with page/item/list coordinates; never guess.
- [x] Generate source-local staged forms, full per-cell audit, and final manifest.
- [x] Derive and test a complete source-specific sound conversion profile.
- [x] Integrate proposed bibliography and nineteen dialect rows from
  `INTEGRATION.md` into shared registries.
- [x] Preserve the frozen 4,738-attestation snapshot and exhaustively filter its
  3,204 target attestations for shared installation; keep all 1,534 controls audit-only.
- [x] Run focused post-review and shared registry/reference/profile tests.
- [ ] Run the consolidated data build, opaque identity reconciliation, full test
  suite, graph validation, and representative browser inspection.

Current progress: physical pages 217–436 are complete, with all 10,450 cells and
all 550 unique prompt glosses manually read from 450/600/900-dpi renders, with
targeted 1200-dpi checks for dense cells.
Physical p.415 / printed p.410 is the first one-column Maddur Betta page and has
25 conceptual cells after the two-column Kalangal/Masinagudi block; physical
pp.415–436 are source-confirmed one-column pages with 25 cells each; p.436 is
the source-PDF endpoint at item 550. All 10,450 cells have 100% direct
rendered-image visual review; OCR/PDF
text supplied no accepted lexical reading.
There are 4,738 attested forms, 5,710 confirmed printed-dash blanks, one ambiguous
reading, one illegible reading, and two explicit unresolved cells: Pudukkottai
item 20 on physical p.239 and Kotagiri Alu item 25 on physical p.261. No forms
from those two unresolved cells are staged; 4,738 manually attested forms and a
10,450-row source-local audit are staged, and zero cells remain pending.
Shared installation contains only the 3,204 audited target-scope attestations.
All 1,534 control attestations, 5,710 printed dashes, and both unresolved target
cells remain explicit in the 10,450-row shared-integration audit. The consolidated
build, browser refresh, opaque identity reconciliation, and commit remain deferred.
