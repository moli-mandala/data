# Source-ingestion checklist status — ESR 2015-012 Noira

## Source and topology

- [x] Canonical official archive record and pinned PDF identified.
- [x] PDF checksum, byte count, page count, title, authorship, and rights note recorded.
- [x] PDF representations inspected; Appendix A3 topology is exactly 210 × 17 = 3,570 cells.
- [x] Survey-wordlist and OCR-heavy/manual-review addenda activated.
- [x] Fourteen regional lists and three language controls inventoried in printed order.
- [x] Source metadata establishes eleven new target lists, three Dhule republications, and three controls.

## Manual extraction and audit

- [x] Eight disjoint item-range ledgers cover items 1–210 and physical pp. 33–78.
- [x] Every cell manually inspected at 400 dpi; selected difficult glyphs rechecked at 800/900 dpi.
- [x] OCR/PDF text did not seed, supply, or verify lexical readings.
- [x] Every ledger is OCR-blind, NFC-normalized, and carries the exact manual-review declaration.
- [x] All 44 source-explicit blanks, all variants, printed labels, page/column coordinates, and controls accounted for.
- [x] Ambiguous cells: 0; illegible/clipped cells: 0; unresolved readings: 0.
- [x] Header-only `unresolved_readings.tsv` makes the empty unresolved set explicit.
- [x] Exhaustive 3,570-row source-local audit generated.

## Identity, staging, and profile

- [x] Source-list registry records target/control/republication status and stable proposed dialect routing.
- [x] Kotli-Narayanpur and Kotli-Taradi are provisionally routed as source-supported Noiri survey-site dialects; labels/uncertainty retained, dialect Glottocode/coordinates blank, and no Kotali/Khandesi conflation.
- [x] Astambha, Mundalwad/Mutalwad, and Toranmal reconciled cell-for-cell against ESR 2013-004; all 630 Noira republication cells excluded from new-form staging.
- [x] Gujarati, Marati, and Hindi controls retained audit-only and excluded.
- [x] Deterministic source-local staging emits 2,714 new-target forms from 2,271 attested cells; 39 new-target blanks excluded.
- [x] Stable source-local entry keys, exact citations, source cognate labels, and dialect tags emitted.
- [x] Source-local conversion profile covers every grapheme in all staged forms.
- [x] Manifest records complete counts and artifact checksums.
- [x] Pre-integration manifest freezes PDF, 46-page fresh topology render audit, all manual ledger/generator and staged artifacts, profile inventory, exact exclusions, immutable keys, and the shared integration contract.
- [x] Focused tests cover guard rejection, schema/NFC/declaration, cell totals, variants, source coordinates, profile, republication policy, staging, and manifest agreement.

## Shared source-specific integration

- [x] Installed all 2,714 source-local staged rows byte-for-byte as `data/other/forms/20260829-sil-noira.csv`; all immutable keys and locators retained.
- [x] Installed the exact source-local profile as `conversion/sil-noira.txt` and routed only `varghesekumar2015noira` through it.
- [x] Applied exact BibTeX/reference metadata, one new Dungra Bhili parent, reused Noiri/Gujari/Korku/Nihali parents, and eleven source-specific dialect rows with blank coordinates.
- [x] Applied the provisional source-supported Kotli-site routing under canonical Noiri; dialect Glottocode/coordinates are blank and no historical Kotali/Khandesi conflation is made.
- [x] Kept all 630 Dhule-republication cells / 834 responses, 630 control cells / 837 responses, and 44 source blanks audit-only.
- [x] Source-specific registry/reference/profile/parser/freeze tests pass.

## Deferred global gates

- [ ] Run the consolidated build to resolve the 210 standard glosses to shared `Parameter_ID` values and inspect all generated diffs/errors.
- [ ] Run full pytest, graph validation, and browser QA with representative entries.
- [ ] Regenerate global source-audit/checklist outputs, refresh the browser database, and commit/ship.

The unchecked items are deferred because this lane was expressly prohibited
from running the consolidated build, global audit regeneration, browser refresh,
or commit. The source-local package and shared source-specific installation are
exhaustive and have zero unresolved lexical coordinates.
