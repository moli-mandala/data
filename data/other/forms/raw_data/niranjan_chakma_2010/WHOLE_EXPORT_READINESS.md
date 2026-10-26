# Whole-source preview readiness, 2026-09-27

**Source-stage installation is now complete.** Revision 2 passed fresh independent
review, guarded installation and repeat parity, actual canonical parsing, and all
40 applicable focused tests. See `whole-installation/source-stage-handoff.json`.
Full CLDF/database, compiled graph/identity/reference, full-suite and browser gates
remain deferred. The following records the preparation and historical freezes.

**Current revision:** `whole-export-r2/` has 3,662 rows after two root-approved,
independently verified language-routing repairs (generic Prakrit and Vedic).
All 3,660 prior rows are identical; the earlier freeze below is historical.
The current revision passed actual generic parsing with no field/tag losses
or unknown registrations. Its guarded regeneration uses an immutable 742-row
baseline and passes three focused approval/route tests. Fresh independent
original-output review subsequently passed and this revision was installed.

The frozen fifteen-column CSV contains **3,660 rows**; its full accounting audit contains **5,232 records**, including physical containers, their witnesses, source analyses and documentary records. All **742 historical Entry_Keys** remain in their original order. The canonical 742-row CSV is unchanged. Output hashes and all input hashes are in `whole-source-export-status.json`.

The assembly applies only explicit root adjudications, including 347 accepted grammar metadata changes, the complete 63-decision reverse review, and four provisional marked-ha encodings. Overlapping approvals must have identical values. Earlier proposals and rejected readings remain evidence. There are **486 emitted rows with an explicit uncertainty tag**; a legible form with an uncertain orthographic interpretation is not excluded merely for that reason.

## Validated

- Eight focused assembly/export tests passed in 2.93 seconds (`tmp/chakma-independent/export-tests.log`, workspace-relative).
- The actual `make_cldf.parse_file` parsed all 3,660 rows with the installed source route, without monkeypatching, conversion, unwanted splitting, field differences or tag differences. `whole-source-export-parser-validation.json` records the exact hashes and checks.
- All emitted languages, grammatical tags and primary citations resolve. The five work/article-level auxiliary references are registered. Their registration does not assert independently consulted editions or passages.
- Bengali-script literal Form, Original and Native are preserved; no IPA is fabricated. Existing transcription policy is `convert: false` with `preserve_hyphens: true`; no phonemic profile is appropriate for this literal-script preview.
- The root-approved Tanchangya and Chenchu registry records are installed with blank collection coordinates/location/quality. Three Tanchangya and four Chenchu witnesses route to these records. Named gojha labels remain exact source provenance, not invented geographic dialect tags.

## Explicit exclusions and decisions still under review

Every exclusion has a keyed audit record. The 395 numbered comparator slots comprise 209 accepted existing-language witnesses, four Chenchu witnesses, 142 unresolved language identities, 37 outside-family controls, two blanks and one unresolved segmentation. Only the first 213 are lexical comparator rows. The 100 separately attested Chakma heads are also emitted.

Other exclusions in revision 2 include 24 unresolved comparative identities, two damaged basic-wordlist glyphs, two unresolved thematic cells, source blanks and controls, and documentary/abbreviation records. The source-analysis prose in numbered groups 48, 65 and 87 preserves every quotation, component, definition and cited work in the audit. Its language/stage remains unresolved; literary work identity alone does not justify assigning Sanskrit or Old Bengali. Independent original-image review confirms those three specific dispositions; they are not a blanket policy excluding source components.

PDF pages 1–6 and 110 have original-page documentary proof and no target lexical quotations. Documentary and abbreviation records do not generate artificial lexical heads. All source evidence remains in the package.

## Remaining gates

1. Independent whole-output completeness and source review; resolve any reported material errors in a new revision, preserving this freeze.
2. Final source-wide metadata, relation and definition-scope acceptance, including the explicitly unresolved source-analysis witnesses.
3. Guarded reproducible source installer and canonical metadata reconciliation, then repeat-install and actual canonical-parser tests. The current historical importer must not overwrite a future whole-source installation.
4. Full CLDF/database build and browser QA remain explicitly deferred under the current instruction not to build the database. This preview makes no production or browser-completeness claim.
