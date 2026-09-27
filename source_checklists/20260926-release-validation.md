# 26 September 2026 release validation — in progress

Baseline: data `c8a88cc637edfd9e6656d41ddb52331c3ae63be4`; frontend `30707ac` / audit follow-up `8c71572`. This records installed inputs and completed checks, not final publication approval. Root must fill the final build/test/artifact/browser results below before claiming release completion.

## Installed scope

The canonical input inventory contains **82 new CSV files and two modified existing files**, with **63,513 current raw rows versus 5,515 baseline rows: 57,998 net additions**. New files span 72 canonical language IDs. These are source rows, not deduplicated application entries. The [completeness ledger](20260925-source-completeness-ledger.tsv) and each source README/audit govern actual coverage.

Whole-source or complete archived-snapshot recoveries include Hahn Asur, Norton Korku, CFEL Koda/Mahali, MLA Gorum/Gta/Kharia, and numerous Bailey/Grierson packages. Scope-limited installations remain explicitly limited: Ho is the archived A–C response, Darai chapter 3, Ollari vocabulary pp. 48–77, Chakma main wordlist, Peterson Turi Appendix 1, Gaddi Appendices II–III, Yerava its comparison table, Roy Birhor Appendix I, and credited numeral packages their source tables. Several Bailey 1920 inputs remain glossary-only, including Rohru. Historical filenames containing “pilot” or two page numbers do not independently define current scope.

Uninstalled Rohru whole-chapter drafts, Darai later chapter/clause proposals, Birhor Living, Fawcett's two-page candidate and the Turner/Census Koda/Attapady analysis holds are **not installed release additions**. They remain open. No claim of complete ingestion is made for these sources.

## Source settings, profiles and references

- Source metadata validation reported 276 configured files and 268 citation keys; the sidecar check covered 77,254 assignments in 90 sidecars.
- All 82 new source YAMLs statically declare importer commands, existing command paths and stem-based legacy identity. Reviewed proposal files required by installed importers remain release dependencies; a proposal filename alone is not an exclusion rule.
- Shared parser settings preserve authorial pronunciation as conversion input where declared and source boundary hyphens where meaningful. Data/frontend tag registries include the newly required grammatical categories.
- House-transcription changes affect 137 existing profiles and 83 new profiles. Source spelling and authorial pronunciation remain separate from display conversion. Release reconciliations are recorded for [Koda print](../data/other/forms/raw_data/cfel_koda_2022/release-profile-reconciliation-20260926.json), [Koda API](../data/other/forms/raw_data/cfel_koda_api_2026/release-profile-reconciliation-20260926.json), [Grierson Koda](../data/other/forms/raw_data/grierson_koda_birbhum_1906/release-profile-reconciliation-20260926.json), [Sansi](../data/other/forms/raw_data/grierson_sansi_1922/release-profile-reconciliation-20260926.json), and [Hahn](../data/other/forms/raw_data/hahn_asur_1900/release-profile-reconciliation-20260926.json).
- [Hislop bibliography review](../data/other/forms/raw_data/hislop_kuri_muasi_1866/release-bibliography-review-20260926.json) records its release reference correction. Source-local references and uncertainty should not be replaced by invented identifications or dates.
- Release regression coverage is in [profile reconciliation tests](../tests/test_release_profile_reconciliation_20260926.py) and [file-default identity tests](../tests/test_file_default_dedupe.py). The cleanup fix applies file-default `dedupe_by_entry_key` settings as well as citation-level settings; otherwise valid keyed records and their source analyses were being merged. Kisan/Khirwar YAMLs now explicitly declare that flag, with package regressions. Their raw CSVs were unchanged by the settings repair.

## Builds and identity continuity checked so far

Four sequential `make all` runs exited 0. The first exposed source-key losses despite process success. The second incorporated the file-default dedupe repair and recovered 1,013 additional expected keys, leaving six Kisan and one Khirwar key. Their explicit YAML opt-ins resolved those remaining losses in the third build: the compiled verifier then resolved **all 63,513 expected immutable keys** across the 84 changed canonical inputs, with global language/edge endpoint closure. The fourth build incorporates the Ho citation serialization repair described below. Its final strict verifier again resolves all **63,513 keys with zero issues and 736 references**, and confirms the malformed Ho reference is gone. Final committed-checkout validation remains pending; successful process exit alone is not release acceptance.

The second-build durable registry was streamed against the exact prebuild local LFS object, SHA256 `5ec2588f97a569cc7a1de68f59474c03612fa7a4266582be61d70dca29efcfc9`:

- All **1,112,553 previous IDs retained**, none missing or newly retired; 55,120 new IDs.
- No previous Source_Key, Language_ID or Fingerprint changed. Fifteen retired IDs returned to active with no other field change.
- Of 2,107 changed existing rows, 1,662 changed legacy positions and 429 source citations. The 47 Original and 24 Gloss changes overlap in 64 records, all checked against canonical keyed inputs: 42 Bonda/Didayi records separated previously folded survey spellings/senses; 22 Nagaraja Nihali records expanded optional length to the long original key plus an existing `:short` counterpart. All changed lexical fields match raw canonical values; all 22 short variants exist.

The second alignment CSV is 99,817,504 bytes (95.193 MiB), below 100 MiB by 5,040,096 bytes. Recheck the final file. Forms and durable identities already use Git LFS. Research queues, source books and local test/build caches are excluded from ordinary release staging; they have not been deleted.

The regenerated source auditor completed successfully across **275 units**; its profile-incomplete count fell from 61 to one. Generated per-source checklists, the manifest and `installed-record-audit.csv.gz` are active release evidence and included in staging. The remaining incomplete profile gate is not silently marked complete.

## Verified changed graph continuity

Bounded postbuild checks, performed after the third full build and before the citation-only fourth rebuild, found:

- All 13 reviewed Kannauji reassignments from CDIAL 10896 to its `10896-5` extension survive as accepted reflexes; none retains the old parent. Their changed Notes explain the stale NeoJambu exact-text cohort count.
- Zargari retains all **522 immutable keys and exact language/Original values**, with exactly **289 expected graph tuples** (78 source variants plus 211 accepted sidecar edges), no missing, extra or duplicate tuple. The 201 sidecar-linked children explain the revised 243 unlinked heads.
- Bhatri's **1,666** and Gondi's **4,718** graph tuples exactly match reviewed sidecars, without missing, extra or duplicate tuples.
- Rajasthani's 15,370 touched tuples are accounted for by 4,173 reviewed sidecar tuples, 9,958 literal source-parent assignments, 1,237 unchanged exact-baseline tuples, and two separately verified source/alias cases. No new unexplained current edge was found.
- Proto-Kherwarian's 2,919 listed source keys include **487 existing unresolved reconstruction-head aliases**, down from 842 in the exact baseline. All remaining unresolved keys already failed baseline resolution; none is newly unresolved. The reconstructed head itself remains present in the checked example. This historical alias debt is explicitly retained, not described as a passing full-source key gate.

## Focused failure reconciliation

The first isolated release selection finished with **61 failed, 2,384 passed and 18 skipped**, including 26 newly failing nodes. This is a diagnostic result, not a passing release gate. An exact `c8a88cc6` checkout reproduced all 35 overlapping failures. Full traceback comparison found **21 matching observed causes, ten same mechanisms with changed operands, and four generic symptoms without offending-record identities**. These are not automatic waivers; changed source/graph operands were separately investigated. The comparison applies to the first failing assertion, not later assertions masked by that failure. Final committed-checkout test results remain pending.

- [Ho's citation serialization repair](../data/other/forms/raw_data/ho_mla_2004/release-citation-serialization-repair-20260926.json) moves archived reference prose from 93 citation locators into Notes, retaining the archive-entry citation. Forms, immutable keys and row order are unchanged; six focused tests passed. This prevents embedded reference prose from producing invalid serialized bibliography keys.
- [Bote family-predicate evidence](audits/20260926-bote-family-predicate-evidence.json) records Page 2024's explicit Indo-Aryan classification. The test's family predicate now recognizes Bote while retaining its `Other` display subgroup; 49 added relation tuples match reviewed sidecar decisions. No canonical linguistic record was changed for that test repair.
- Two source importers emitted dialect tags with literal spaces in the canonical language component. The [Kagani](../data/other/forms/raw_data/bailey_kagani_1920/release-dialect-token-correction-20260926.json) and [Hill Madia](../data/other/forms/raw_data/vaz_hill_maria_2011/release-dialect-token-correction-20260926.json) repairs percent-encode those components in canonical Tags, importer constants and the registry. All lexical fields, keys and citations remain unchanged; prior source audits remain immutable.
- Marked-borrowing continuity now accounts for the additional short Nihali `hela`, preserving its reviewed parent 14158. The provisional Nihali cohort remains exactly 4,299 reviewed records plus 22 explicit short readings (21 variant edges and one existing borrowing analysis).
- Domaaki compiled tests now account for three reviewed moon-form sidecar links to 4661. Source graph checks use exact reviewed sidecar decisions through [the shared test helper](../tests/reviewed_graph_policy.py), with negative fixtures for missing/unapproved/rank/position/status drift.
- Coordinate tests retain an exact allowlist for source-locality-only languages; no locations were invented. Source ordering tests preserve the historical prefix while allowing declared stem-keyed additions. Southworth and Kharia tests distinguish house display transcription from exact source Original/Phonemic fields.
- The extension detector now excludes generic derivative sections from inference candidates and witnesses while preserving numbered-context reset and explicit extension handling. Five focused detector tests passed; the final build must regenerate the inference table and record its actual delta.

A bounded batch of source-package, coordinate and input-order tests passed **17 tests**. Global compiled checks remain pending regeneration and the parent's sequential rerun.

## Frontend semantics and validation

The coordinated frontend changes distinguish cross-language compounds/derivatives from same-language derived terms, preserve exact-copy aliases and citations, and keep all distinct typed-edge notes/sources after deduplication. Repeated component positions remain in the graph while displaying their child once. Derivative-only search counts its displayed languages.

Completed focused frontend checks: **10 builder tests passed**, **two Node display regressions passed**, and `npm run check` reported **0 errors and six warnings in unchanged components/routes**. ConceptPicker is unrelated pre-existing work and excluded from this release scope. A lightweight production-compacted parity fixture exercises all 13 parity sections and six corruption classes. Compact/v1 parity must include hydrated lexical fields, references, aliases, typed edge positions and unioned evidence; parity alone cannot detect evidence lost before v1 construction.

## Final gates — root to complete

- [x] Fourth full data build exited 0 after metadata, dialect-token, inference and Ho citation repairs. Final artifact hashes belong in the release audit.
- [x] Changed-source verifier resolves all 63,513 expected immutable keys; scoped changed-graph checks above pass. Historical Proto-Kherwarian aliases remain explicitly unresolved. Reconfirm final committed artifacts in the release audit.
- [ ] Isolated full release test selection: final result, baseline comparison and explicitly excluded pending-only tests. The first result and exact-baseline reconciliation are recorded above; the full committed-checkout suite remains pending. An earlier shared check had a profile-policy failure before reconciliation, so its output is not a final success record.
- [ ] Browser database transform, compact/v1 parity, exact compressed/uncompressed sizes and asset integrity.
- [ ] Browser checks on ordinary reflexes, cross-language derivatives, repeated components, source-defined homonyms, preserved citations, alias redirects and missing-entry SPA behavior.
- [ ] Final changelog/source-scope reconciliation, staged dependency review, commits/push/publication and live verification under the release workflow.

Working evidence and reproducible staging/identity review scripts are retained locally under `tmp/release-20260926`; final durable results should be summarized here or in the committed frontend release audit. No full-source completion claim should be inferred from this provisional release checklist.

## Subsequent preservation correction

The exact-commit suite found that duplicate-attestation cleanup discarded dialect and grammatical tags from later witnesses. Cleanup now unions those tags in stable order. The fifth full `make all` passed, followed by seven focused checks including every Malvi response. Comparing all 991,379 compiled rows found tag additions on 10,654 records across 93 languages, no tag losses, and no changes to IDs, order, spelling, glosses, citations or other fields. All other 15 published CLDF files are byte-identical. The ignored `forms-legacy.csv` development diagnostic is outside the publication scope. See [the preservation audit](audits/20260926-tag-preservation.json).

The prior browser databases are superseded. The final committed suite, rebuilt browser assets, compaction parity and live deployment remain pending; their final results are recorded in the frontend [db-v39 release audit](https://github.com/aryamanarora/jambu/blob/main/release-audits/db-v39.json).
