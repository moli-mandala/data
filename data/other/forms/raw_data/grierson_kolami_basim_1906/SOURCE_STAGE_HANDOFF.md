# Source-stage handoff, 2026-09-27

The ingestion checklist is active, including grammar/interlinear, comparative-source and OCR-heavy requirements. Whole-source extraction, original review, metadata/profile/reference reconciliation, source-local graph checks and guarded canonical installation pass. Full ingestion completion is **not** claimed: CLDF/DB rebuilding, compiled identities and graph/reference checks, full tests and browser QA are deferred by the current source-stage task.

## Scope and editorial decisions

- All nine chapter pages, plus source-attributed pp. 182, 623, 626 and 627, are accounted for: 169 structural parents, 224 grammar interpretations, 64 interlinear lines and 561 physical atoms.
- Installed 827 rows: 447 Kolami, 367 Naikri, six Telugu, two Kannada, two Toda, and one each Tamil, Tulu and Kui. Section counts are 212 grammar, 560 atomic, 51 complete source-bounded expressions and four outside attestations.
- Accepted audit: 1,570 records and 830 represented keys, including three existing rows reused for two physical printings. No new duplicate physical occurrence is emitted. Ten notation symbols and one hypothetical `avnd` are documentary. The `Bāp-` / `phōkān` continuation keeps both raw atom keys and one lexical emission.
- Two historical source lects are registered beneath existing Kolami/Naikri with blank coordinates/Glottocodes. Fifteen local variants and two ordered component sets (four links) are closed and acyclic; no borrowing or cross-language ancestry is invented.
- Original literals, aligned English, punctuation, boundaries and historical rejected readings are retained. The display lowercases and changes `ṅ` to `ŋ`, retains source hyphens/`ñ`/`ï`, and explicitly accounts for 155 terminal punctuation trims. Native and Phonemic remain blank. There are 37 uncertain emitted rows; current mark and grammatical holds are visible, rather than silently guessed or excluded.
- The sharper-original revision guards 35 grammar Form repairs, one source-label correction and 24 specimen field repairs. The fresh-output packet exhaustively checks the 59 Form/Gloss repair values plus the separately documented `3.m.` label correction. Old full-page hypotheses rejected by native crops remain historical.
- Three reuse scope keys are audit-only. Compiled persistent aliases, additional citations and IDs have not been proved; [REUSE_AND_PIPELINE.md](integration/REUSE_AND_PIPELINE.md) states this limit explicitly.

## Review and installation evidence

The final independent original-output review is `root-independent-review/fresh-final-output-original-audit.json` (SHA-256 `2bc4a5c8be3ec9d127d04211ac6729f5eb2451bcba132bb7c61c46b56c737864`): 20 fresh actual rows, zero literal/gloss mismatches, all literal-repair controls, all 560 specimen metadata rows, nine whole-source pronoun consistency repairs and 20 supplemental controls. `scholarly-notes-r6-acceptance.json` (`6c7b7ee4ee605a2800080bf37ca801f967c01c4f5a6fade8d2134e894d722547`) separately accepts only two source-defined suffix Notes replacements, preserving all other fields and the same sample.

Final revision: `integration/revision6/review-freeze.json` (`8586c1bcdf129767b41fd195e4e18ccaaf2c07dc81e7bb62e4d15f406acc9ed3`). All earlier frozen candidates and the prior disabled bundle remain immutable. The six-target transaction was activated and executed by root; `installation/root-acceptance.json`, transaction and receipt identify the exact approved outputs. Installation never allocates durable IDs.

Canonical hashes:

| Asset | SHA-256 |
| --- | --- |
| CSV | `747f12ae281fbb7b6054ec3a22614ff962adc5e283bbe26dd79a2c6501e1d223` |
| YAML | `ecb61670e4c65579386799cd2955cd905ae409a362ca24ca6f2c36dda155caff` |
| Accepted audit | `8744607a57d48ee4b7e5cb4df8f848e7fec5cf1f10493dfb94e5292b7c91e46b` |
| Profile | `89ae32f7df58cc3298cdd62bab1e106a7842cfdb02debc2efaabca34c6fc979f` |

`CANONICAL_SOURCE_STAGE_VALIDATION.json` records the actual, unmodified canonical parser and SourceMeta: all 827 ordered keys and all 15 fields pass in canonical/NFC/NFD modes; 827 initial identity keys prevent prompt/gloss collapse; all graph endpoints close and are acyclic; both source-reference policies and two registered dialect rows pass. All 632 prior Bib entries are semantically preserved, both added references format, durable identity SHA-256 `4977c4e205f66233366ee1c00a9c938c9c78e3c087b63e02a94832647525c28a` is unchanged, and repeat installation is an exact no-op. Root log: `tmp/kolami-root-canonical-source-checks.log` in the workspace.

Focused checks: eight transaction fixtures passed, including recovery after interrupted writes, rollback and drift rejection; 18 source/protected-metadata tests and two Notes-only/history tests passed. The independent conservation review reproduces all physical atoms, the compound join and both punctuation-driven expression merge/split changes. Three installed regressions in `data/tests/test_grierson_kolami_basim_installed.py` pass in 0.73 seconds; the final workspace log is `tmp/jambu-next-source/kolami-installed-focused-tests-final.log`. The initial new test queried a source-level identity flag instead of the installed file-default policy; its failed log is retained, and the corrected test verifies the actual public file policy plus all 827 initial dedupe keys.

## Deferred gates

No full CLDF/DB build, compiled alias/persistent-ID/reference/graph verification, full suite, browser entry QA, commit, push, publication or deployment was performed. Representative source-stage witnesses include `ïm`, the reviewed `amd` reuse, the literal optional-letter pronouns, `Bāp-phōkān`, the ordered `ad` + `net` component claim, and source-qualified Basim whole expressions. Representative app-entry evidence must follow a later authorized rebuild and browser pass.
