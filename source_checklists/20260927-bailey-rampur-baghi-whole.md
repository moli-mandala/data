# Bailey 1920 Rampuri–Baghi whole-source review

Source files installed locally on 2026-09-27. This is **not completed ingestion**: the user has prohibited a new database build, so full compiled identity/graph, reference rendering, build and app/browser gates remain deferred.

## Scope and extraction

The active source-ingestion checklist applies, including glossary, survey/grammar and scanned-source addenda. The source is the original 310-page Bailey PDF, SHA256 `7a1577a2d2a24eca8b95e9b1ed700ef01f716d1069e4468f36a673611d093b39`. Shared introduction pp.113–115, Rampuri introduction p.116, Baghi introduction pp.116–117, grammar pp.131–140, 44 complete lect-specific responses pp.141–142, commentary pp.142–143 and glossary pp.144–147 are recovered. Boundary/context witnesses are explicitly accounted; comparisons are not assigned to target lects.

Installed counts: Rampuri package **647 rows**, Baghi **698 rows**, **640 audit units each**. All **584 legacy keys** survive in order. There are **761 additional rows**; 67 Rampuri and 57 Baghi legacy rows have reviewed field changes, including notes/tags as well as literal corrections. These counts are derived from the actual old/new rows in `pass4-metadata-reconciliation.json`.

The glossary accounts for all 245 physical headword cells: Rampuri has 243 accepted cells and two cross-references; Baghi has 240 accepted cells, two cross-references and three unpaired Rampuri controls. Supplementary audit units with no target export are 114 Rampuri and 91 Baghi, including companion-lect evidence and contextual controls. They are retained in the audit rather than silently dropped. Forty-four physical responses yield 45 complete expressions because one explicitly printed Baghi alternative expands to two whole responses.

## Editorial decisions and unresolved interpretation

Literal breves, macrons, nasal marks, underdots and underlined digraphs are preserved. The profile applies only the established display conventions `w → v`, `ṅ → ŋ` and lowercase display; Original preserves the source spelling. NFC and NFD input checks pass. Explicit grammatical labels become registered tags; full responses carry both sentential and multiword-expression tags.

Four generic Kochi examples are emitted once under existing `kc`, with explicit attribution uncertainty; they are not assigned to Rampuri or Baghi. The former body hold is recovered as `jĕā́` with interpretation uncertainty. Damaged/stacked vowel marks and tentative unpunctuated alternatives retain typed uncertainty and source notes. No phonemic, donor or etymological interpretation is invented. Only explicit within-lect variants receive links; all variant keys resolve within the installed proposals. Existing language IDs, coordinates and identity append orders are retained.

## Audit and validation

Three failed whole-output audits are preserved. After a complete glossary rereading and the confirmed sentence16 correction `ōrah → ōruh`, the fresh fourth sample passed **20/20 units**, covering **33 exact emitted rows** and excluding all earlier samples. Related return expressions were also rechecked against the original. Evidence is `raw_data/bailey_rampur_1920/root-independent-whole-output-audit-20260927-pass4.json`.

Both CSVs and audits regenerate byte-identically in an isolated source-only sandbox. The accepted output, profile and producer hashes are frozen. Actual canonical parsing preserves all 1,345 rows, ordered keys, Original, Notes, Source and Tags. **13 focused tests passed** in `tmp/koya-whole/paired-installed-tests.log` at the workspace root. Earlier glossary-only tests are retained as historical evidence in each package; the active installed-source tests now cover the expanded scope.

The coordinated installer updates CSVs, audit trails, profiles, YAML and only the two bibliography entries' coverage/provenance fields. Other bibliography fields and all other entries are preserved. Both old glossary installation commands delegate to the guarded whole-source installer, preventing regression to a partial glossary.

## Deferred gates and representative forms

Full CLDF/database generation, persistent identity/graph validation, formatted-reference regeneration, production build, and browser QA against rebuilt data are deferred under the explicit user instruction. No remote job, publication or database build was run. Representative source entries are Rampuri `gōhṛo` (horse), recovered `jĕā́` (body, uncertain interpretation), and Baghi complete response `Ĭs ănda rupaue mŏṅga ōruh`. They are source-file representatives, **not verified app entries**; the app remains on the prior database.
