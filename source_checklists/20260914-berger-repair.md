# Berger repair review — 14–15 September 2026

**The audited Berger defects have been repaired in the source importer, installed source files and cognate catalog. Full CLDF integration remains pending.** The [14 September audit](20260914-berger-audit.md) is retained as the before-state. This is a substantial repair of a malformed legacy source, not a certification that the full dictionary OCR or machine translation is clean.

The mandatory `SOURCE_INGESTION_CHECKLIST.md` is active with dictionary/glossary, OCR-heavy and etymological/comparative addenda. Survey, website/API and external-CLDF addenda are inapplicable. Browser refresh and publication were not requested.

## Counts and accounting

| Measure | Repaired source |
| --- | ---: |
| Reconstructed source units | 9,602 |
| Parsed entries, including variants | 11,029 |
| Auto rows | 11,733 |
| Historical hand-entered rows | 39 |
| Total installed rows | 11,772 (formerly 10,703) |
| Per-record audit rows | 11,974 |
| Excluded parsed records | 207 |
| Compatibility records added by the preservation pass | 810 |
| Attested paradigm rows added | 110 |
| Additional image-reviewed forms | 18 |
| Rows with canonical class tags | 549 (formerly zero) |
| Retired source-key redirects | 542 |
| Unaccounted old source keys | zero |

Audit statuses count parsed records, while an explicit multi-Turner record may emit several rows. Five extra emitted rows account for the difference between non-excluded audit records and installed rows. Every emitted key is covered by the audit. Every prior installed key is either active or has a redirect. Full source grammar/context is retained for all 11,772 rows.

Canonical number tags: 522 plural, 179 singular and 20 double-plural rows. Class tags overlap for forms with multiple classes: H 33, HF 14, HM 15, X 152, Y 366. These counts are not claims of exhaustive image-verified grammatical analysis.

## Repairs addressing the audit

### Article identities and accepted cognate evidence

The old crosswalk remains frozen for public-key continuity; its fuzzy regeneration command is disabled. Stable keys are attached to the original physical starting lines. New source articles receive position-based keys, and surviving keys are not renumbered after a recovered entry. Variant keys require matching forms rather than reuse of an old ordinal.

For the 157 changed-form candidates affecting 149 accepted cognate sets, all 327 Berger evidence keys in those sets now refer to separate `:legacy-graph` records containing the original evidence. This is conservative protection, not a finding that all 157 candidates were wrong. The four proven wrong bindings—udder disease/back of head, partridge/mill part, flowering herb/feather and bad/head cold—therefore no longer attach the current article to the old meaning. Their current public IDs stay associated with the current article; the catalog uses separate preserved evidence. Re-evaluating the scholarly correctness of every historical set remains future work.

The 39-row hand-entered tranche had additional neighbouring-article contamination. Explicit source alignments now separate `awaáji` from `awáaz`, Yasin `ayáś` from `ayáa`, porridge `baát` from verbal `bá-at-`, Nager `balóṅ` from `balbán`, and the historical `baɣ` component record from `ba` “kiss”. Both original evidence and distinct source articles remain represented. The `baɣ` component interpretation remains explicitly unreviewed. **25 historical direct Turner links** are now withheld where the repaired source does not establish an unhedged claim; their numbers and reasons remain in Notes. `T 1197 oder 1221` is an unresolved alternative, not accepted ancestry.

### Layout, compounds and grammar

The main systematic boundary error was the use of fixed horizontal cutoffs on shifted scans. The new parser learns four column margins and accommodates hanging `davon`/`dazu` heads. PDF font evidence separates italic lexical material from Roman German definitions. Header expansion retains compounds and bound verbal material. The frozen hidden text supplies typography only: its damaged spelling never replaces the OCR.

Grammar is read from the current form's header rather than an arbitrary prefix of the whole article. A later verb or printed plural does not change the headword's POS or number. Noun classes are structured as canonical H/HM/HF/X/Y tags, including combined classes. Literal suffix-only paradigms, class labels and less certain scope remain in full Notes. Full explicitly attested paradigm forms can become rows; suffixes do not generate unattested words.

Image-verified repairs include `trin` as a Y-class noun “favour”, the class distinctions and plural forms of `ámin/ámis/ámit`, the separate plurals of `agón` and `aadát`, Nager `halčík`, `-phíliṣ`, `luṭhúri gaíṅ`, `ġaáṣ`, `phaláan`, `dáal :t-`, `zanqán/zanzán`, literal homonym numbers, and the false German headword `dumm`. The formerly blank `balaneéś man-́` shares the source definition of `balán man-́`; the tentative Turner comparison stays in etymological notes.

### Relations and index references

All emitted variant and derivation endpoints now exist; the variant graph is acyclic. The remaining 63 source relations (40 derivation, 23 variant) are withheld with their original target and a typed reason in Notes and `relations.json`.

Exact complete-form lookup resolves **736** index references. Another **1035** lack a unique defined target, and **205** are relative references requiring further source review. Their literal target remains in Notes; their English definition is blank. This avoids presenting a lookup instruction as a definition or inventing a match. Ambiguous homonyms, missing targets and reference cycles are not linked by similarity.

### Transcription, languages and provenance

The destructive global `ġ → g` OCR replacement is removed, and `conversion/berger.txt` covers `ġ`. Source images correct reviewed readings. Existing cached `g` spellings cannot safely be changed globally: the g/ġ distinction remains a typed review issue where unverified. Full installed-form profile coverage and the real source conversion path are tested.

Every row uses canonical `Bur`. Hunza, Nager, Yasin and literal `NH` have registered source-qualified dialect tags. Coordinates reuse existing quality-C approximate registry points. The meaning of the `NH` abbreviation was not independently verified; its registry description says so. Source bibliography/provenance now points to the repair package and accurately describes the historical hand-entered and unreviewed translated layers.

## Source, reproducibility and exclusions

Berger (1998), *Die Burushaski-Sprache von Hunza und Nager*, Teil III, dictionary printed pp. 9–486. Pinned PDF SHA-256: `864cd94f8c41237aae2408e154f7b8d9e21c911b90f7d3ff6dd261568f05c0dc`.

Included: PDF 7 right, 8–50, 52–246, 247 left. Excluded: preface on PDF 7 left, weaker duplicate spread 51, proper names on 247 right, index and back matter on 248–327. The scan is local and not redistributed; no open licence for the book is asserted. The frozen OCR and editorial records are source-derived review evidence.

The [repair package](../data/other/forms/raw_data/berger_2026/README.md) contains pinned OCR pages, font alignment, original installed rows, historical graph evidence, source-image overrides, gold alignments, translation cache, source aliases, per-record audit and manifests. Ordinary reproduction needs neither the PDF nor the translation model. Runtime hash checks reject changed editorial inputs. The original August map/audit/editorial files remain historical inputs.

Changed German definitions were translated with pinned Argos `de_en 1.3` using CTranslate2 4.8.2 and SentencePiece 0.2.2, one CPU thread and saved source-hash-bound output. English machine translations remain unreviewed except explicit editorial corrections. Model SHA-256: `becc2b0011f8249fcb89be9ecb75ba0d876b1fab93c28ee6ff0420936897d637`.

## Source-image QA and residual exceptions

The original audit's 20 entries and targeted cases were revisited. A fresh deterministic sample (seed **20260915**) from the repaired proposal had **17/20 material errors before its recorded corrections**, with three lacking a material error. Every correction is preserved in `reviewed.json`; the frozen before rows and per-entry source-image decisions are in `fresh-sample-before.json` and `fresh-sample-review.json`. Additional source inspections covered false Roman-text heads, shifted and hanging-indent columns, split compounds, circled headings, missing definitions, source-key continuity, gold mappings and profile failures.

This is the checklist's documented malformed-legacy exception. The repair does **not** claim a clean fresh 0/20 sample or exhaustively corrected dictionary. Residual OCR, g/ġ distinctions, unreviewed machine translations, uncertain grammar scope, excluded readings, unresolved references/relations and the scholarly validity of legacy graph evidence remain review work. Typed reasons and exact source context accompany the records; generic `uncertain` remains on every installed row. The generated source-checklist wording has been corrected so it will not report “unresolved cases: none”.

## Validation and deferred gates

**48 focused checks pass.** Coverage includes repaired source cases and the frozen fresh-sample corrections; column/hanging-indent boundaries; the 39-row gold tranche; canonical class, plural and dialect labels; all-row sound-profile coverage; zero dangling relation endpoints or variant cycles; all catalog evidence keys and real construction of the 149 protected sets; complete audit coverage; and durable-ID/alias checks against the current Berger registry.

The real `make_cldf.parse_file` path preserves every source key, Original form, Note and supplied grammatical tag, with zero source conversion errors. This is a scoped source conversion check, not a complete CLDF build.

Regeneration after installation is exact: both source CSVs, alias tables and JSON reports match byte-for-byte; the decompressed per-record audit matches exactly. All per-record hashes verify. No source key, dialect or class-registration gaps remain in these scoped checks. The identity registry was read, not rewritten. See `berger_2026/validation.json` for checks and installed-file hashes.

The full `make all` pipeline and complete test suite are **deferred** under the 8 GB workspace policy. Existing `.github/workflows/python-app.yml` is a push/PR pytest workflow, not a dispatchable full-build runner. No suitable authorized remote full-build runner was available, and no publication was authorized to trigger one. Required gates remain open; they were not replaced by the scoped checks.

Consequently the generated CLDF forms, edges, identity snapshots, references and browser database have not been refreshed for this repair. Existing generated files may contain unrelated changes from other workspace work. Integration still must run the full pipeline, inspect source/ID/edge diffs and conversion errors, and confirm the catalog changes in compiled CLDF. Browser construction and app QA become applicable only after a requested refresh. No commit, push or deployment was performed.

## Checklist gate disposition

| Gate | Disposition |
| --- | --- |
| 1–3 Source/scope, extraction, durable identifiers | Source package installed; scope and exclusions pinned; aliases checked |
| 4–5 Language/dialect model and rich schema | Canonical Bur, four registered dialect tags, 15-column rows |
| 6 Linguistic structure | Scoped grammar plus complete source context; residual OCR scope explicitly unreviewed |
| 7 Sound profile | Complete token coverage and focused conversion checks |
| 8 References/provenance | Source records updated; unresolved lexical references audited; generated reference rebuild pending |
| 9 Graph | Source endpoints and protected catalog binding checked; compiled graph gate pending full build |
| 10 Audit | All emitted keys accounted for; fresh sample and legacy residual exception documented |
| 11 Focused tests | Results below and in package validation record |
| 12 Full pipeline/full suite | Deferred to a suitable authorized runner |
| 13 Browser database/app QA | Not requested; no refresh performed |
| 14 Review/publication | This review and source checklists updated; publication not requested |

## Representative entries for the next app refresh

These existing public IDs were checked against the durable registry; the browser currently reflects the older build.

| Entry | Source locator | Existing public ID | Expected repaired feature |
| --- | --- | --- | --- |
| `trin` | printed p. 431 | `f_j5imouv6au7sc` | noun, Y class, “favour” |
| `ámin` | printed p. 17 | `f_myj4yh4eek5cg` | human classes and separate attested plurals |
| `halčík` | printed p. 187 | `f_r55e556ixipyo` | Nager plus full plural suffix notation |
| `-phíliṣ` | printed p. 329 | `f_y7cqzpz74uptw` | complete bound form, X class and cultural definition |
| `balaneéś man-́` | printed p. 33 | `f_2mn6rttn2iwp4` | restored shared definition, tentative Turner link withheld |
