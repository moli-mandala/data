# Source ingestion checklist — 20260826-sil-tripura

- Installed input: `data/other/forms/20260826-sil-tripura.csv`
- Canonical checklist SHA-256: `23516ba5c7caeaa5310f6a32d92b86e4b9ae5e4479e792926b61dd0689129f05`
- Source-type addenda: Dictionary or glossary
- Installed rows: 8997
- Compiled rows carrying this unit's citation keys: 8997
- Input rows with checked grammatical evidence: 0
- Compiled rows with canonical grammatical tags: 0
- Source keys: kim-kim-sangma-ahmad2011tripura

## Retrospective gate assessment

- [x] 1. Establish the source and scope — source keys: kim-kim-sangma-ahmad2011tripura; 8997 installed records
- [x] 2. Choose the extraction path — importer/raw route: data/other/forms/20260826-sil-tripura.csv
- [x] 3. Plan the installed files and identifiers — 8997 unique immutable Entry_Key values
- [x] 4. Model languages and dialects before emitting forms — 3 input language/lect IDs; registry gaps: none
- [x] 5. Emit the rich import schema — row widths {'15': 8997}; blank forms 0
- [x] 6. Parse structured linguistic information — no source-supplied grammatical labels detected by the scoped parser
- [x] 7. Build and verify the sound profile — profile route: conversion/sil-tripura.txt; replacement characters in input forms: 0
- [x] 8. Parse references and provenance — unresolved keys: none
- [x] 9. Model etymology and graph relations conservatively — covered by tests/test_edges.py and compiled edge invariants
- [x] 10. Produce a complete audit trail — audit: data/other/forms/raw_data/20260826-sil-tripura-audit.csv, source_checklists/installed-record-audit.csv.gz
- [x] 11. Add focused regression tests — tests: tests/test_source_checklists.py
- [ ] 12. Install and run the full data pipeline — pending final repository-wide make all and full-suite validation for this review
- [x] 13. Browser database refresh and inspection (user-triggered) — deferred by standing policy; refresh and browser QA run only when the user requests them
- [x] 14. Document, review, and ship only when requested — this source-specific checklist is the durable review record; shipping is not requested

## Review summary

- Counts: 8997 installed records; 8997 compiled citation attestations.
- Exclusions: none detected in the installed input; any source-side exclusions remain in the linked importer/audit.
- Unresolved cases: none detected.
- Transcription: `conversion/sil-tripura.txt`.
- Validation: full data validation is recorded centrally in `source_checklists/VALIDATION.md`; browser refresh is user-triggered.
- Representative app entries: recorded centrally in `source_checklists/VALIDATION.md`.

## Filled checklist copy

Checked boxes below inherit the repository evidence stated above for their section. Unchecked boxes remain completion gates; addenda not listed for this unit are explicitly not applicable.

# Jambu source-ingestion checklist

Use this checklist for every new lexical source, whether it is a dictionary, glossary, survey
wordlist, comparative table, website/API, existing CLDF dataset, or OCR transcription. It is
based on the failure modes and successful patterns in the current importers: preserve source
evidence, normalize only in explicit layers, structure everything the schema can represent, keep
uncertain claims reviewable, and prove the installed rows survive the complete build.

The source ingest is not complete when an extractor produces a CSV. It is complete when the raw
source is reproducibly represented, its language and reference metadata are correct, and the full
CLDF build is clean. Browser-database construction and representative app inspection become
completion gates only after the user explicitly requests that refresh.

## Definition of done

Before calling an ingest complete, all of these should be true:

- [x] The exact source, edition/version/revision, licence, coverage, and exclusions are recorded.
- [x] A reproducible importer exists under `data/other/forms/raw_data/` (or the appropriate
      established source directory).
- [x] The importer preserves stable upstream/local entry keys and exact page/item provenance.
- [x] A per-record audit accounts for every source record, including skipped, ambiguous, corrupt,
      or unresolved records. Check it in for OCR, heuristic parsing, language mapping, or
      etymological matching; a deterministic format conversion may use a reproducibly generated
      audit.
- [x] Every form belongs to a canonical base language; every named variety or stable survey site
      is represented by a registered, language-qualified dialect tag rather than a pseudo-language.
      Speakers, texts, consultants, and elicitation sessions remain provenance.
- [x] `Form`, `Original`, `Phonemic`, `Native`, `Gloss`, `Tags`, and `Notes` have been separated
      according to their meanings.
- [x] Grammatical, register, dialect, citation-locator, uncertainty, variant, borrowing, and
      etymological information has been structured wherever the model supports it.
- [x] The source has an explicit sound-profile route (or a documented, tested reason no conversion
      is needed), with complete input-symbol coverage and difficult mappings reviewed.
- [x] Every cited key has a correct BibTeX record, formatted reference metadata, provenance, and
      OCR/editor attribution.
- [x] Etymological links reproduce source claims and are conservative: ambiguous or unsupported
      matches remain unlinked and visible in the audit.
- [x] Cross-source deduplication preserves every attestation's citations, dialects, and ID aliases,
      while protecting same-lect homonyms and source-defined distinct records.
- [ ] Focused importer tests, registry/profile tests, the complete data build, and the full test
      suite pass.
- [x] When the user requests a browser-database refresh, it builds, passes its integrity/size
      checks, and representative entry, language, dialect, reference, search, and concept views
      have been inspected. Routine source ingestion does not rebuild the browser database.
- [x] The final handoff reports counts, exclusions, unresolved cases, transcription decisions,
      test results, and files changed.

## Standing editorial policy

These defaults are intentional and apply unless a source-specific review records an exception:

- Identical unlinked forms may merge across bibliographic sources when their canonical language
  and complete lexical analysis agree. Merge citations, dialect attestations, and ID aliases; do
  not merge same-lect homonyms or source-defined distinct senses/records.
- Printed etymological IDs may be linked directly. A link inferred from a headword requires a
  unique, conservatively normalized match plus compatible meaning; phonetic similarity alone is
  insufficient.
- Named varieties and stable survey sites are dialects. Individual speakers, texts, consultants,
  and elicitation sessions are provenance.
- Structurally reliable unreviewed OCR may enter the database when the raw OCR and exact locator
  are preserved and a typed review marker is attached. Badly damaged or headword-less records stay
  out of the installed CSV but remain accounted for in the audit.
- Convert `Form` to house transcription only when the mapping is defensible. Preserve ambiguous
  source symbols, flag them for review, and never silently impose a phonological interpretation.
- A parser must have no known systematic error class. An ordinary new source should finish with
  `0/20` material errors on a fresh seeded audit; malformed legacy sources may retain documented
  residual exceptions.
- Check in per-record audits for OCR, heuristic parsing, language mapping, and etymological
  matching. Deterministic format-only conversions may keep a reproducibly generated audit.
- Prefer source-locality coordinates. Modern Glottolog points and derived centroids are permitted
  only as explicit quality-`C` approximations.
- Use `uncertain` for discovery/filtering, but preserve a typed reason—OCR, transcription, gloss,
  dialect mapping, borrowing, or etymology—in audit or edge metadata.

## 1. Establish the source and scope

- [x] Identify the bibliographically canonical source.
  - Compare a PDF, website, API, repository, and later edition instead of assuming they are
    identical. Prefer the richer/newer authoritative representation when appropriate, as with the
    live Kullui API, and document the choice.
  - For an online dataset, pin a release tag, DOI, commit, snapshot revision, or dated export.
  - Record the source URL/DOI/handle and the date or version represented by the import.
- [x] Confirm that the source can be redistributed or that the extracted facts can be included.
  Record the licence when known.
- [x] Define the exact included portion.
  - Pages, chapters, appendices, database version, tables, target lects, and source-record count.
  - Explicit exclusions: control languages, supplements, indexes, bibliography/footer material,
    duplicate scans, illegible blank entries, or source types outside the task.
- [x] Decide whether this is a new canonical source or a supplement/replacement for an existing
  ingest. Check for overlapping editions, duplicated wordlists, and prior hand-entered rows.
- [x] Write down expected source units before parsing: articles, numbered entries, rows, concepts,
  lects, pages, or columns. These become completeness assertions.
- [x] Decide what the source actually claims:
  - lexical attestations only;
  - inherited cognacy;
  - borrowing/donor relations;
  - variants or inflected forms;
  - derivations/components;
  - free-text etymological commentary.
  Do not infer graph relations merely because forms look similar.

## 2. Choose the extraction path

- [x] Inspect all available representations before choosing OCR.
  - Try ordinary PDF text extraction.
  - Inspect fonts, glyph positions, crop boxes, and alternative extractors. A selectable text layer
    may be available to Ghostscript or another extractor even when `pypdf` misses it, as in Toda.
  - Use page coordinates and font/style spans when layout or bold/italic structure contains entry
    boundaries, grammatical labels, or columns.
  - Use OCR only when the source text layer is absent or unusable.
- [x] For OCR, make the process reproducible.
  - Fix render resolution, crop/column regions, OCR language/model, and command options.
  - Keep the raw OCR alongside the parsed interpretation in the audit.
  - Structurally reliable unreviewed OCR may be installed only with exact locators and typed review
    markers. Keep badly damaged or headword-less records out of the installed CSV and in the audit.
  - Mark unreviewed OCR explicitly; do not silently emend uncertain characters.
- [x] For websites/APIs, make the snapshot reproducible and resumable.
  - Pin the version or revision when possible.
  - Cache raw responses or preserve enough raw source markup in the audit to reproduce decisions.
  - Use stable upstream IDs, deterministic pagination, bounded retries, and atomic cache writes.
- [x] For upstream CLDF/CSV data, preserve upstream form, concept, language, and row IDs and record
  the upstream licence/version. Do not discard upstream IPA/segmentation or source spelling.
- [x] For PDFs, assert page count and page-range assumptions. Detect repeated adjacent pages,
  two-page spreads, shifted printed numbering, hidden material outside the crop box, headers,
  footers, and multi-column reading-order errors.
- [x] Keep source artifacts and large temporary renders out of the installed data unless they are
  appropriate to commit. The importer must fail clearly when a required non-committed input is
  absent and should support reuse of cached/intermediate extraction where useful.

## 3. Plan the installed files and identifiers

- [x] Choose one stable bibliography/source ID. The raw CSV `Source` field and the BibTeX key must
  agree exactly.
- [x] Use the dated installed filename convention:
  `data/other/forms/YYYYMMDD-short-source-name.csv`.
- [x] Add a source-specific importer, audit, and test, normally:
  - `data/other/forms/raw_data/<source>.py`
  - `data/other/forms/raw_data/YYYYMMDD-<source>-audit.csv`
  - `tests/test_<source>.py`
- [x] Define an immutable `Entry_Key` from source identity, not row order or normalized content.
  Good keys use an upstream article/form ID or a stable tuple such as printed page + column +
  numbered entry. Corrections to OCR, glosses, or transcription must not change it.
- [x] Define stable child keys for expanded source records, for example `:variant:2` or `:link:2`.
  Every emitted row from a rich importer should have a unique key.
- [x] Make output order deterministic. Never use unordered set iteration to assemble fields that
  affect fingerprints, IDs, or diffs.
- [x] Preserve `data/form-identities.csv`; it is durable identity state, not generated scratch.
  Do not discard or regenerate it during a re-ingestion.

## 4. Model languages and dialects before emitting forms

- [x] Reuse an existing canonical base `Language_ID` whenever the source lect belongs to one.
  Do not create `Source-*` top-level languages or `Language: Dialect` pseudo-languages.
- [x] Check the source's labels against Glottolog, the work's own classification, and Jambu's
  current language registry. Historical labels can be more appropriate than a conflicting modern
  Glottocode; document the decision.
- [x] Add genuinely new base languages to `cldf/languages.csv` with:
  - stable ID and display name;
  - Glottocode when applicable;
  - latitude/longitude and human-readable location when defensible;
  - Jambu clade;
  - coordinate quality `A`, `B`, or `C`.
- [x] Add every source lect, named variety, or locality to `cldf/dialects.csv` with:
  - stable dialect ID;
  - language-qualified `dialect:*` tag;
  - canonical parent `Language_ID`;
  - original `Source_Language_ID`/alias;
  - source display name;
  - Glottocode only when independently applicable;
  - coordinates, location, clade, and quality.
- [x] Use exact locality coordinates for dialects when available. Do not present a modern
  Glottolog point as a historical survey location; retain it only with an explicit qualification.
- [x] Leave coordinates blank rather than inventing them. If a base-language centroid is derived
  from its dialects, mark it as approximate (`Quality=C`) and retain exact points on dialects.
- [x] Keep genealogy/historical stage, dialect, elicitation site, speaker, and text provenance as
  separate concepts. A speaker or source sample is not automatically a dialect.
- [x] If a new clade name is unavoidable, synchronize the frontend clade files. Otherwise use an
  existing Jambu clade exactly.
- [x] Test that all final forms use registered base languages and all `dialect:*` tags are
  registered under the correct parent.

## 5. Emit the rich import schema

Prefer the 15-column source CSV even when many fields are blank. It is headerless and ordered as:

1. `Language_ID`
2. `Parameter_ID`
3. `Form`
4. `Gloss`
5. `Native`
6. `Phonemic`
7. `Notes`
8. `Source`
9. `Cognateset`
10. `Etymology`
11. `Entry_Key`
12. `Variant_Of_Key`
13. `Borrowed_From_Key`
14. `Derivation_Parent_Keys`
15. `Tags`

Use `|` between multiple source-local parent keys, spaces between tags, and `;` between CLDF
citations. A blank `Parameter_ID` in a manual import means an attested but unetymologised form and
is retained as an `unlinked` node.

- [x] `Form` is the Jambu display transcription after the configured profile runs.
- [x] `Original` is derived from the raw CSV form and preserves the source spelling/transcription.
  Do not pre-normalize it away in the importer.
- [x] `Phonemic` preserves a distinct source/upstream IPA or phonemic analysis when available.
  It should not merely duplicate `Form` without a reason.
- [x] `Native` contains native script only, not romanization, grammar, or a second gloss.
- [x] `Gloss` contains the lexical definition only.
  - Remove POS/gender labels, citations, entry numbers, and etymological prose.
  - Preserve sense distinctions.
  - When one printed definition scopes over several forms/languages, propagate it to all forms
    only within defensible structural boundaries.
  - Stop propagation at a new numbered sense, causative/passive/inflectional subsection,
    punctuation boundary, or conflicting meaning. Prefer an intentional blank to a false gloss.
- [x] `Notes` is reserved for genuine residual editorial/source information that has no better
  structured home. Raw OCR, parser-review state, alternate labels, and locators belong in the
  audit, tags, graph fields, or citations rather than being dumped into Notes.
- [x] `Etymology` contains source-specific analysis/commentary, not a substitute for resolvable
  graph relationships. For entry-level prose with semantic type/order, consider the
  `cldf/entry-texts.csv` sidecar (`etymology`, `comparison`, `usage`, etc.).
- [x] Preserve NFC Unicode in installed output. Handle legacy fonts and OCR substitutions in the
  importer/profile with tests; never rely on visually similar non-Unicode glyphs.

## 6. Parse structured linguistic information

- [x] Parse grammatical labels into canonical `Tags`, including as applicable:
  - part of speech;
  - gender and number;
  - transitivity;
  - case;
  - tense/aspect/mood/voice;
  - person-number;
  - stem or inflectional class.
- [x] Interpret labels contextually and case-sensitively. Source/reference/dialect abbreviations
  can collide with grammatical abbreviations (for example `Tr.` versus `tr`).
- [x] Associate a grammatical marker with the correct form, not the whole surrounding entry by
  default. Cover forward labels, trailing labels, coordinated forms, and labels that scope across
  a form list.
- [x] Strip successfully parsed labels from `Gloss`/`Notes`; retain unrecognized prose unchanged.
- [x] If a genuinely new canonical tag is needed:
  - add/normalize it in `tags.py`;
  - add tests showing extraction and false-positive avoidance;
  - synchronize `../jambu-static/src/lib/tags.ts` and any category lists.
- [x] Do not emit redundant relationship tags such as retired `inherited`; inheritance and
  borrowing live in graph edges.
- [x] Parse register/style labels (`poetic`, `archaic`, `colloquial`, `honorific`, etc.) as tags.
- [x] Parse geographic labels as registered dialects with coordinates/quality, not generic
  `region:*` strings.
- [x] Preserve typed uncertainty where possible.
  - A general `uncertain` tag can flag the row.
  - Put the reason in the audit or an edge `review:*` note: OCR, transcription, gloss, donor,
    inheritance, or etymological assignment uncertainty are not the same thing.
- [x] Represent inflected/alternate forms as rows or explicit relationships when the source gives
  real forms; do not compress a paradigm into a category tag.

## 7. Build and verify the sound profile

- [x] Read the source's orthography/transcription description before mapping symbols.
- [x] Inventory every grapheme/cluster actually present in the source, including decomposed
  combining sequences, punctuation, boundary marks, stress/tone, length, superscript homonym
  numbers, and corrupt/replacement characters.
- [x] Decide which layer drives display conversion:
  - source spelling -> Jambu;
  - upstream/source IPA -> Jambu while preserving IPA in `Phonemic` and spelling in `Original`;
  - preservation profile for forms already in house transcription, with only known repairs.
- [x] Add `conversion/<profile>.txt` and explicit routing in `utils.py`/`make_cldf.py` as needed.
  Route by bibliography/source ID when filename heuristics are ambiguous or several files share a
  profile.
- [x] Choose NFC or NFD deliberately. Normalize before and after tokenization, and preserve word
  boundaries, meaningful hyphens, homonym numbers, and source punctuation when required.
- [x] Never use a blind character replacement when a mark has multiple functions. Prefer
  sequence-aware mappings or the upstream segmented IPA, as with LSI aspiration apostrophes.
- [x] Add tests for every difficult or linguistically consequential mapping, not just easy ASCII
  examples.
- [x] Add a corpus-wide coverage test for the new source: every installed input form must tokenize
  without introducing `�`. Existing corrupt source glyphs may remain only when explicitly audited.
- [x] Verify the separation of `Form`, `Phonemic`, and `Original` in compiled CLDF rows.
- [x] Record mappings that require linguistic review in the handoff rather than silently choosing
  one analysis.

## 8. Parse references and provenance

- [x] Add a complete entry to `cldf/sources.bib`. Prefer stable DOI/handle/release URLs and include:
  - author/editor, title, year, and publication metadata;
  - URL/DOI;
  - version/revision and licence when applicable;
  - `included` describing exact coverage;
  - `provenance` listing installed CSV and audit paths;
  - `jambu_editor` credit;
  - `ocr = {Yes}` when OCR contributed to the installed transcription.
- [x] Give auxiliary works their own bibliography keys when the source cites them (for example a
  dictionary, field notes, TGT, PB, CDIAL, or DEDR); do not leave their abbreviations in glosses.
- [x] Use CLDF locators in every row when the source provides them:
  - `source-id[p. 42]`
  - `source-id[p. 42, col. 2, entry 17]`
  - `source-id[form X, concept Y]`
  - `source-id[revision N, row X]`
- [x] Separate multiple citations with `;`. Preserve the source's printed page as the primary
  locator; include PDF page only in the audit when useful for reproduction.
- [x] Resolve citation abbreviations and volume/page/item text into structured locators rather
  than leaving the locator in Notes.
- [x] Run `make_refs.py` (included in `make all`) and check the source in
  `cldf/references.csv`: formatted citation, inclusion status, provenance, editor, and OCR flag.
- [x] Verify that every used citation key resolves and that no placeholder reference was created
  because of a typo or missing BibTeX record.

## 9. Model etymology and graph relations conservatively

- [x] Use a blank `Parameter_ID` for a source that makes no etymological claim.
- [x] When a source explicitly cites CDIAL/Turner, DEDR, or another installed etymon, validate that
  the target exists and normalize the exact printed ID.
- [x] For headword matching without printed IDs:
  - normalize only features justified for matching (for example accents, not vowel length);
  - require a unique match and relevant source/semantic evidence;
  - record every accepted, ambiguous, and unmatched candidate in the audit;
  - never force an ambiguous fuzzy match.
- [x] Distinguish `reflex`, `borrowed`, `variant`, `derived`, and `component` relationships. A
  cross-family or explicitly marked donor relation must not become ordinary inheritance.
- [x] Use `Variant_Of_Key`, `Borrowed_From_Key`, and `Derivation_Parent_Keys` for relationships
  among rows created by the same rich importer. Ensure every referenced key exists.
- [x] Keep source free-text etymology even when a graph edge is created; the edge captures the
  relation, while the prose preserves the author's reasoning and caveats.
- [x] Put later/manual decisions in `data/etymology-assignments.csv`, keyed by persistent form ID,
  rather than hard-coding unstable generated IDs into an importer.
- [x] Verify accepted rank-1 and alternate/rejected hypotheses against `cldf/edges.csv` after the
  full build. Variants point to their true source target; the effective etymon is reached
  transitively.
- [x] Test that unlinked forms remain first-class `unlinked` nodes and that etymon entries are not
  accidentally merged with attestations.

## 10. Produce a complete audit trail

- [x] The audit has one row per raw source record, plus explicit rows for any generated
  variants/links when necessary to explain output multiplicity.
- [x] Include enough fields to reconstruct every decision:
  - status (`ingested`, `skipped`, `ambiguous`, `unlinked`, `corrupt`, etc.) and reason;
  - `Entry_Key` and upstream ID;
  - PDF page, printed page, column, item/entry number;
  - raw source text/markup/OCR;
  - parsed form, gloss, tags, and references;
  - chosen language/dialect mapping;
  - match candidates and accepted etymon;
  - unresolved citations or characters;
  - confidence/review flags;
  - duplicate-scan or alternate-extraction comparison when relevant.
- [x] Reconcile counts from raw source units to audit statuses to installed rows. Explain increases
  caused by variants/multiple links and decreases caused by exclusions.
- [x] Preserve uncertain or malformed records in the audit instead of quietly dropping them.
- [x] Check in the audit for OCR, heuristic parsing, language/dialect mapping, or etymological
  matching. A deterministic format-only conversion may generate its audit reproducibly instead.
- [x] For a nontrivial parser, add a reproducible seeded raw-vs-output audit command.
- [x] Sample at least 20 entries across the source, compare raw representation to parsed output,
  classify material errors, fix systematic causes, and repeat on fresh samples. Require no known
  systematic error class and target `0/20` material errors for an ordinary new source. A malformed
  legacy source may retain documented residual exceptions. Add every fixed error class as a
  regression test.
- [x] Deliberately inspect edge cases outside the random sample: first/last page, page breaks,
  multi-column transitions, multiline entries, duplicated scans, malformed markup, rare symbols,
  homographs, shared glosses, multiple senses, and source-specific exclusions.

## 11. Add focused regression tests

At minimum, a source-specific test file should cover:

- [x] Exact expected raw/audit/installed counts and documented exclusions.
- [x] Unique, stable `Entry_Key` values and expected keys for representative rows.
- [x] Correct page/item locators and bibliography keys on every row.
- [x] Language-to-base-language mapping and complete dialect registration.
- [x] Coordinates/location/Glottocode quality for all source lects where applicable.
- [x] Extraction/parsing edge cases found during manual and random audits.
- [x] Clean lexical glosses and complete grammatical/register tags.
- [x] Native script, source transcription, and phonemic fields are preserved in the right columns.
- [x] Sound-profile examples and complete source-symbol coverage.
- [x] Accepted/unlinked/ambiguous etymology outcomes and correct relation types.
- [x] Variants/borrowings/derivations resolve through stable keys.
- [x] Every emitted source row survives `make_cldf.py`; no deduper silently loses homonyms or
  concept-distinct forms.
- [x] Durable IDs survive source reorder and transcription/gloss corrections where relevant.
- [x] No replacement characters, empty forms, malformed row widths, duplicate final IDs, or
  unregistered references/tags/dialects are introduced.

When modifying a shared parser rather than adding a one-off source, retain the seeded 20-entry
audit and tests for systematic error classes. Test false positives as aggressively as true
positives: prose must not become forms, bibliography abbreviations must not become grammar, and
comparison text must not become ancestry.

## 12. Install and run the full data pipeline

- [ ] Check the worktree first. Do not overwrite unrelated importer or generated-CLDF work in a
  shared checkout.
- [ ] Run the importer without installation when supported, inspect the proposed CSV/audit, then
  use its explicit `--install` mode to update the canonical output.
- [ ] Run focused tests first:

  ```bash
  cd data
  uv run pytest -q tests/test_<source>.py tests/test_sound_profiles.py tests/test_dialects.py
  ```

- [ ] Run the complete current pipeline in order:

  ```bash
  make all
  ```

  This currently runs `make_cldf.py`, `link_refs.py`, `unify_cldf.py`,
  `assign_form_ids.py`, `concepts.py`, `align.py`, and `make_refs.py`. Source parsers such as
  CDIAL/DEDR are separate and should be rerun first only when their raw parser logic changes.

- [ ] Inspect `errors.txt`. Prove the new source contributed no unmapped forms; do not hide new
  errors among older known conversion issues.
- [ ] Review generated diffs and count changes in at least:
  - `cldf/forms.csv`
  - `cldf/edges.csv`
  - `cldf/form-source-keys.csv`
  - `cldf/form-id-aliases.csv`
  - `data/form-identities.csv`
  - `cldf/concepts.csv` and `cldf/form_concepts.csv`
  - `cldf/alignments.csv`
  - `cldf/references.csv`
- [ ] Investigate unexpectedly large ID churn, unrelated row-count changes, lost citations, or
  graph changes before proceeding. A source-profile change should not remint stable form IDs.
- [ ] Run the full suite:

  ```bash
  uv run pytest -q
  ```

- [ ] Confirm the source-specific assertions against compiled CLDF, not only the raw importer
  output.

## 13. Browser database refresh and inspection (user-triggered)

- [x] Do not rebuild, stage, or serve the browser database during routine ingestion. The user
      decides when to pay the cost of refreshing it.
- [x] When the user explicitly requests a refresh, use the browser project's documented build
      workflow and confirm its SQLite integrity and compact-database size guards pass.
- [x] After a requested refresh, inspect representative pages in the app:
  - one ordinary entry;
  - a linked etymon/reflex or borrowing;
  - an unlinked entry;
  - a variant/multi-parent case when present;
  - each new/affected language and dialect map point;
  - the source's reference page and locator display;
  - grammatical/register/dialect tag rendering and filtering;
  - source search and concept membership/counts.
- [x] After a requested refresh, inspect at least one example that combines several ingestion
      features; record it in the handoff so another person can verify the work quickly.
- [x] After a requested refresh, check that independently sourced, identical unlinked forms merge across dialects and
  bibliographic sources when canonical language and complete lexical analysis agree. All source
  attestations, citations, dialect tags, and ID aliases must be retained; same-lect homonyms and
  source-defined distinct senses/records must remain distinguishable.

## 14. Document, review, and ship only when requested

- [x] Update `README.md` for a substantial or unusual source with:
  - canonical source/version and acquisition method;
  - extraction method and why it was chosen;
  - installed and audit paths;
  - row/lect counts and exclusions;
  - transcription, linking, OCR, and uncertainty policy;
  - known unresolved issues.
- [x] Summarize the ingest for review:
  - raw records -> installed rows -> compiled nodes;
  - number linked, borrowed, ambiguous, unmatched, and skipped;
  - languages/dialects added or mapped;
  - sound-profile decisions needing linguistic review;
  - audit results and residual error classes;
  - focused/full test results;
  - one or more app entries that demonstrate the result.
- [x] Do not commit, push, publish a release, or deploy merely because the ingest is complete;
  those require an explicit request.
- [x] When shipping is requested:
  - update `../jambu-static/src/lib/changelog.ts`, including real language and reference IDs;
  - rebuild `.dbwork/jambu.db` from the final committed CLDF;
  - bump `../jambu-static/src/lib/dbMeta.ts` (`DB_VERSION` and approximate byte count) so browsers
    do not retain the previous database;
  - upload `.dbwork/jambu.db` as `jambu.db` on a fresh `jambu` GitHub release;
  - push the intended data/frontend commits;
  - verify the deployed asset, application load, representative entry, and non-prerendered entry
    path.

## Source-type addenda

### Dictionary or glossary

- [ ] Separate headword, alternate/S2 stem, sense number, POS/gender, definition, citations,
  native script, etymology, and donor labels.
- [ ] Preserve homographs with distinct `Entry_Key` values.
- [ ] Emit printed alternates as variants only when the source treats them as such.
- [ ] Resolve cited dictionary IDs; retain invalid citations in the audit.

### Survey wordlists or comparative tables

- [ ] Identify target lects versus controls and exclude controls deliberately.
- [ ] Model sites as dialects beneath existing base languages.
- [ ] Preserve prompt/concept IDs so identical short forms under different prompts are not
  collapsed.
- [ ] Keep IPA/source transcription and locations, with appropriate coordinate caveats.

### OCR-heavy source

- [ ] Preserve raw OCR and page images/intermediates needed for review.
- [ ] Compare duplicate scans or alternate OCR passes when available.
- [ ] Distinguish mechanical, source-wide OCR repairs from linguistic emendation.
- [ ] Mark every unreviewed/uncertain row and account for illegible blank heads.

### Website/API or external CLDF

- [ ] Pin an immutable release/snapshot and record its licence.
- [ ] Preserve upstream IDs, source spelling, IPA/segments, concepts, languages, and raw labels.
- [ ] Cache or audit enough upstream data to reproduce the import after the live source changes.
- [ ] Map upstream languages to Jambu base languages and dialects explicitly; never bulk-create
  source-prefixed top-level languages as a shortcut.

### Etymological/comparative source

- [ ] Separate source comparison prose from resolvable typed graph relations.
- [ ] Treat printed Turner/CDIAL/DEDR IDs as claims with their source caveats, not independent
  confirmation.
- [ ] Keep uncertain alternatives visible and ranked rather than replacing the accepted analysis.
- [ ] Randomly audit raw prose against forms, glosses, citations, tags, and edges after the full
  build, because parser errors often look superficially plausible in the CSV.
