# Darai whole-thesis proposal

The complete 486-page thesis has owner source review and an explicit page census. The approved whole-source stage is installed. All 485 earlier keys are preserved. Independent sample206 passed 0/20; the dependency-only baseline patch reproduced exact approved outputs. Full integration gates remain deferred.

## Reviewed scope and output

`whole-proposal-freeze.json` pins the two CSVs, audit, sound profile, and graph/tag reconciliation. There are 11,304 expanded source units: 8,373 emitted entries, 2,431 citation-preserving reuses, and 500 explicit controls. The emitted entries comprise 8,361 author-transcription rows and 12 literal-preservation rows. All 485 installed keys remain distinct.

Chapters 1–13, all grammatical paradigms and morphemes, all five sample narratives, and their aligned analyses are accounted for. Front matter, questionnaire/respondent pages, and references have explicit context dispositions. The narrative inventory has 337 printed text identifiers; editorial statement boundaries and literal/idiomatic translations may expand these identifiers without pretending they are additional physical attestations.

All 539 unique source notes have editorial decisions. The complete 100-case residual label census distinguishes exact interlinear analyses from calendar dates, text identifiers, sentence-internal roles, and genuinely ambiguous source groups. Resolved original-page alignments override obsolete extraction flags; conditional particles and isolated markers do not inherit lexical verb POS. Explicit source categories are structured only from exact labels, headings, and individually reviewed statements. Subject and object agreement retain their distinct roles. Historical quoted witnesses do not inherit the contemporary Chitwan scope. Tanahun and Nawalparasi forms are explicitly distinguished; recording locations are retained as source context, not invented dialect assignments.

## Uncertainty and exclusions

Four compressed notation groups remain literal because the source does not determine alternate scope. Eleven entries have blank Phonemic because source glyphs or notation cannot be assigned a secure interpretation; one further entry preserves Roman spelling without a claimed phonemic reading. Original retains the source text. These are retained attestations with typed uncertainty, not omissions or guessed readings. Source conflicts retain their contradictory evidence and withhold only the conflicting grammatical labels.

The 500 controls retain exact source-unit records and exclusion reasons. Abstract grammatical labels, explicit other-language comparisons, and nonlexical context are not silently turned into Darai words. Identifiable source morphemes and complete responses are included.

## Reuse and graph decisions

Exact reuse requires equality of exported linguistic fields, Notes, effective old and new tags, sound route, source claims, attribution, and uncertainty. Every occurrence retains its original row, citation, and evidence pointer in `whole-proposed-audit.jsonl`. Every retained target accumulates all occurrence citations. Eighteen independently recoverable Chapter 3 expression blocks align with their earlier physical attestations; their historical keys and complete source translations are retained rather than duplicated.

Existing graph fields are preserved. All 49 Appendix 3 root records and 555 inflections retain their source-defined verb role. Thirty-seven root-boundary hyphens previously consumed as gloss delimiters are restored from the printed root columns; the unmarked p376 root remains unmarked. Eight additional variant links require explicit source statements of variation or pronunciation, and four derivations follow the source's labelled base-plus-suffix diagrams. Fourteen additional base relations preserve explicitly paired adjective and causative derivations; no constituent alignment is inferred. Merely co-listed or interchangeable words receive no inferred variant edge. Reuse redirects these reviewed references to surviving targets; distinct parent roles are preserved.

## Applied installation changes

1. Replace `20260921-dhakal-darai.csv` with the frozen author-transcription CSV, preserving its stem and all old keys. Add a separate literal-preservation CSV with conversion disabled and alternate splitting disabled.
2. Install the frozen expanded `dhakal-darai` profile. The additions include literal `g` identity and canonically decomposed input equivalents; existing 485 Phonemic values remain unchanged. Structural removals are context-specific and recorded per row.
3. Update the existing source YAML and importer to reproduce the approved whole-source files and audit. The literal companion reuses `dhakal2011darai` without a duplicate global source declaration. Keep source-key identity deduplication enabled for both routes.
4. Apply `whole-dialects-proposed.csv` for Tanahun and Nawalparasi only after shared metadata review. Retain existing Chitwan and Pidrahani registry records and blank coordinates for the new source-defined districts.
5. Replace only the `dhakal2011darai` bibliography scope with `whole-source-proposed.bib`; preserve the independent Kotapish entries and all unrelated bibliography.
6. Reconcile the source README, checklist, completeness ledger, historical importer tests, and independent review reports. Keep failed or superseded samples as historical evidence.
7. Run focused canonical importer parity, generic parser, sound/profile, citation, tag, graph-target and old-key preservation checks. No database build is implied by installation.

## Validation and deferred gates

The final focused run passed 66 tests in 5.37 seconds, including the actual generic parser for all proposed rows, exact Original/Phonemic retention, full profile coverage and NFC/NFD equivalence, tag preservation, citation preservation, reuse accounting, and sampler exclusions. The parser uses both source-local proposed YAML files and supplies the proposed profile in memory; this is not a claim that the canonical YAML or dialect registry has already been updated.

`whole-independent-selection-20260927206.json` is the fresh sample following the heading, role, root-boundary and interlinear-scope repairs. Its exclusions include both historical selections, the superseded whole-source samples, and their consolidated targets. Original sample files remain unchanged. `root-whole-targeted-source-review-20260927.json` separately confirms the source interpretations on pages 156, 164, 170 and 343.

Independent whole-source acceptance, canonical installation and focused postinstallation checks passed. Full database/CLDF builds, compiled identity and graph survival, and browser QA remain deferred under the user's instructions. No complete-ingestion or production claim is made.
