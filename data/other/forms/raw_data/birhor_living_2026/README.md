# Birhor Living Dictionary — whole-snapshot recovery

Current status (2026-09-27): **whole pinned snapshot installed at source stage**. The user authorized reuse verbatim, “living dictionary is allowed per sarva”; see `whole-snapshot-recovery/user-reuse-authorization-20260927.json`. Earlier preparation and permission holds below are historical.

The canonical `20260927-birhor-living.csv` has **3,480 rows**, covers every one of the **2,746 head records**, and preserves all **3,100 prior preparation keys**. The complete annotation review accounts for 98 notes, 967 parenthetical spans and 82 outside-note decisions. Its 4,090 candidate analyses include 610 exact repeated note attestations reused with every citation retained; the durable audit has 3,795 source units. Ninety-nine documentary/control units remain audit-only, not omitted target forms. Thirty-three installed rows retain typed uncertainty, including head/phonetic discrepancies, scoped head recovery and source-note interpretation. All 41 explicit variant links resolve locally.

Independent review passed 20 sampled source units (22 output rows) and five controls, with zero material errors, alongside targeted head/grammar, full morphology/interlinear and prose-attribution reviews. Seven focused tests passed. Postinstallation checks used the actual registered YAML/profile route: all 3,480 keys and Original/Native/Phonemic/Gloss/Notes/Source fields survived, with zero conversion errors. The existing Birhor (`birh1242`) registry identity is used; no new dialect or locality was inferred. All three primary/secondary references resolve and format in memory.

Reproduce the reviewed proposal with `prepare_full_source.py`; `install_full_source.py --install` checks the frozen source/review/test hashes and exact root approval before copying source inputs. `validate_installed_source.py` performs scoped registered-route verification without a database build. Historical preparation files remain unchanged. The full audit and independent evidence are in `whole-snapshot-recovery/`; `SOURCE_CHECKLIST.md` records the gates.

**Deferred by the user's explicit boundary:** full CLDF/database build, complete test suite, compiled reference/identity/graph validation and browser QA. No claim of end-to-end ingestion completion or live application verification is made.

## Earlier preparation history

Pinned public snapshot acquired2026-09-22; upstream last modified2026-08-08.
2,746 entries and2,746 senses, all with IPA and Devanagari spelling. This is
preparation only: no source CSV or bibliography installation yet.

`inventory.py --database <downloaded-birhor.db> --output <directory>` verifies
the pinned hash and SQLite integrity, reads the source in read-only mode, and
freezes selected public lexical fields. It never builds a Jambu database.
`lexical-snapshot.jsonl` preserves original IDs and JSON field values.
`source-references.json` preserves91 auxiliary bibliographic records.
No audio/photos were downloaded; speaker/account metadata is excluded from
the lexical snapshot. The compressed upstream database remains in scratch.

The dictionary/external-dataset ingestion checklist applies. Compare the live
version against the archived2021 dictionary; do not present them as identical.
The live snapshot includes98 notes,10 morphological analyses, one interlinear
analysis and90 entries with auxiliary sources. No dialect assignments or entry
relationships occur. These need source-specific review rather than reuse of
the Kharia parser's restrictive assumptions. See the manifest for acquisition,
rights metadata and outstanding gates. No compiled validation is authorized.

## Sense preparation checkpoint

`prepare_senses.py --output <directory>` reproducibly emits3,097 candidate
meanings from2,746 entries, using upstream entry and sense IDs plus explicit
meaning numbers. 260 English glosses start with1.;259 actually contain multiple
numbered meanings, while SK2Zpf18HB7HpPPFHWLK only says “1. bend head”.
The provisional parser removes recognized grammatical labels within each
numbered meaning; it preserves botanical author labels and semantic qualifiers.
It does not propagate a final POS label backwards across meanings.

29 entries carry review issues:23 headword/IPA discrepancies, four single-letter
optional-segment notations, two botanical headword-scope questions (one also a
transcription discrepancy), and one empty English gloss. Both transcription
fields are preserved without choosing a correction. Optional forms are not yet
expanded; alkusi is not treated as an optional string of Birhor phonemes.
Notes, morphological analyses and interlinearization remain in the pinned
snapshot/audit pending structured mapping. The ten morphological strings and
one interlinear string must not be silently discarded. Auxiliary citations
include media credits and botanical resources, so they require role review
before entering the bibliography.

Two focused tests pass (0.09s). These test preparation invariants and known
edge cases, not acceptance of all entries. Source interpretation, reuse basis,
profile, fresh output audit and installation remain outstanding.

## Provisional CSV and profile

`import_source.py --output <scratch>` now writes3,100 proposed15-column rows
and a2,746-entry audit; there is deliberately no install or build mode yet.
Two botanical headwords with unresolved lexical scope and the missing-English
entry are withheld. Four explicitly parenthesized single segments yield six
additional readings across their meanings, with resolvable variant keys.
Twenty-five rows retain headword/IPA-disagreement uncertainty. Source headword
stays in raw Form/Original; source IPA drives display conversion, and the native
Devanagari spelling stays separate. For optional readings, Phonemic carries the
expanded reading and the audit retains the exact parenthesized source string.

Candidate profile preserves short vowels, nasalization, glottal stop, spaces
and hyphens; conventional consonant symbols are transliterated explicitly.
Coverage passes all3,100 forms in NFC and NFD. Four focused tests pass(0.51s).
This is not a fresh acceptance audit or proof of pipeline integration.

The109 rows with residual source annotations retain source notes and explicitly
labelled morphological/interlinear analysis. Do not turn the source's
ethnobotanical reports into independent medical claims. Of91 reference records,
89 identify botanical taxa in Tropicos, one is the2022 ethnobotanical work,
and one is an excluded photo credit. Every original reference survives in
`reference-roles.json`; auxiliary bibliography registration remains pending.

## Reviewed draft, awaiting reuse clarification

`audit_output.py --output <scratch> --seed 2026092201` exercises the actual
source parser with temporary in-memory metadata/profile settings. It writes
only scratch CSV/sample files and runs no compiled-data or database build.
The20-record source-versus-output sample found0 material errors, including
numbered senses, nasal vowels, aspiration, botanical glosses, native spelling,
notes and citations. `output-audit-2026092201.json` records the review.
All3,100 parser outputs preserve expected source layers, notes and tags.
Five explicit person/number labels have source-anchored decisions in
`grammar-decisions.json`; the lice/louse English gloss is not treated as proof
of Birhor grammatical number. Six focused tests pass(1.09s).

`sources.bib` contains the main dictionary and two secondary works. All used
citation keys resolve and each reference formats successfully. Canonical global
bibliography/settings/profile/CSV are still untouched for this source.
Only source-file installation remains after the reuse basis is established;
compiled/full validation remains deferred. A question is pending with the user
whether existing Living Dictionaries permission extends to Birhor. The source
checklist requires confirming redistribution/inclusion, and the permission
record available locally names Kharia rather than Birhor.
