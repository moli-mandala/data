# Whole Naikī recovery status

Root, 2026-09-27. This package is in recovery, not installed or complete.

Current first readings account for 83 prose evidence groups on 570–571,
11 omitted pronoun-table cells on 570, and one separately located genitive
suffix witness on 623. These 95 groups are not final row counts: some contain
multiple explicitly printed forms, multiple analyses, controls, or qualified
editorial readings. Do not count them as 95 ingested lexemes.

`record_prose_first_reading.py` regenerates the prose inventory and asserts
unique source-unit keys. `snapshot_scope.py` regenerates the five-page keyed
text selection and the nine-context whole-volume name search. Both ran
successfully. These checks validate recording mechanics, not linguistic accuracy.

The original four-page interlinear material (572–575) has been scope-inspected.
Both specimens on 572–575 now have reproducible first-reading aligned evidence in
`record_specimens_first_reading.py` and `specimens-first-reading.jsonl`, including
glossed speaker labels. Its count and page breakdown are generated in
`specimens-first-reading-status.json`: 537 aligned units across all four pages.
This is an extraction milestone, not audited completion. All first readings need independent reconciliation; no accepted-only
sample can substitute for full source-unit accounting.

The independent original prose review resolves two first-reading errors:
`īnun` → `inun` in the second-column pronoun object cell, and
`kakṭēn anlēn` → `kaktēn anlēn` in the done phrase. These are occurrence-specific
corrections from original glyph inspection, not harmonization. The documented
plural personal ending `r` adds one morphology witness. `reconcile_prose.py`
pins the independent report and both first-reading hashes, writes reviewed
derivatives, and rejects stale input; the first readings remain unchanged.
Preserve the author’s tentative e/o quantities and questioned `Iūn`
reading with explicit qualification rather than declaring them corrected forms.

See `ROOT_SCOPE_REVIEW.md` for original-page inspection, comparison-table
exclusions, later-addendum review, and the unresolved modern identity detail.
`scope-snapshot-status.json` describes the extraction operation only; the later
review document records completed scope checks beyond that snapshot.

Grayscale second reading now covers all 537 aligned units on printed 572–575
(60 lines). `grayscale-corrections.json` records 51 form corrections and one
gloss correction, with five explicitly retained glyph uncertainties.
`reconcile_grayscale.py` ran
successfully and emitted all 537 units to `specimens-second-reading.jsonl`,
preserving the first reading alongside each proposed correction. Every row still requires the
independent final-output review. These checks establish reproducible correction
application, not completed ingestion. The seven-paragraph free translation on
575 is separately preserved in `specimen-II-free-translation.json`; its English
wording must not be used to invent additional word-level gloss alignments.

The complete evidence inventory now contains 699 source units. The unfrozen
`build_proposal.py` review output accounts for every unit: 686 target units
produce 694 rows after splitting explicit prose alternates, five control-language
units are excluded, and eight contextual units remain documentary evidence
(seven free-translation paragraphs and one uncertain collective attribution).
No row is installed. Source spelling is retained, Phonemic stays blank, and no
etymological or other graph relationship is inferred. The 84 reviewed prose groups and
11 pronoun cells have explicit grammatical proposal sidecars.
`structure_specimens.py` now removes only terminal sentence delimiters and
speaker-label dashes, preserving the removed bytes in its audit; 113 of 537
forms change. It assigns 70 compatible, exact-form pronoun occurrences tags
backed by the source's pronoun table. An explicit interlinear `-to` selects
dative; bare English object translations do not resolve the stated syncretism.
An additional 95 occurrences now have explicit supporting prose locations on
570–571 for number, noun plurality, case, pronoun function, tense, negation,
imperatives, infinitives and participles. These use individually listed form
and translation pairs; person features do not propagate from prose examples.
Qualified relative-pronoun analysis remains uncertain. `specimen_context.py`
adds explicit form/gloss-pair decisions for 263 noun, conjunction, verb,
adjective and adverb occurrences after comparison with originals 572–575.
These are labelled editorial contextual POS analyses, not printed POS
abbreviations. The latest 151 additions preserve every form, gloss and entry
key, and do not infer tense, aspect, person or voice from English verbal
inflection. Nominal decisions likewise do not infer case, gender or number.
`specimen-context-review-20260927.json` records this review and its scope.
`independent-context-pos-review.json` independently checks the 151 new
contextual decisions against originals and identifies seven missing source
qualifications. These are now represented: `iūn` carries the author's questioned
reading as transcription uncertainty; the three r-final singular-translated
copula occurrences retain the author's competing plural analysis; three verbal
`kaktēn` contexts retain the qualified agency-noun alternative. Four occurrences
therefore move from contextual-only POS to explicit prose support.
`specimen_relations.py` explicitly enumerates genitive and locative tokens
whose endings and scope are supported by p570 and the original aligned
specimens. Their lexical glosses omit the grammatical English preposition,
while the complete original translation remains in the audit. No English
`-to` rule is applied to the other spatial expressions. The qualified source
analysis of `kaktēn` as a noun of agency remains uncertain; `bānēkil` and
`pōṭāl` receive noun POS without an invented case assignment.
All remaining 114 aligned tokens now have independently proposed, owner-reconciled
decisions in `reconcile_specimen_metadata.py`, which pins its input review hash
and checks exact source occurrences. Nine ambiguous contextual interpretations
and the conflict between `annē` “me” and the grammar table's “my” remain typed
uncertainties. Five explicitly -lōpul-marked expressions also receive the
source-supported locative analysis. Personal metadata follows each occurrence:
`āplē` is not assigned one person across its differently translated contexts.
All 537 aligned tokens now have decisions; this does not establish final-output
acceptance.
All punctuation and grammar proposals require independent review before a freeze.
`whole-review-proposal-status.json` records the proposal hash and remaining gates.

`prepare_metadata.py` now prepares a review-only profile, adjacent YAML settings,
and proposed historical Chanda dialect record; `proposed-reference.bib` records
the source and OCR-assisted provenance. The profile covers all 47 input graphemes,
lowercases display initials and maps velar nasal ṅ to ŋ while preserving the
remaining source notation, meaningful boundaries and qualified vowel quantities.
Original retains source spelling; Phonemic and Native remain blank. Two focused
tests pass for complete NFC/NFD coverage and actual generic-parser preservation
of all 694 keys and source spellings. These are proposal checks, not registry
installation or compiled verification.

The independent whole-source census confirmed the aligned-token coverage and
identified an expression-level gap. That gap is now reconciled: 51 utterances
and seven separately translated blocks are represented with exact source-token
spans. Six dialogue labels remain provenance. Specimen I and unpaired sentence
records have blank free glosses; their concatenated interlinear analyses remain
explicitly typed in the audit. All seven English paragraphs map only to their
complete source blocks. Root reviewed all four originals and recorded the
integration decision in `root-utterance-integration-review.json`. Token-level
uncertainties propagate into whole-expression audits and tags. Fresh final-output
audit remains required.

Three focused correction-layer tests pass: immutable first-reading hashes and
exact reviewed deltas; the added bound plural witness and reconciled output
counts; rejection of stale review inputs. No canonical installation, profile
certification, or full-output acceptance is implied by these focused checks.

Still required: independent specimen reconciliation and source-unit census,
independent source comparison, identity/dialect and bibliography reconciliation,
final sound-profile coverage after utterance recovery, conservative relationships, final frozen output audit,
focused validation, and explicit source-stage installation. Full builds and
database/browser verification are deferred under the user’s standing instructions.
