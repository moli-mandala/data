# Whole-source Kāgānī integration, 2026-09-27

This is an unfrozen source-stage proposal, not an installed ingestion. The
mandatory source checklist and survey/comparative addendum remain active.

`prepare_integration_scaffold.py` reproduces `integration-working/` from the
legacy glossary, owner grammar/notes, independently reviewed introduction,
assigned grammar and reviewed narrative files. It never writes canonical data.
The manifest pins inputs and outputs. All 329 existing glossary keys and rows are
currently retained; an independently reviewed keyed metadata correction may
replace selected metadata, but never silently changes keys or source forms.

Pages 87 and 91 are explicitly accounted as source context, including the
separate Bahramgala heading. Every other source unit is either emitted or remains
an audit control with a reason. The 362 parable atoms, six explicit parentheticals
and 16 contiguous expression spans are retained; the spans partition the primary
atoms exactly. The separate page105 story retains all34 atoms and three complete
expression spans. Whole expressions use the printed interlinear gloss, without
inventing a free translation or copying a line break as a sentence boundary.

The first-reading introduction files are immutable evidence. The root's
61-unit review drives a SHA-bound derivative: plain italic l remains plain in
phonetic context, and the Panjabi chăḍḍṇā comparison is not a Kāgānī head. Four
causative and two habitual vowel readings were independently adjudicated as
breve ă; the printed kărna versus kărnā distinction remains.

Explicit English interlinear constructions are parsed by
`interlinear_metadata.py` into lexical glosses and grammar tags. The exact raw
English stays in the audit and Notes where normalized. This does not segment
Kāgānī forms internally or infer a donor language. Ambiguous English past versus
participle translations are not forced into a unique tense. Whole-expression
glosses remain complete and unnormalized.

Tone pairs are alternate written representations supplied by Bailey. Their
shared source unit and exact source claim are preserved; this alone does not
assert a lexical variant or derivation edge. Parenthetical sheep/pigs and
God/prophet remain distinct meanings. No graph edge is inferred from matching
strings. The source's vertical tone strokes are retained literally as U030D and
U0329, not reinterpreted as IPA stress or syllabicity.

`validate_integration_parser.py` uses an isolated tiny source tree and unmodified
actual parser code with intended input=form and preserve_hyphens=true settings.
It verifies NFC and NFD originals, keys, Notes, citations, tags and normalized
literal display. Existing w→v and ṅ→ŋ house conventions are preserved; new stacked
mark aliases preserve the same output across Unicode normalization. Source-facing
Notes omit crop/reviewer process descriptions, which remain in the audit.

Outstanding: independently reviewed legacy metadata delta, source-wide final
metadata/reference reconciliation, profile-policy acceptance, fresh stratified
actual-output audit and guarded reproducible installation. No database/full CLDF
build, complete test suite, browser QA, commit, push or release has been run.
Those full integration gates remain explicitly deferred by task scope.
