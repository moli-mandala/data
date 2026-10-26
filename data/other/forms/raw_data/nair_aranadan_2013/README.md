# Nair 2013: complete Aranadan chapter

Ravi Sankar S. Nair, *Tribal Languages of Kerala*, Language in India 13(7),
July 2013, Aranadan chapter pp.9–11. The [publisher's article page](https://languageinindia.com/july2013/ravisankarkeralatriballanguages.html)
provides the source. The local original is
`tmp/pdfs/tribal-languages-kerala-nair2013.pdf`, 58 pages, SHA256
`4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af`.
No explicit open reuse licence has been established. The import contains
attributed lexical facts and analyses; the full original PDF remains outside
canonical source assets. Native positioned text and original images were used,
without OCR. Nair is the chapter author; Usha 1992 is reported bibliography,
not an independently collated source or an asserted source for every token.

## Coverage and representation

The complete source-defined chapter and a whole-volume search for additional
attributed occurrences are inventoried in `scope/`. This is not a claim that all
other chapters of the multilingual book have been ingested. The complete
first reading and independent original reading cover 97 editorial source units
and 105 physical witness slots, including comparisons, ten whole translated
expressions, all 32 verb-table cells, explicitly printed morphology and six
final vocabulary items.

The source files installed on 2026-09-27 contain 102 rows: 81 Aranadan, 20 Malayalam and one
Kannada. Four community-name occurrences remain documentary because their
lexical-language attribution is not established. Twelve other units are
documentary prose or notation contexts. One explicitly shared future suffix
adds a Malayalam witness under its existing physical source unit, without
inventing a second printed token. Two explicitly supported j/y variant pairs
have directed variant links, with the second listed form pointing to the first
as a display convention, not a claim of historical priority. Ordinary tense
inflection, broad influence claims and conjectural name derivation do not become
derivation or borrowing edges.

Existing language IDs are reused. Distribution places are not assigned as
elicitation sites; no dialects or coordinates are invented. Prior numeral-source
rows and their keys remain untouched. Stable source/witness keys retain chapter,
section and position independently of corrections to form or analysis.

## Transcription

The source has Roman transcription, including private-use font codes for lower
dots and nasal tildes and drawn low lines absent from extracted text. Independent
image review recovers those marks as Unicode; underlined n/l remain distinct
from dotted letters. Original case, doubled vowels and affix boundaries remain.
The source does not provide a complete phonemic key, so `convert: false` preserves
the reviewed source layer; Native and Phonemic stay blank. No PUA characters,
replacement glyphs, guessed IPA values or automatic slash expansions may enter
the final output. See `root-review/transcription-and-provenance.json`.

## Reproduction and acceptance

`first-reading/prepare_first_reading.py` records the complete first reading.
`root-review/prepare_approved_overlay.py` freezes the independent metadata and
scope reconciliation. `integration/prepare_review_candidate.py --overlay
<absolute-path-to-root-review/approved-reconciliation.json>` produces the review
candidate. Historical review artifacts remain unaccepted snapshots; they are
not silently rewritten into proof of acceptance.

After independent output and original-page checks, the root source-acceptance
record permits `integration/prepare_accepted_audit.py` to produce a separate
accepted audit. `import_source.py` defaults to read-only preflight and requires
pinned installation approval; `--install` writes the CSV, YAML, accepted audit
and bibliography additions through a recoverable transaction. Installation is
proved only by a matching `installation-receipt.json`. Until it exists and
verifies, the source remains a candidate. The installed receipt now verifies;
a repeat invocation found all approved inputs, outputs and durable identity
state unchanged. Five installed-source tests passed, alongside 20 isolated
installer tests, all-row NFC/NFD parser checks, exhaustive independent output
validation and a fresh 20-entry original-page audit. See `source-stage-handoff.json`.

Full CLDF/database generation, compiled graph/identity checks, the full test
suite, browser refresh/representative app inspection and publication are
deferred under the user's instructions. Source-stage validation does not prove
those downstream gates.
