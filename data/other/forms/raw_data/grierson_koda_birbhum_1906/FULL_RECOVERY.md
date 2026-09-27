# Complete Koda source recovery

The proposal covers the whole Koda/Kora chapter of Grierson's original 1906
*Linguistic Survey of India*, volume IV, printed pp. 107–115, and all 241 prompts
of its Dhangar standard-list column (printed pp. 241, 245, 249, 253, 257, 261,
265, 269, 273). The introductory pages supply qualified contextual names and
other-language controls, not additional invented target-language translations.
The original DjVu page number is the printed Arabic page plus 19.

The 735 physical extraction units comprise 41 Birbhum grammar/prose examples,
333 Birbhum aligned specimen cells, 93 Bankura aligned specimen cells, 27
Dhangar/Bankura prose units, and 241 Dhangar table cells. They yield 654 forms:
256 Birbhum, 96 Bankura and 302 Dhangar. Alternatives increase the form count;
114 exact repeated units merge citations within the same lect. Fourteen bound
grammatical elements, one repeated untranslated fragment, five physical line
continuations and one explicit Mundari control remain individually accounted
for in the audit. The five continuations are joined with both locators; a
separate English gloss wrap is not mistaken for a lexical continuation.

The five pilot keys are preserved, with four source-reading corrections;
649 additional forms are proposed. Historical pilot evidence remains available.
Bankura is retained with Grierson's warning that the specimen was corrupt and
partly restored. Its parentheses are not expanded into invented alternatives;
the printed `sic` is a warning in Notes. Dhangar table items 103–104 retain their
complete abbreviated relational patterns without supplying missing stems.

Historical transcription is preserved literally, including stacked vowel marks,
underdots, apostrophes for semiconsonants and the raised vowel in table item 237.
The preservation profile covers 84 graphemes. Its `IPA` column is a converter
interface, not a phonemic claim: raw Original remains unchanged and Phonemic is
blank. Display-only house mappings are `w` to `v` and `ṅ` to `ŋ`.
Typed glyph uncertainty remains for Birbhum p. 111 line 4 word 6 and table item
209. Source-explicit grammatical categories and the attributed Aryan loanword
claim for numerals six through ten are retained; no donor lemma or graph edge
is inferred. Prose language names and ambiguous historical population labels do
not introduce invented coordinates or language identities.

The full proposal has no exact NFC-literal-form/casefolded-gloss matches against
6,462 other current source Koda rows (CFEL print, CFEL publisher data and Zoller).
This comparison does not establish source independence from spelling alone;
separate historical attestations remain separate unless dependence is demonstrated.

Validation evidence is in `full-preview-validation-20260926.json`, the immutable
independent audit reports, `legacy-identity-reconciliation-20260926.json`, and
`whole-source-overlap-review-20260926.json`. Pass 1 inspected 20 sampled units
and four additional edge cases against original pages: the literal readings
passed, but metadata omission classes required correction. The fresh pass 2
passed all20sampled units with zero material errors;654forms are source-stage installed. Focused tests cover source-unit accounting,
stable keys, physical continuations, abbreviated patterns, source warnings,
grammatical tags, complete profile coverage and a scoped 654-row parser check.

`prepare_full_preview.py` generates the proposal only. Once reviewed,
`import_source_full.py --install` reproduces and installs the source CSV without
building a database. The full CLDF pipeline, global compiled references/IDs/graph
checks, full suite, browser database and representative app entries remain
deferred under the user's explicit no-build/no-remotes instruction. This source
must not be described as full-pipeline complete while those gates remain deferred.
