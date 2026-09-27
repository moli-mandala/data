# Bhat1971 Belari appendix — preparation

The ingestion checklist is active with glossary and comparative-table addenda.
This source supplements the existing bhat1971koraga record with its Belari appendix,
printed119–123/PDF126–130. The existing Koraga importer covers printed88–118 only.
The full PDF is already cached locally and pinned in source-manifest.json.

Existing108 Lindgren Belari rows derive from this source (thesis printed27/Table4);
the thesis describes114 comparative concepts. Do not treat the appendix as an
independent attestation or assume that the six-count difference is six missing words.
Examples, paradigm cells, comparison languages, variants and concept scope require
separate review. Existing derivative rows remain unchanged.

page-scaffold.jsonl preserves existing PDF extraction for five pages, explicitly
unreviewed. First-page rendering shows OCR-corrupt phonetic symbols. No lexical
rows, profile, registry edits or installation are produced at this stage. See the
manifest for pending source review and user-deferred build gates.

## Structural review and first transcription

All five pages are structurally reviewed. section-inventory.json accounts for191
printed target form occurrences: initial vocabulary, comparisons, sound examples,
personal suffixes, a root and its paradigm, other finite examples, pronouns/deictics,
and locative/plural suffixes. Repeated forms count as source occurrences. Seven
Tulu comparison-table cells are controls. Historical suffix discussion is analysis,
not invented lexical forms; borrowing is explicitly tentative in the author's prose.

lexical-candidates.jsonl contains36 first-pass readings for printed119 (sections2a/b),
with section/column/row keys. They are not accepted output. Barred-i/retroflex details
need a second visual pass, especially sigɨṇi and saṭrɨ. The give-to-person restriction
belongs to lexical meaning and must not become verb-person agreement tags. Keep j
and y distinct in this source; do not import the Pattapu IPA-glide rule blindly.

## Complete first pass

All191 inventoried target occurrences now have first-pass candidates with page,
section, column and row keys. Each retains its printed gloss and paradigm labels
where applicable. These are not yet accepted lexical rows. The next pass must
check barred vowels independently, especially suffix tables and italic pronouns.
Printed brave/bravo are retained, not regularized to barve/barvo. The food-list
chilly gloss is preserved with a semantic review flag. Shared forms in distinct
gender/number cells remain separate occurrences. Two preparation tests verify
complete section topology, stable keys and these source distinctions.

## Second glyph pass and grammar proposals

43 occurrences have second-pass glyph records. One correction changes first-pass
battigo to battɨgo in the neuter plural past cell; lexical-candidates.jsonl remains
unchanged so the correction is auditable. Enlargements confirm sigɨṇi and saṭrɨ.
The personal-suffix table's ordinary-i plural forms remain literal despite barred
vowels in related paradigms; do not impose phonological uniformity.

prepare_analysis.py generates192 proposed analyses for191 source occurrences.
The explicit female-singular/plural you yields two child analyses. Gender-number
labels attach to their cells, not to every form on a table row. Source concessive
and assertive headings remain notes rather than forced canonical mood labels.
The since-you-came example retains its temporal gloss without acquiring a
conditional tag solely from its section heading. Glyph review is still incomplete
(148 occurrences pending), and no proposal is installable yet. Four focused tests
pass, including canonical tag validation and regeneration of the analysis ledger.

## Second glyph review complete

All191 occurrences now have second-pass records; ten first-pass corrections are
retained separately. Three readings remain explicitly uncertain: the washed-out
first-person pronoun n/ṇ distinction; the first vowel in italic battiḍɨ; and a dot
printed between ay and final italic i in ay.i. Do not normalize that dot into vowel
length or silently remove it. The food-list chilly gloss supplies a fourth typed
uncertainty. All192 analysis proposals now retain review status and uncertainty
tags. Four focused tests pass(0.07s). Sound profile, derivative reconciliation,
source-row emission and acceptance audit remain before installation.

## Derivative and paradigm reconciliation

reconcile_lindgren.py reproduces a reviewed108-row ledger:89 ordinary shared-source
correspondences, seven derived stems, three entries not located in the appendix,
two grammatical disagreements, two transcription differences, one gloss ambiguity,
two recipient-restriction generalizations and two unsupported inclusive/exclusive
analyses. Coast, hand and tree remain unlocated; no primary locator is invented.
Female-singular and plural you map to their correct child analyses. Existing
Lindgren rows are unchanged at this stage.

relationship-decisions.json records45 proposed inflectional variant links to the
explicitly attested root bar in section7. This does not create historical ancestry.
Seven Tulu comparisons remain controls; contact ambiguity and proto-sound discussion
remain commentary. Repeated gaḷde occurrences retain their distinct locators without
being counted as independent field evidence. Six focused tests pass(0.08s).

## Draft source emission and Unicode checks

Prepared 192 draft rows from 191 reviewed source occurrences, preserving 45 explicit
paradigm links, four uncertainty flags and 14 historical-commentary notes without
inventing historical graph edges. The source-local profile preserves the printed
ay.i dot and distinguishes affricate j from glide y. A focused test exposed missing
decomposed-diacritic coverage; explicit NFD spellings now produce identical output
to NFC for every draft form. Seven focused tests pass (0.20s); the source-local
profile policy audit reports no violations. The row parser accepts all 192 rows
and preserves Original, Gloss, Notes, Etymology, Variant_Of_Key and Tags.

The seed-2026092203 sample is prepared but its acceptance audit remains pending.
Draft CSV and parser output are in workspace tmp only; canonical source installation
and derivative annotations remain pending. No remote work or database/data build.

## Source-output acceptance and source-file integration

The preceding goal turn made progress: 192 draft rows, Unicode coverage and parser
checks. Seed 2026092203 now has a completed 20-row visual source-output audit with
zero material errors (output-audit-2026092203.json). All five printed pages were
consulted; spelling, glosses, grammar, locators and parent relationships checked.

Installed 192 rows and their source-local profile/settings. Reused bhat1971koraga
without duplicating its registry entry; its existing owner routes Belari first and
retains the Koraga fallback. Extended the canonical bibliography's inclusion and
provenance fields to cover appendix8. No new dialect/location is asserted: the
source has no named village suitable for a more specific attestation.

Eight focused tests pass (0.49s), including parsing the installed CSV with global
metadata, preserving all reviewed fields and checking the shared reference. Source
metadata validation passes (203 files,198 keys). The rows retain four uncertainties,
45 explicit paradigm links,14 historical-commentary entries and no inferred
historical edges. Seven Tulu control cells remain excluded. Lindgren annotations
remain pending; these sources share provenance, not independent field evidence.
Full pipeline/graph/reference output, full-suite and browser checks remain deferred
under the user's no-build constraint. No remote execution or database build.

### Belari derivative annotation integration

The prior goal turn made progress by installing 192 source rows and completing the
source-output audit. Applied 19 reviewed comparison notes to existing Lindgren
Belari records, with seven typed uncertainties. Seven derived stems, three
unlocated entries and two recipient-scope generalizations receive provenance notes
without automatically becoming transcription errors. Original forms, glosses,
phonemic fields, citations, cognate claims and keys are unchanged; no primary
locator is fabricated for coast, hand or tree. The source importer loads both
Pattapu and Belari annotation sidecars and reproduces installed CSVs exactly.
Eleven focused IA/Dravidian tests pass (compiled test excluded), and the added
Belari distinction test passes separately. Full data/database and browser gates
remain explicitly deferred. No remote execution or database build.
