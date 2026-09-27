# Dhakal (2011), The Darai Language — chapter 3 acquisition

Selected for sparse Indo-Aryan coverage: Darai has 32 compiled/manual source
rows at selection. This is Dubi Nanda Dhakal's Tribhuvan University PhD
thesis, not the Darai-titled 2015 survey whose body and appendix describe
Danuwar. That rejected source is documented separately under
`source_checklists/evidence/darai-survey-mismatch/`.

Catalogue: https://elibrary.tucl.edu.np/items/4c2a3db9-8d61-4dc4-8906-60db09180a1e/full
Download: https://elibrary.tucl.edu.np/bitstreams/377ce7b5-e2b3-4a13-97b7-13b3de36e595/download
Year 2011 is supplied by the university catalogue. The complete 486-page PDF
is pinned by SHA256 in snapshot.py and kept in untracked workspace scratch.
No open redistribution licence is asserted. Intended import is attributed
lexical facts; no full PDF is checked in.

## Scope

Complete chapter 3, printed pp.42–76 (PDF pp.64–98), covering phonology and
morphophonology. Include its explicit Darai form/gloss attestations across
prose, numbered examples and tables; account separately for phoneme labels,
empty distribution cells, formant measurements and sentence-length examples.
This defines a chapter-level lexical extraction, not a complete thesis lexicon.
Other chapters and appendices are outside this package's lexical scope.

The chapter states that its description is based mostly on Pidrahani village
speakers in Chitwan. Do not assign every example to that village automatically:
review explicit consultants, source comparisons and other dialect labels first.
Use existing base Language_ID Darai; its current fallback clade Other needs
review in light of the thesis's Indo-Aryan classification. Do not invent a
new base language or a coordinate for a consultant.

## Evidence and decoding

`snapshot.py` asserts the exact PDF hash, 486 pages and chapter boundaries.
It stores all 40,425 glyphs from all 35 chapter pages, with original PDF order,
font, size and bounding boxes, plus ordinary extracted text and per-file hashes.
No OCR is used: native font decoding is available. The glossary/comparative
addenda apply; OCR-heavy addendum is inapplicable here.

`decode.py` maps only SILDoulosIPA93 font glyphs: U+F0AB → ə, U+F048 → ʰ,
U+F067 → ɡ, U+F029 → combining tilde, U+F04E → ŋ. A Times New Roman U+F020
is a space. Unknown private-use glyphs fail loudly. Decoding is distinct from
house transcription; no sound profile has yet been chosen. In particular,
source superscript h is not silently reinterpreted as a breathiness symbol.

Printed pp.46 and 53 were visually inspected at 150 dpi. On p.46, ordinary
positional text extraction produces misplaced nasal marks in the forms for
'eye' and 'short'. Original glyph order correctly yields ãkʰi and hõco,
matching the rendered page. Preserve this order rather than moving combining
marks with broad substitutions. Source retroflex underdots are preserved and
NFC-normalized. Source-internal gloss variation also needs preservation:
geutʰəli is glossed 'skylark' on p.47 and 'sparrow' on p.53; do not silently
harmonize them.

## Reproduction and status

From data/:

```
.venv/bin/python data/other/forms/raw_data/dhakal_darai_2011/snapshot.py
.venv/bin/python data/other/forms/raw_data/dhakal_darai_2011/decode.py
.venv/bin/python -m pytest tests/test_dhakal_darai_2011.py -q
```

Three acquisition/decoding tests pass, covering all page hashes, all native
private-use glyphs, wrong-font rejection and the confirmed nasal-order errors.
These do not constitute a lexical acceptance audit. No source rows installed.

Pending: complete per-record lexical inventory and exclusions, extraction of
forms/glosses/grammar/source relations, transcription profile and dialect
review, citations and auxiliary references, seeded 0/20 acceptance audit,
canonical installation and focused integration checks. Full pipeline, full
suite, compiled survival, IDs/deduplication/graph/references/concepts and
error/generated-diff verification remain required on an authorized runner.
Browser database and app QA are not applicable without a requested refresh.


## Quoted-span inventory and contextual triage

`inventory.py` reproduces `inventory.json` from the frozen glyph stream: 495
single-quoted spans, keyed by printed page and immutable PDF glyph position,
with original text, font-decoded candidates, bounding boxes and surrounding
context. It handles mixed straight/curly quote delimiters without splitting
possessives or contractions. Tables retain repeated attestations and different
source glosses rather than deduplicating prematurely.

444 spans have adjacent slash-delimited form candidates. The remaining 51
have explicit `context-dispositions.json` entries: 32 bare-form candidates,
18 interlinear sentence examples still requiring scope review, and one English
metalinguistic term ('cluster') excluded. `inventory-exceptions.json` preserves
the p.43 caralə / graze-PST example whose opening gloss quote is missing.
Thus 477 potential lexical occurrences are currently accounted for, not 477
accepted rows. Further prose-only forms with shared glosses (notably jətka on
p.56) and structural completeness still need review. The inventory is not yet
a complete per-record lexical audit or source-to-output acceptance audit.

Six focused acquisition/inventory checks pass. No importer output, canonical
metadata, sound profile, graph edges or installed rows exist for this source.
Reproduce the inventory with:

```
.venv/bin/python data/other/forms/raw_data/dhakal_darai_2011/inventory.py
```


## Context review and current proposal

`proposal.py` now reproduces **485 lexical candidates** from the 495 quoted
spans plus seven explicitly anchored prose additions. There are 17 exclusions:
16 multiword interlinear clauses and the English metalinguistic word 'cluster'.
Two single-word negative inflections in p.74 examples 67c–d are retained with
source person/number and negation tags, not discarded as sentence examples.
Their complete source translations remain in the audit while the proposed
lexical glosses are 'not kill' and 'not do'.

Visual review of printed pp.43, 56, 73–75 establishes the malformed quote,
shared-gloss jətka, past -lə, non-past -tə and its variants -tahə/-t, and
causative -a. All seven additions have exact text anchors in frozen evidence.
`reviewed-relations.json` records four variant claims: jətka → etka,
wətka → otka, -tahə → -tə, and -t → -tə. Source-local targets are explicit;
no ancestry is inferred. `borrowing-dispositions.json` records the author's
Nepali-loan attribution for sriman and krija; donor forms are unspecified,
so only source prose and loanword tags are proposed.

Seven focused checks pass. Reproduction:

```
.venv/bin/python data/other/forms/raw_data/dhakal_darai_2011/proposal.py
```

This supersedes the earlier contextual triage counts, not the frozen inventory.
The proposal is **not installed**. Printed grammatical labels, source spacing
and diacritics, metadata/profile decisions, auxiliary citations, full chapter
completeness and fresh source-to-output acceptance audit remain open. Existing
full build/full suite/compiled identity, graph, deduplication and reference
gates also remain open; browser refresh is unrequested.


## Grammar and diacritic review

`grammar.py` now separates the chapter's printed labels from lexical glosses,
using exact source labels and preserving ordinary hyphenated English. It
structures imperative, absolutive, prospective, numeral classifier, high
honorific, gender, conditional particle, onomatopoeia, person-number and the
explicitly described suffixes. 30 candidate rows carry tags; four grammar-only
glosses are blank while their exact source text remains in `source_gloss`.
No POS, tense or inflection is inferred merely from an English translation.

The shared registries (`data/tags.py` and frontend `src/lib/tags.ts`) now include
prospective, non-past, classifier and high-honorific, with frontend labels.
The source distinguishes these categories: pp.112–113 establish high honorific,
p.129 numeral classifiers, p.203 prospective non-finite forms, and p.74 non-past.
Context pages and hashes are recorded in `grammar-evidence/` and
`grammar-review.json`; no lexical scope was expanded to those other chapters.

`form-corrections.json` documents three visually verified combining-dot repairs
on pp.59 and 68. The original extracted strings remain in proposal records as
`extracted_form`, while corrected Unicode reflects the displayed consonants
(cʰəṭ.pə.ṭi, ceḍ.nəi.ke, pəṭ.ka). This is glyph-placement repair, not an
unrecorded phonological emendation. All 16 focused Darai and shared form-grammar
checks pass. Counts remain 485 candidates and 17 exclusions. No source rows
installed; profile/metadata, remaining full-chapter visual/structural review,
auxiliary references and the seeded acceptance audit remain pending, followed
by full integration gates.


## Transcription preview

`conversion/dhakal-darai.txt` now covers all 485 proposed forms. Printed
pp.43–49 distinguish ə/a by quality and explicitly deny distinctive vowel
length, so the source-specific house policy preserves both vowels without
inventing macrons. Table 3.7 on p.51 places c/dz among alveolar affricates
and j/w among glides: these become ʦ/ʣ and y/v; ɡ becomes g. Source superscript
h and nasalization remain intact. Syllable periods disappear only in house
forms; initial/final morpheme hyphens remain.

`transcription.py` reproduces `transcription-preview.csv` with literal decoded
Original, proposed Form, source and lexical glosses, grammar, source keys,
page references, variant targets and loan attribution. Visual review of
pp.58/61 supports two scoped word-internal spacing normalizations in the
house form only; the reduplicated p.58 expression retains its word space.
The source transcription is not asserted to be a standardized IPA column.

`transcription-review.json` pins the preview/profile hashes and decisions.
All 17 focused Darai/shared grammar checks pass, including every candidate,
NFC/NFD nasal equivalence, no invented length, preserved affricate/glide
contrasts, literal originals, and compatibility with shared house policy.
This remains a review preview: no canonical Darai rows are installed. Whole
chapter completeness, metadata and reference review, fresh seeded acceptance
audit and all previously recorded full integration gates remain pending.


## Source installation and acceptance (supersedes preview status above)

All 35 chapter pages were read and visually inspected. `completeness-review.json`
records expected lexical counts and nonlexical exclusions page by page. The
495 quoted records plus seven explicit prose additions yield 485 installed
rows and 17 exclusions (16 multiword clauses and one metalinguistic term).
`import_source.py` produces the complete 502-record audit and deterministic
15-column CSV. Its `--install` mode is gated by the reviewed output and metadata
hashes. Seed 2026092107 produced **0/20 material errors**; the exact sampled
records, raw context and decisions are checked in. Repeated installation passes.

Installed paths: `data/other/forms/20260921-dhakal-darai.csv` and matching YAML.
All rows belong to Darai and registered Chitwan dialect; six acoustic examples
on p.49 additionally have the explicit Pidrahani site. Som Lal Darai and the
2008 recording are provenance, not dialects. New dialect coordinates are blank.
Darai's base clade is corrected from Other to existing Bihari using Glottolog's
classification; its quality-C modern coordinate stays unchanged.

Original preserves the decoded source transcription; Phonemic preserves the
author's phonemic analysis and syllable notation, with only the two explicitly
reviewed internal spaces normalized. The profile converts that field to house
Form. It does not claim that the author's symbols are standardized modern IPA.
Six rows carry typed uncertainty: five occurrences in two conflicting-gloss
groups and one unusual printed dʰz sequence. No correction to their scholarly
interpretation is made. Four variant edges are explicit; two Nepalese-loan
attributions remain prose with loanword tags because donor forms are absent.

Seven bibliography entries were installed; existing LSI is reused.
`reference-review.json` records every auxiliary citation and the thesis's
inconsistent Kotapish years. Two bare 1973 references remain bibliographically
ambiguous between the phonemic summary and glossary; no lexical row is
attributed to either auxiliary work. Main lexical citations all resolve to the
university thesis with exact printed pages; the two single-word discourse
examples also retain text IDs in their citation locators.

Validation: 22 focused source/grammar/transcription-input tests and 20 shared
sound-profile/dialect tests pass. Actual `parse_file` emits all 485 rows with
zero conversion errors, exact reviewed house forms, original/phonemic fields,
and variant targets. All seven references format with Pybtex. Settings validate
as 199 files / 196 citation keys. The compiled-source gate remains **0/485**
until the required full build runs; `test_compiled_darai_source_survival`
checks this explicitly. Full pipeline, full suite, compiled IDs/graph/dedup/
reference/concept checks and generated-diff review remain required. No browser
refresh was requested, so browser build and app QA are inapplicable for now.

Reproduce from data/:

```
.venv/bin/python data/other/forms/raw_data/dhakal_darai_2011/import_source.py
.venv/bin/python data/other/forms/raw_data/dhakal_darai_2011/import_source.py --install
.venv/bin/python -m pytest tests/test_dhakal_darai_2011.py -k 'not compiled' -q
```
