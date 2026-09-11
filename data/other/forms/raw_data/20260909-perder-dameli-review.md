# Perder (2013), Dameli — source ingestion review

Reviewed 2026-09-09 under [SOURCE_INGESTION_CHECKLIST.md](../../../../SOURCE_INGESTION_CHECKLIST.md).
Applicable addenda: dictionary/glossary, comparative tables, etymological/comparative source.
OCR and website/API/CLDF addenda are inapplicable: the source is a born-digital PDF with usable text.

## Source and extraction

Emil Perder. 2013. *A Grammatical Description of Dameli*. PhD thesis, Stockholm University.
ISBN 978-91-7447-770-2; [official DiVA record](https://urn.kb.se/resolve?urn=urn:nbn:se:su:diva-93888).
The [official 242-page PDF](https://su.diva-portal.org/smash/get/diva2:651418/FULLTEXT02.pdf)
was acquired on 2026-09-09. Its SHA-256 is
`6d740b309f86534157ea8e8c741e2bcc27ac464ac9e9c719e530c3193219fe1c`;
the SHA-512 in the manifest matches the checksum published by DiVA.
Physical page = printed page + 24 throughout the lexical scope.
The PDF is not redistributed. It is copyrighted and has no explicit reuse licence;
the repository contains extracted linguistic facts and sufficient context to audit them.

Following the Knobloch Sauji precedent, `perder_dameli_2013.py` reads positioned words,
font distinctions, explicit table columns and x-aligned interlinear tiers. No OCR is used.
The pinned JSONL snapshot supports offline rebuilds. The explicit curation layer repairs
font splits and displaced diacritics, scopes prose definitions, and records source-side
ambiguity without silently resolving it.

## Coverage and accounting

The source census covers glossed Dameli forms in running prose, lexical/phonological/
grammatical tables, numbered examples 1–177, the complete verb-root appendix on printed
pp. 207–208 and Appendix 2 on pp. 210–216. Example 38 is recovered as a full segmented
word from prose because the schematic layout is unsuitable for automatic tier alignment.
All table continuations were reviewed, including kinship Table 17 through p. 69,
the merged gender cells of Table 27, and all 25 conjuncts with their printed components
in Table 39. Table 48 contains exactly 152 root records. Per-region counts are in the manifest.

Tables 1–3, 33, 37 and 42 contain abbreviations, transcription conventions, corpus metadata,
affixes or grammatical schemas rather than independently glossed lexical records. Phoneme
inventories and the kinship diagrams are excluded; the more extensive kinship table is retained.
Free translations and bibliography are not lexical records.

| Stage | Count |
| --- | ---: |
| Frozen extracted units | 3,386 |
| Explicit child records from six joint citations | 12 |
| Total audited units | 3,398 |
| Installed without curation repair | 2,801 |
| Installed after explicit repair | 75 |
| Audit-only exclusions | 522 |
| Forms after expanding complete alternatives | 2,900 |
| Identical repeated analyses collapsed | 1,044 |
| Installed lexical records | **1,856** |

The 522 exclusions comprise 296 unglossed/repeated/metalinguistic fragments, 165 isolated
affixes or phonological templates, 17 non-Dameli/comparison-notation units and one secondary spelling retained in notes, 16 kinship-diagram
units, 14 zero/punctuation/omission markers, six joint citations replaced by explicit children,
two unintelligible units, two rejected pronunciations, two English translation fragments and
one rejected hypothetical plural. Every exclusion has its original context and reason.

The rich CSV has all 15 fields. Stable keys encode printed page, table/example, position
and cell/word, with explicit child/variant suffixes. Repeated attestations collapse only when
form, lexical meaning, grammatical analysis, dialect, phonemic form, notes, etymology and graph
claims agree. The audit records the original key and canonical survivor and contains the
complete parsed records. The CLDF route preserves all surviving source keys, including homonyms
and identical shapes with different grammatical functions.

## Transcription, grammar, language and references

`Original` preserves the source's Standard Orientalist transcription. `Phonemic` contains
only separately printed bracketed IPA. The 75-rule `perder-dameli` profile produces the house
transcription in `Form`, with complete corpus coverage and no introduced replacement character.
Mappings include č→c, ċ→ʦ, c̣→ʦ̣, š→ś, ǰ→j, ɡ→g, w→v and ẉ→ɻ. The source distinction
between ẉ and ṛ is retained. Length becomes macrons; tone, nasality, u/o, ŋ, æ, syllable dots,
stress, morphology boundaries and zero markers are preserved. `−` becomes `-` and ∅ becomes Ø.
Content-stream order and explicit reviewed corrections repair the misplaced underdot on c;
the caron in žǎn ‘watermill’ was checked against the rendered page.

Lexical definitions are separated from grammatical labels. Source abbreviations supply
canonical tags for person, number, gender, case, valency, TAM, participles and particles;
new category labels are mirrored in the frontend registry. Question clitics add `interr`
without changing the host word's part of speech. Valency and compound components remain scoped
to their printed cells. Table 47's repeated kya label is interpreted using identical Table 41,
with the source typo and repair explicitly audited.

All forms use canonical Dameli `Dm`, Glottocode `dame1241`, clade Kunar.
Eight source-explicit Aspar records have `dialect:Dm:perder2013-Aspar:Aspar`.
Aspar is registered at 35.36798, 71.70814, Domel/Damel Valley, Chitral, Pakistan, with locality
evidence from [the GeoNames-sourced gazetteer entry](https://mapcarta.com/15180148)
(GeoNames 1417235), disambiguated from the northern Aspar. Consultant identities and elicitation
sessions are provenance, not invented dialects. Other Dameli forms remain unassigned to a locality.

`perder2013dameli` has complete publication, acquisition, editor, OCR and mapped-source
etymology metadata. Secondary references `morgenstierne1942dameli` and `cacopardo2008dameli`
are catalogued for explicitly discussed attestations and are clearly marked as not independently
ingested. Printed pages, table/example numbers, positions and available data IDs are retained
in the citation/audit. Bibliographic abbreviations are not converted into grammar or donor nodes.

## Graph and unresolved source claims

The input has 76 records with 113 source-explicit derivation/component links, 27 variant
records and one uniquely resolved external etymon: ištrii ‘wife’ on p. 34 → CDIAL 13734
strī́ ‘woman, wife’. Named or tentative loan donors have source-qualified prose and loan tags
where supported; no donor node is guessed. Tentative vowel strengthening in Table 30 does
not become an asserted derivation. Unetymologised forms remain first-class unlinked nodes.

Twenty-five installed source units carry explicit uncertainty reasons. These include questioned
glosses, tentative donor/compound interpretations, three plants with no taxonomic identification,
example 37's conflicting gloss and translation, and Morgenstierne's historical a'zâr numeral,
which Perder's consultants did not recognize and whose interpretation Perder questions.
The four printed conjunct complements čalii, pui, rawan and široo have no independent lexical
gloss in the source: their full expressions are glossed, but no component translation is invented.
There are no unresolved installed source glyphs.

## Visual audit and reproduction

Round 1 sampled 20 records from 1,859 with seed `3662392299860923243`. One material
grammar error was found: the Q clitic wrongly made the whole copular word a particle.
The parser was fixed and the error class has a regression test.
Fresh round 2 sampled 20 records from the final 1,856 with seed `766152442941636403`:
**20 passed, 0 material errors**, checked against rendered PDF crops. All three verdict ledgers
are checked in. After restoring the mixed-language p. 30 finger citation and retaining Cacopardo’s alternative spelling verbatim in notes, round 3 (seed `7698818706101321111`) freshly checked another 20 records with **0 material errors**. First/last appendix records, table page breaks, merged cells, rare glyphs,
source exclusions, homographs, optional forms and explicit graph claims received additional review.

Run from the `data` repository:

```sh
.venv/bin/python data/other/forms/raw_data/perder_dameli_2013.py --write
.venv/bin/python data/other/forms/raw_data/perder_dameli_2013.py --sample 7698818706101321111 --output /tmp/perder-fresh-sample.csv
.venv/bin/python -m pytest -q tests/test_perder_dameli.py tests/test_sound_profiles.py tests/test_dialects.py
make all
.venv/bin/python audit_source_ingestions.py
.venv/bin/python -m pytest -q
```

The sample command requires a new output path and preserves the reviewed verdicts.
Re-extraction requires the exact pinned PDF and `--pdf PATH --output NEW_PATH`;
ordinary rebuilding needs only the checked snapshot and curation JSON.

## Validation and handoff

All **11 source-specific tests pass against the final compiled CLDF**, including exact survival
of every source key, Original/Phonemic separation, grammatical tags, all citation locators,
and every intended graph endpoint/type/rank. The final graph has 74 component, 39 derived,
27 variant and one reflex edge from Perder records. `errors.txt` is empty. A second offline
importer run produces identical CSV, audit and manifest hashes. Focused diff checks are clean;
generated CSVs retain the repository's existing CRLF convention.

Every data-generation stage in `make all` completes through reference generation. Its final
manual-survey test target fails two checks outside Perder: the expected Rajasthani link count
and duplicate source-owned etymologies in the existing assignment overlay. The initial checkout
already failed the duplicate-overlay test and both global dialect metadata tests; pending and
concurrent user edits to etymology assignments and their ID handler are preserved. The complete
test-suite result is recorded separately below; the full-pipeline checklist gate remains open.

| Generated artifact | Before | After | Delta |
| --- | ---: | ---: | ---: |
| Forms | 677,797 | 679,653 | +1,856 |
| Edges | 357,003 | 357,586 | +583 |
| Source keys | 327,680 | 329,536 | +1,856 |
| ID aliases | 1,024,289 | 1,026,151 | +1,862 |
| Identity ledger | 847,558 | 849,415 | +1,857 |
| Concepts | 3,260 | 3,261 | +1 |
| Form–concept links | 500,962 | 502,769 | +1,807 |
| Alignments | 1,978,492 | 1,980,937 | +2,445 |
| References | 523 | 526 | +3 |

All **677,797 existing form IDs survive**; exactly 1,856 new IDs carry Perder citations.
Dameli increases from 1,184 to 3,040 records. The 141 new Perder edges account for the source
change; the other 442 net new edges and 442 existing-node `Status` changes result from compiling
the existing/concurrent manual-etymology work. Three of those existing nodes are Decker Dameli
forms; their spellings, meanings and citations are unchanged. Registering `animate` also moves
one existing Drasi annotation from Description into Tags while preserving its ID. This is the
only other changed lexical-record field pair; no existing lexical spelling or definition is lost.

The default full-suite command encounters an existing duplicate module basename between the
Bhumij and Noira `test_preintegration_contract.py` files. The complete suite is therefore also
run with `--import-mode=importlib`, which gets past collection. It finishes with
**1,668 passed, 14 skipped and 30 failed in 444.11 seconds**. No Perder test fails.
The two source-local test files also import a helper under the same bare `preintegration_audit`
name, so two Noira checks see Bhumij's helper in the combined run; **both pass when rerun alone**.
The other 28 failures concern existing source row counts/legacy IDs, older reference/editor/OCR
expectations, language/dialect metadata, or existing/concurrent etymology edits. The exact failing
test names, command outcomes, initial baseline, graph counts, stable-ID reconciliation and log
hashes are preserved in [the validation report](20260909-perder-dameli-validation.json).
These repository-wide failures are not hidden or treated as a clean full-suite result. The final
generated-checklist freshness check also passes (1 passed in 14.13 seconds).

The frontend type check passes with **0 errors and 7 existing warnings**.
The user explicitly deferred the browser database refresh. No browser database was rebuilt,
staged or served, and browser QA is inapplicable to this ingestion handoff. No commit, push
or deployment was requested.

Representative records for the eventual app refresh: ištrii ‘wife’ (p. 34), c̣ai ‘body’
with source IPA (p. 40), draakmuṭ ‘vine’ and its components (p. 50), Aspar kinship terms
(pp. 10, 67–69), and aċap / žup at the verb appendix boundaries (pp. 207–208).
