# CFEL Mahali 2024

The source-ingestion checklist is active with dictionary/glossary, scanned-source and publisher API comparison addenda. The source is Pradhan and Tripathi’s 2024 *Mahali–Bangla–Hindi–English Dictionary*, Centre for Endangered Languages, Visva-Bharati, ISBN 978-81-957226-9-3. The 387-page PDF is pinned by SHA256 `4214334508edc9668f35301249d5c1a5110c2580594c83fa5259bf70d921c5a2`. A supplied local copy is `../tmp/pdfs/cfel-koda-mahali/mahali-multilingual.pdf` from the data repository; the similarly named other PDF is a different edition.

## Full source coverage

All 2,451 lexical entries on printed/PDF pp.10–387 have stable page/item keys. The table of contents numbers 52 sections, of which 50 are lexical domains; the other two are introductory. `candidates.jsonl` preserves first extraction, `positioned-candidates.jsonl` preserves corrected combining-mark placement, and `domains.json` preserves domain boundaries even when two domains share a page. Ten numeral entries have no description. Fig on p.380:item2 lacks the opening IPA slash in print and remains explicitly accounted.

On 2026-09-26 both complete candidate inventories were regenerated from the pinned page caches and matched exactly. The same 2,451 keys occur in raw candidates, positioned candidates, Native review, analysis, draft audit and canonical CSV. `full-source-review-20260926.json` retains the physical page census and grammar inventory. There are no omitted lexical records or pending native readings.

## Transcription and native evidence

Default PDF line clustering displaced elevated IPA tie bars. Geometric correction anchors all 1,377 combining glyphs and changed 204 IPA candidates; all 211 ties inside IPA remain explicit. All 28 unusual-symbol cases were visually reviewed. Literal question marks, internal dots and unusual diacritics remain typed uncertainties. A missing opening slash or mark attached to the delimiter is recorded without imposing it on a consonant. Eyelid’s meaningful hyphen survives a line wrap.

Native PDF extraction has unmapped glyphs in 1,176 candidates. A Bengali OCR pilot was rejected for unattended installation. `native-recovery-review.jsonl` instead records all 2,451 accepted readings: 2,443 publisher-assisted visual readings and eight direct printed transcriptions. The publisher is a correspondence aid, not a substitute edition. Eight print/online disagreements remain in `publisher-edition-review.jsonl`; the printed source controls. Earlier `native-font` issues describe raw extraction damage, while `native_status` records accepted reviewed Native.

The source profile retains source vowel qualities and unusual notation, maps IPA affricates and glide to house transcription, preserves hyphens and handles NFC/NFD consistently. Original IPA is the raw form; Native is the reviewed Bengali-script headword. No duplicate phonemic layer is invented. Twenty-two rows carry source-specific transcription or semantic uncertainty, including identical source forms assigned to thirty-four and thirty-nine and one printed Native/IPA disagreement.

## Meaning, grammar and source restrictions

Simple POS is structured. All 180 component-label chains remain in Notes and in the analysis audit; the complete expression receives `multiword-expression`, rather than every component POS. Numeral, ordinal, causative and compound domains contribute their explicit grammatical labels. Short English labels are preserved, with earlier disambiguation for weight-unit Grain, material Glass, two Fall senses, Bachelor senses and the flower labelled Belly.

Fresh independent review on 2026-09-26 found source-description omissions despite matching transcription. Four failed samples and the transparent correction of one reviewer mistake are retained. We then read every one of the 2,451 descriptions: root reviewed pp.10–69 (336 entries), and asur_finish reviewed pp.70–387 (2,115 entries). `description-class-census-20260926.json` records all decisions; `description-review-20260926.json` contains 358 source-backed amendments. They retain lexical restrictions, local cultural timing, grammatical claims, polysemous English senses and source head/definition discrepancies. Ordinary encyclopedic expansion is excluded.

The amendments affect only Notes and Tags; every existing key, IPA, Native and English label is preserved. Source inconsistencies remain attributed rather than silently corrected, including Thunder described as lightning, Find described as searching, and several erroneous numerical neighbours. Explicit source plurals and a command receive corresponding tags; completed events do not acquire an inferred tense. Historical samples remain immutable, with the later documented Notes/Tags amendments explicitly incorporated by `audit_draft.py`. Fresh independent pass5 (seed 2026092649) passed with zero material errors in 20 entries; the reviewed proposal is installed.

## Metadata and reproduction

All rows use existing canonical `Mahali` (Glottocode `maha1291`), source key `pradhan-tripathi2024mahali`, and printed page/item citations. The source’s regional distribution and named resource persons do not establish entry-level collection sites; no dialect or coordinates are invented. Coinages are part of the source’s stated method, but individual entries receive no inferred coinage, borrowing or ancestry labels. This is an independent bibliographic witness; no other-source rows are removed or automatically linked.

```sh
.venv/bin/python data/other/forms/raw_data/cfel_mahali_2024/import_source.py
.venv/bin/python -m pytest -q tests/test_cfel_mahali_preparation.py
```

The importer regenerates `review-draft.csv` and `draft-audit.jsonl`; `--install` updates `data/other/forms/20260925-cfel-mahali.csv`. It never builds a database. The existing canonical CSV contains all 2,451 rows; the 358 semantic amendments are installed. All 46 focused source/profile/dialect tests pass, scoped parsing retains all 2,451 rows without errors, and source metadata validation passes (275 files, 263 citation keys). Full-source parser/profile, identity and metadata checks are lightweight. Full CLDF build, full suite and browser gates remain deferred under the user’s explicit instruction.

## Reuse basis

The printed edition identifies copyright Visva-Bharati; the publisher site states all rights reserved. No open licence or permission was identified. The package records attributed lexical facts in Jambu’s schema under the project’s existing editorial policy, not a claim that the dictionary is public domain. The installed CSV excludes PDF pages, images, prose definitions, Bengali/Hindi translations and complete API responses. Source comparison evidence remains local. No public release is performed by source-stage installation.
