# Bailey 1908 Pādari — complete source stage

Thomas Grahame Bailey, *The Languages of the Northern Himalayas* (London: Royal Asiatic Society, 1908), public-domain original. Source: https://archive.org/details/languagesofnorth00bailrich. The retained original `tmp/pdfs/bailey-sainji/bailey1908.pdf` has 358 pages and SHA256 `953f5da5ee9bc341cdf7eb558920e824dc6f15f7473a8c0d5c8134c401ad60b5`.

The complete dedicated scope is Part III pp. 76–84 and Part IV pp. 33–35, together with explicitly attributed Padari examples in the introductions and comparisons. All **753 physical units** have two original-image readings: **748 nonblank units yield 768 rows**, including 20 alternate forms; **five printed blanks** remain accounted for. All 13 pilot keys survive. The old bounded importer, audit, CSV, profile and tests are preserved in `legacy-pilot/`; the old `transcription.tsv` and sample remain historical pilot evidence, not the current coverage claim.

`full-transcription.jsonl` is the complete reviewed inventory. `audit.jsonl` records every unit, source location, explicit grammatical category, source variation and blank. `import_source.py --install` regenerates the rows and verifies the independent review's four input hashes before copying the proposal CSV, audit and profile to their canonical source-stage paths. Run it from the data root with `.venv/bin/python data/other/forms/raw_data/bailey_padari_1908/import_source.py --check-pdf --install`.

## Editorial and metadata decisions

Forms preserve Bailey's literal historical Roman notation, including macrons, breve vowels, superscripts, underlined consonants and shared macron e͞u. These are not asserted to be IPA. The literal profile is fully covered and round-trips every form; its source-specific `w` exception is registered in `profile_policy.py`. Two ink-fused i-length uncertainties on p. 77 remain explicitly typed; the clearly long alternative maī̃ is not flattened. No source word is silently regularized by analogy.

Explicit correlative columns, Time/Place adverb headings, comparison degrees, paradigms and verbal headings are represented in tags. Contextual glossary POS is identified as such in Notes. Six unlabeled verbal column positions remain recorded without invented person-number assignments. Source asterisks identifying resemblance to Pangwali are preserved as source claims, not invented etymological edges. Twenty alternate rows point to their own existing source head; no dangling local relation exists.

The source lect maps to the existing `Padri` language. Part IV locates Padar east of Kishtawar on the Cinab; Part III places it north of Pangi and contiguous with Bhales. No finer locality, coordinates or unsupported modern classification is inferred. Bibliographic and importer metadata cover the full scope and identify manual image reading (`ocr: false`).

The hair attestation re-quoted by Zoller is retained with explicit reuse accounting in `same-source-reuse-alignment-20260926.json`; it is not counted as independent field evidence. The former exclusion of Bailey's fox as the same print as Zoller's LSI citation was unsupported and has been removed. Existing Zoller rows remain unchanged; compiled identity reconciliation is deferred.

## Audit and validation

Retained independent passes 1 and 2 document their failures and the systematic repairs. Pass 3 (`independent-full-audit-20260926-pass3.json`, seed 20260926120) independently checked a fresh 20 units, excluding the earlier 40: **20/20 passed**, plus targeted auxiliary hauᵃ̆, sentence hauᵃ and comparison checks. All input hashes match the installed files.

**11 focused tests passed** across `test_bailey_padari_1908.py` and `test_bailey_padari_full_stage.py`: deterministic regeneration, complete inventory, all legacy keys, all 768 profile round-trips, metadata routing, all 20 local edges, shared tag parity and actual scoped parser conversion of 768 records with no errors. See `source-stage-validation-20260926.json` and `SOURCE_CHECKLIST.md`.

This is complete **source-stage** extraction and installation. Full database/CLDF generation, compiled graph and global identity/reference checks, full-suite testing and browser QA remain deferred under the user's explicit no-build instruction. No production or browser completion is claimed. Representative source keys are `bailey1908padari:p82:right:item:1` (pig), `bailey1908padari:p77:correlative:3` (interrogative adjective), and `bailey1908padari:part4:p33:aux:pres:1` (hauᵃ̆).
