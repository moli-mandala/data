> **Whole-source scope reopened (2026-09-26):** Existing installation covers only the glossary. Preglossary grammar and complete translated responses remain to be recovered; see `whole-source-reopening-20260926.json`.

# Bailey 1920 Rohru glossary

Thomas Grahame Bailey, *Linguistic Studies from the Himalayas* (Royal Asiatic Society, 1920), is public-domain primary print evidence. The [Internet Archive scan](https://archive.org/details/linguisticstudie00bailrich) has 310 PDF pages; the ignored local original is `tmp/pdfs/bailey1920/bailey1920.pdf`, SHA256 `7a1577a2d2a24eca8b95e9b1ed700ef01f716d1069e4468f36a673611d093b39`. Printed p. 113 identifies this Koci dialect as spoken around Rohru. It maps to existing canonical `roh` (Rohrui); the spelling is a source name for that language, not a separate survey site. No new dialect or invented historical coordinate is introduced.

## Full glossary scope

All **255 source cells on printed pp. 127–130 (PDF 153–156)** are accounted for, from *able* through *you*. The left/right column counts are 35/31, 37/32, 36/35, and 26/23. An editorial cell includes wrapped continuation lines and separately glossed answers within one English headword. The following chapter is Rampur and Baghi, excluded as other lects. Grammar, paradigms, and example sentences preceding this separately titled vocabulary are excluded explicitly.

`transcription.tsv` and `completion.tsv` preserve the earlier partial transcription and its audit evidence. The authoritative `full-transcription.tsv` incorporates full 450dpi review of all four pages, with the individual recovery/correction decisions in `literal-recovery-20260926.tsv`. **250 accepted cells yield 309 rows; two specific vowel-stack readings and three English cross-reference-only cells remain audit-only.** The two held cells are *storm* (p.129 right25, stacked mark over e) and *village* (p.130 left25, exact nasal/macron stacks); their legible candidates and precise uncertainty are recorded. All generic withheld-reading placeholders have been removed. Grammar, paradigms, and example sentences outside this separately titled glossary remain outside the declared scope.

## Representation and validation

The deterministic importer emits the headerless 15-column `data/other/forms/20260925-bailey-rohru.csv` and the 255-record `audit.jsonl`. Keys identify printed page, column, and cell; additional answers have stable `:answer2` etc. suffixes. All 94 previously installed entry keys are preserved, with source-backed spelling corrections. Distinct concepts and homographs retain their own keys; source identity settings prevent accidental collapsing. Multiple forms under a shared definition are expanded, and separately defined answers have their own glosses. Explicit noun/feminine labels are structured as tags. No phonemic, cognacy, borrowing, derivation, or etymological claim is inferred.

The source-specific literal preservation profile retains source transcription rather than asserting phonemic equivalence. Every accepted grapheme is covered; `Original` is preserved by the scoped parser. The YAML declares the conversion, stable entry identity, no automatic comma splitting, editor, and OCR participation. OCR was a comparison input for the completion inventory; every accepted form received visual checking. The BibTeX entry records the complete glossary extent, provenance, and public-domain edition.

`sample-review-20.tsv` and `independent-sample-audit-20260926.json` preserve historical accepted-subset checks and do not prove recovery completeness. The new independent literal audit pass1 found one error in twenty (p.130 *very*: bōhri, not bŏhri); pass2 checked twenty fresh cells, five per page, with zero errors. Reports are `independent-literal-audit-20260926-pass1.json` and `independent-literal-audit-20260926-pass2.json`. Edge checks cover underlined digraphs, nasal/macron stacks, causatives, gender/number and transitivity. The source inconsistently prints p.128 *much* bŏhri versus p.130 *very* bōhri; both are retained. The focused checks verify literal profile coverage, registered tags, source scopes, reproducibility and scoped parser survival. Full-pipeline validation remains deferred.

Reproduce from the `data` directory:

```sh
.venv/bin/python data/other/forms/raw_data/bailey_rohru_1920/import_source.py --install --check-pdf
.venv/bin/python -m pytest -q tests/test_bailey_rohru_1920.py tests/test_sound_profiles.py tests/test_dialects.py
.venv/bin/python source_meta.py
```

Full CLDF build, graph/durable-ID reconciliation, formatted-reference regeneration, full-suite tests, and browser database/app QA remain deferred under the user's explicit no-build restriction. Source-stage extraction now includes all legibly recoverable glossary cells; the two specific unresolved readings remain disclosed. It does not assert those compiled gates passed. Existing Zoller/Patyal Rohru attestations remain untouched; overlapping quotations in the later LSI addenda are derivative comparators, not independent observations.

## Screened alternative: Doda Siraji

Bailey 1908, Part IV, printed p. 43 has 36 direct-gloss left-column items headed Siraji; its preface explicitly calls this chapter “Siraji (Doda Siraji)” and printed p. 36 locates it near Doda. All 36 gloss prompts in that column occur in the already installed `LSI-DODASIRAJI` vocabulary, with conspicuously matching source readings such as `ikk` one, `satt` seven, `nau` nine, `das` ten, `zanan` woman/wife, and `mattho` child. Jambu's earlier LSI audit manually mapped `DODASIRAJI` to canonical `dod` Dodi (registry Glottocode `kash1277`), while that audit, its registered LSI dialect, and current Glottolog identify the source lect with `sira1264` “Siraji of Doda”; canonical `sir` Sarazi already carries that code. This is a genuine historical classification/registry conflict, not grounds for silently remapping forms by name. The substantially overlapping Bailey page was **not** installed or registered as another source. It remains a review lead if a later linguistic registry reconciliation is authorized.
