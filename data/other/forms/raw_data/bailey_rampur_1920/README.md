> **Whole-source scope reopened (2026-09-26):** Existing installation covers only the glossary. Preglossary grammar and complete translated responses remain to be recovered; see `whole-source-reopening-20260926.json`.

# Bailey 1920 Rāmpur full glossary

Thomas Grahame Bailey, *Linguistic Studies from the Himalayas* (Royal Asiatic Society, London, 1920), “The Koci Dialects of Rampur State.” The original edition is public domain. The [Internet Archive scan](https://archive.org/details/linguisticstudie00bailrich) has 310 PDF pages; ignored cache `tmp/pdfs/bailey1920/bailey1920.pdf`, SHA256 `7a1577a2d2a24eca8b95e9b1ed700ef01f716d1069e4468f36a673611d093b39`. Printed p. 113 places Rampuri north of Rohru; the named lect maps directly to existing canonical `ramp`. No invented historical coordinate or new survey site is added.

## Complete scope and recovery

The glossary header on p. 144 explicitly assigns the first word(s) to Rampur and the answer(s) after the colon to Baghi. All **245 source headword cells on pp. 144–147** are transcribed in `full-transcription.tsv`, *above* through *your*: 59, 66, 63, and 57 cells respectively. The *much* cell at the end of p. 145 includes its continuation at the top of p. 146. Single answers lacking a colon are treated as the first/Rampur answer; they are not copied into Baghi. Baghi controls, grammatical paradigms, prose examples, and other lects are explicit exclusions.

**242 accepted cells produce 288 rows**, with two English cross-reference-only exclusions and one actual unresolved reading: p. 144 item 19 *body*, where the stacked final vowel accent/length marks cannot yet be distinguished securely (provisional `jĕā́`). There are no blanket typography placeholders. Readable macrons, breves, consonant underdots, nasal marks, and underlined digraphs are retained even when a phonemic interpretation is not asserted. Mixed semantic or grammatical cells are expanded rather than discarded.

The old `transcription.tsv` and `sample-review-20.tsv` remain historical pilot evidence. They no longer drive the importer or define scope. All 17 formerly installed pilot keys survive; four spellings were corrected after fresh image comparison, recorded in `pilot-corrections-20260926.json`. Source text-layer snapshots `raw-ocr-p144.txt` through `raw-ocr-p147.txt` support reproducible comparison; final readings come from enlarged source images.

## Source notation and modelling

Bailey's pronunciation discussion on pp. 114–115 distinguishes long marks from short marks: the latter may indicate vowel quality rather than just duration. Those introductory pages are also cached as text. We therefore preserve literal source distinctions instead of guessing IPA. Underlined *sh* is represented by `s̲h̲` (U+0332 after each character); clear underdots use `ḍ ṇ ṭ ṛ ḷ`, breves use `ă ĕ ĭ ŏ ŭ`, and clearly stacked nasal/length marks retain both combining marks. The profile preserves these distinctions and applies the common house conventions `w → v`, `ṅ → ŋ`; `Original` retains all original source spelling, including `w` and `ṅ`.

Each printed answer has a stable page/item key, with `:answer2` etc. for expansions. Explicit noun/adverb/preposition, feminine, and transitivity labels become tags. Lexical glosses distinguish causative/base answers, cardinal/ordinal values, direction/deixis, and instrument/comitative meanings. The goat suffix notation is expanded to complete source stems. No phonemic value, etymology, borrowing, or directional variant relation is inferred. Same-form answers with different meanings keep distinct keys. Source-level identity settings protect those distinctions; the Baghi side belongs to its separate package.

The importer regenerates the 15-column CSV and a 245-cell audit. Focused importer/profile/dialect checks passed **25 tests**, scoped parsing retained **288/288 rows** without conversion errors, and source metadata validated 275 files/262 citation keys. Independent full-scope audits retained all failures: pass1 3/20, pass2 1/20, pass3 2/20. Eleven cell corrections and source-wide checks of nasal marks, k/kh, and a/u are recorded in `literal-corrections-20260926.json`. Fresh pass4 sampled 20 cells (five/page, excluding all prior 60) with **0 material errors**, and rechecked previous error classes; `independent-full-scope-audit-20260926-pass4.json` pins the image-audited readings before the final explicit grammar-tag pass; `grammar-completion-20260926.json` pins final hashes and records the tag-only completion. The final source-only suite passes **6 tests**, including audit-hash and corrected-reading regressions. Full CLDF build, graph/ID reconciliation, formatted-reference regeneration, full test suite, and browser database/app QA are deferred under the user's explicit no-build instruction; none is claimed passed.

```sh
.venv/bin/python data/other/forms/raw_data/bailey_rampur_1920/import_source.py --install --check-pdf
.venv/bin/python -m pytest -q tests/test_bailey_rampur_1920.py tests/test_sound_profiles.py tests/test_dialects.py
.venv/bin/python source_meta.py
```
