# Bailey 1908: complete Kotkhai chapter

Thomas Grahame Bailey, *The Languages of the Northern Himalayas* (Royal Asiatic Society, London, 1908), printed pp. 23–24 / PDF pp. 45–46. The original edition is public domain. The 358-page [Internet Archive scan](https://archive.org/details/languagesofnorth00bailrich) is cached outside installed data at `tmp/pdfs/bailey-sainji/bailey1908.pdf`, SHA256 `953f5da5ee9bc341cdf7eb558920e824dc6f15f7473a8c0d5c8134c401ad60b5`.

The complete source stage has **77 source units and 73 installed forms**: all noun and pronoun cells, six adverbs, auxiliary and finite-verb paradigms, the printed pluperfect expression, and all five vocabulary differences. Eleven genuinely blank pronoun cells and one Kiunthali comparison-control unit are accounted in the audit. Explicit plural ditto statements and stem-plus-suffix combinations are expanded; the pluperfect's unspecified “&c.” is not. No grammar or sentence section is excluded merely for not being a headword.

The previous two-row pilot covered only the final vocabulary paragraph. Its two entry keys and literal forms survive. The old importer, CSV and audit are archived as `legacy-import_source.py`, `legacy-pilot.csv`, and `legacy-pilot-audit.jsonl`; `transcription.tsv` and the old five-cell review remain historical evidence. The formerly held field and cold forms are now read as `pāṭṛī` and `shēḷā`. Rice `bīūjṇā` is retained as a primary-source attestation with explicit evidence that Zoller 2023 p. 686, entry 1223 quotes this very Bailey cell. That later quotation is not represented as independent elicitation, and no linguistic graph relation is inferred.

`full-transcription.jsonl` is the reconciled whole-chapter input. The original first reading is preserved separately. An independent reader checked all 77 units; the sole correction was `tinē` to `tīnē`, confirmed by comparison with the adjacent printed macron forms. A fresh stratified 20-unit final-output review (22 emitted alternatives), seed 20260926143, passed with zero material errors. The report pins the exact CSV, audit and profile hashes.

The literal profile lowercases display text and preserves source macrons, breves, underdots, nasal stacks and underlined `sh`. `Original` retains source case. The preface's contrastive vowel styling was checked; this chapter has no isolated contrastive vowel spans requiring additional encoding. Whole-word italic headings and lexical examples are ordinary source styling. `Native` and `Phonemic` remain blank. The printed English **“these”** in the place-adverb column is deliberately retained with a typed source-gloss uncertainty rather than silently changed to “there.” No glyph holds remain.

Forms use the existing canonical `Kotkhai` language and the registered `bailey1908-kotkhai` source dialect. The latter has no invented elicitation point or Glottocode. The existing canonical language's approximate regional point is not presented as Bailey's field site. All rows are unlinked; the source asserts no etymological, borrowing or derivational graph links here.

From the data repository, regenerate with:

```sh
.venv/bin/python data/other/forms/raw_data/bailey_kotkhai_1908/import_source.py --check-pdf --install
```

Installation requires exact agreement with the independent audit's frozen hashes. Reproduce its sample with `sample_full.py --seed 20260926143 --output review-selection.json`. Seven focused tests pass, including full regeneration, old-key continuity, literal symbol coverage, real reference parsers, actual registered profile/YAML/dialect, in-memory bibliography formatting, and all 73 rows through the scoped source parser with exact Original/Native/Phonemic preservation.

Full CLDF generation, compiled graph/reference/identity/alias checks and the full suite are deferred under the current no-build instructions. No browser refresh was requested or run. This is source-stage completion, not full-pipeline completion. No remote work, database build, commit or publication was performed.
