# Eight selected surveys: ingestion review, 2026-09-11

Status: five sources installed and compiled; source-focused validation passes. Full completion is blocked by existing pipeline failures and three selected sources that cannot supply reliable wordlists from their public copies.

## Coverage and exclusions

| Source | Target cells | Installed rows | Excluded cells |
|---|---:|---:|---:|
| LSI Jharkhand (online 2023; prepared 2021) | 4,000 | 4,407 | 251 |
| LSI Himachal Pradesh (2023) | 7,042 | 6,442 | 976 |
| LSI Rajasthan I (2011) | 3,500 | 4,567 | 20 |
| LSI West Bengal I (2016) | 1,500 | 1,835 | 4 |
| Mahato, Kisan (2014) | 1,050 | 1,067 | 0 |
| **Total** | **17,092** | **18,318** | **1,251** |

Lexical alternatives expand some cells. Exclusions are blanks/placeholders, two Pangwali embedded images with missing-glyph boxes (water and flute), and one unresolved Kangri cell (name). No OCR was used. Published form/gloss inconsistencies are retained, never silently corrected. Only Indo-Aryan and Dravidian columns are included: five Austro-Asiatic Jharkhand columns, three Tibeto-Burman Himachal columns, and six non-Indo-Aryan West Bengal columns are outside scope. Census regional lists are targets, not automatically excluded as controls.

Himachal has 503 printed rows per lect, not exactly 500: item 341 is absent, and 327, 362, 493, 494 recur. Page, item, column and vertical cell position form stable source keys. These preserve same-page repeated numbers without inventing corrected numbering.

**Bote (Shrestha and Saphkota 2013):** the 100-page file lists Annex D but contains no response table; the annex moves straight to informants. **Rajbanshi–Tajpuriya (Yadav 2014):** the 100-page file ends with references, with no response-table annex. Similarity percentages cannot supply lexical forms. **Darai (Regmi and Thakur 2015):** all 1,050 response cells in its purported Annex E match the already-ingested canonical Danuwar table ignoring layout. The sites are also Danuwar sites. Excluded rather than falsely attributing Danuwar forms to Darai. `darai-excluded-audit.jsonl` accounts for every cell. Government mirrors of all three are byte-identical to the university copies; URLs and results are in `unavailable-sources.json`. A complete/corrected source is required to continue those three.

## Reproduction and source evidence

Importer: `data/other/forms/raw_data/more_surveys.py --output /tmp/more-surveys-proposal`; `--install` updates canonical outputs. Positioned table extractor: `more_surveys_2026/extract.py`. PDF hashes, URLs, date and extracted-cell hashes are pinned in `snapshot.json`. PDF page counts and extraction ranges are asserted. Offline imports use hashed cell snapshots; original PDFs and full text are local cache files, not committed publications. No explicit open redistribution licence identified; the installed data consists of lexical facts with attribution.

Every included cell has raw text, page, bounding box, image flags, language mapping, emitted rows and typed review reasons in a JSONL audit. `readings.json` records individually parsed annotations and reviewed line continuations. Raw spelling becomes Original; only the sound-profile layer converts display Form. No separate source phonemic analysis is invented. Native-script prompt translations in Kisan are controls, not lexical native forms.

## Editorial and metadata decisions

All 18,318 rows carry `uncertain` with typed transcription/gloss reasons in the audit because the sources use inconsistent transcription and sometimes demonstrably mismatched prompt meanings. This is a discovery/review flag, not a claim that every cell is corrupt. `more-ascii` converts explicit ASCII retroflexes/central vowels and known length/aspiration notation. `more-ipa` handles Kisan and the Sirmauri, Pangwali and Dogri columns. Ambiguous symbols are preserved and inventoried rather than assigned guessed phonemic values. Source optional parenthetical endings remain exact notation; no unattested expansions are synthesized. Grammatical labels are removed from forms and tagged; sense-specific annotations enter glosses. Source “taboo” is retained in notes with the broad existing `vulgar` tag.

Canonical language mappings are in `language-map.json`. Existing bases are reused for Khortha/Magahi, Keonthali/Mahasu and Panch Pargania/Kurmali; Sadri covers the source's distinct Nagpuria and Sadan/Sadri lists. Sanori is kept under its own conservatively unclassified Western Pahari base because the source treats it separately and independent narrower placement was not established. Kisan is explicitly Indo-Aryan in this report, not Dravidian Kisan. No Glottocode is invented where uncertain. Regional/locality names are retained with blank numeric coordinates rather than false site precision. Explicit Sikaripara forms receive a locality tag.

No source makes accepted etymological or borrowing claims: Parameter_ID and graph-parent fields remain blank. Synonyms and numbered repeated prompts retain separate keys. Shared-file deduplication preserves citations and source-key aliases; all 18,318 source keys and citations survive the compiled build.

## Audit and validation

A preliminary 20-cell sample per source (seed 20260919) was visually checked, followed by targeted annotation and wrap fixes. Fresh seed **20260920**, 20 cells per source: **0/20 material extraction errors for each**. Samples are checked in; `review_sample.py` reproduces selections, with `--render` making contact sheets from normalized cached PDFs. Extra reviews cover first/last pages, split table boundaries, all bracket annotations, missing embedded glyphs, source numbering, the Kisan legacy-font prompt column, and the entire Darai/Danuwar overlap.

Focused importer/compiled/profile/dialect tests: **33 passed**. Re-extraction reproduces the pinned cell snapshots. The generated source catalog is current (`audit_source_ingestions.py --check`); its full-pipeline boxes remain unchecked because the global validation is not clean. Full suite: **1,808 passed, 31 failed, 18 skipped** (527.39s). All 31 failing test names also failed in the preceding baseline run; **no newly failing tests**. Two former failures no longer fail. The exact comparison is in `audits/20260911-more-surveys-test-comparison.json`. The suite is run as `python -m pytest -q --import-mode=importlib` because plain `pytest -q` encounters existing duplicate-module/path collection errors.

The complete data stages of `make all` ran and produced **785,057 forms**, up from 766,739: exactly **18,318** new survey nodes, all unlinked. It then exited 2 on the two already-failing manual-survey-etymology tests. All 18,318 source keys survive, all citations survive, five references were added, and the new nodes introduce no graph edges. The complete edge and alignment files have exactly their baseline hashes. Source-key and legacy-alias registries each gained 18,318 rows. See `audits/20260911-more-surveys-build-validation.json`.

**Global build caveats:** concept counts changed 3,272→3,269 and concept links 556,154→573,762. A read-only recomputation without this batch did not reproduce the previous concept counts. Inspection found that pysem resolves equal-scoring matches from a set without a deterministic final tie-breaker; identical glosses under hash seeds 1 and 2 produce different concept matches (six demonstrated cases in `audits/20260911-more-surveys-mapper-ties.json`). Thus these concept deltas must not be interpreted as exact new-source coverage. No mapper code was changed in this ingestion.

An older 746,336-form snapshot comparison found no changed lexical content among surviving IDs, but four older Toulmin/Bundeli IDs were retired and their identical lexical records have new IDs. Current legacy aliases locate the replacements; the identity registry was preserved. The registry grew by 18,333 rows, 15 more than the new-node count, across the interim and final builds. This older-identity churn remains a pipeline caveat, not an editorial change authorised by these surveys. See `audits/20260911-more-surveys-existing-forms.json`. The batch itself has no missing source keys.

Representative compiled entries (browser refresh deferred):

| Source | Form ID | Form | Gloss |
|---|---|---|---|
| Jharkhand Hindi | `f_d7i5k2bxbahuo` | hawa | air |
| Himachal Kulvi | `f_s2hveufm2mfkm` | bagər | air |
| Rajasthan Marwari | `f_2tfogbztlh3lc` | həwa | air |
| West Bengal Bengali | `f_mxwdcdb46lf6m` | haoa | air |
| Kisan Dhaijan | `f_f3axfgo6cgp6u` | deɦ | body |

Files changed include five dated form CSVs, the importer/extractor and hashed snapshots, per-cell audits, transcription inventories and two profiles, language/dialect and bibliography metadata, source-key-preserving build routes, focused tests, generated CLDF and source-checklist records. Full pipeline and focused logs are in `source_checklists/audits/20260911-more-surveys-*`.

Browser database refresh and browser QA: **not requested for this ingestion**, per checklist section 13; deferred until user requests refresh. Shipping, commit, push and deployment: not requested. Dictionary, OCR-heavy, website/API and etymological-source addenda are inapplicable; the survey/comparative-table addendum applies.
