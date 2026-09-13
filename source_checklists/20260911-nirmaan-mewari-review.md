# Nirmaan Mewari dictionary — ingestion and local rebuild review

**Locally rebuilt and staged, 11 September 2026, following the user's explicit “rebuild db!” instruction.** This supersedes the earlier data-only restriction for this rebuild. Compilation, database transformation, staging and browser checks succeeded; test-suite exceptions are recorded below. This is a new dictionary reference, `nirmaan2018mewari`; it does not replace the survey reference `mewari` or its curated etymologies. Nothing was committed, pushed or deployed.

## Local rebuild validation

- All CLDF compilation stages completed, including references, concepts, alignments and persistent IDs. The compiled corpus contains **686,791 forms**, with **6,560 added IDs and zero removed IDs** relative to the pre-build backup.
- Exact raw-to-compiled reconciliation passed for all **6,535 dictionary forms and 484 variant links**, including native text, source IPA, original text, dialect tags and endpoints. There were zero dictionary conversion errors. All 6,535 opaque IDs are present in the compact browser database; 484 nodes have variant parents and 6,051 remain parentless.
- `npm run db:transform` and `npm run db:stage` succeeded. The SQLite image is **99,368,960 bytes**; the compressed browser artifact is **43,861,495 bytes**, below the 50 MB hard limit. The uncompressed image exceeds the 97 MB soft expectation. SQLite integrity is `ok`; graph transformation dropped zero unknown endpoints and zero unreachable alignment rows. Browser cache version is **31**.
- `npm run check`: **zero errors, seven warnings in five files**.
- After refreshing generated checklist copies, their manifest and the installed-record audit: **12/12 dictionary and source-checklist tests passed**. This resolves the two stale-audit failures from the full run; 33 broader failures remain unaddressed. Authored source reviews were preserved separately from generated checklist copies.
- The final `make all` survey-test gate failed two assertions after compilation: a fixed Rajasthani count expects 15,876 but the expanded corpus has 15,887; another assertion forbids manual survey overlays although **1,604 such overlays already exist in the pre-build backup**. Accepted etymologies were preserved.
- Full Python suite: **1,699 passed, 35 failed, 18 skipped** in 714 seconds, using `PYTHONPATH=. uv run pytest -q --import-mode=importlib`. Default collection encounters six module/import errors. The companion `audits/20260911-nirmaan-mewari-build-validation.json` records every failing test. Failures include obsolete compiled-column/ID expectations, other sources' graph/metadata assertions, coordinate gaps, fixed survey counts, and stale generated ingestion audits. The audit artifacts were subsequently refreshed while preserving authored reviews; focused retest results appear in the companion artifact. Broad failures were not all baseline-tested, so they are not collectively asserted to be pre-existing.
- Browser QA passed for the [source page](http://localhost:5173/references/nirmaan2018mewari), its searchable 6,535-row table, [akkar ‘letter’](http://localhost:5173/entries/f_alo7glcltiybk), [ɓārī ‘a kind of broom’](http://localhost:5173/entries/f_v7k6slu4dag5o) and its link to bārī, and the [Mewari page](http://localhost:5173/languages/mewari_dholpura), which shows all 6,535 dictionary forms under Kapasan area. IPA, grammar and printed-location citations render correctly. A source-page screenshot was visually checked. The existing author-only short-reference formatter displays `?` for this editor-only source's heading; the full bibliography and entry citation chips correctly identify Jat et al. 2018.

## Scope and counts

- Source: *मेवाड़ी शब्दकोश / Mewari Dictionary*, Nirmaan Society, second publication January 2018 (first publication April 2017).
- Editors: Govardhan Lal Jat, Ratan Lal Gadri, Ram Singh Charan, Kavita Yogi, Mukesh Kumar Yogi. Linguistic assistance: Royson Norman D'Souza.
- [Source PDF](https://eternalmewarblog.com/documents/mewari_dictionary.pdf): 389 pages, SHA-256 `9adf6f4aefdd0705d29c255b664ac1bf193df2f7ab85cf0bfa33af30532344bd`.
- **6,406 articles**, exactly matching the stated dictionary count and the independently counted left-margin bold headword anchors.
- **6,535 installed raw rows**: 6,406 article rows + 128 additional numbered senses + one additional pronunciation. There are 121 multisense articles and 376 heads with explicit homograph numbers.
- **484 variant edges**: 483 connections between printed articles plus the printed `baɾi/ɓaɾi` alternative. The 1,146 printed variant target occurrences include reciprocal/repeated claims: 1,121 are represented in variant families; 19 have unresolved sense scope and six have ambiguous/unmatched targets. Counts of target occurrences, articles, and graph edges intentionally differ.
- **27 articles carry unresolved relationship review**: 24 variant-target/sense cases and three compound-parent cases. Their lexical rows remain included with `uncertain`; unsupported relationships are not emitted. The table below identifies them.
- No inherited/borrowed etymologies inferred; 6,051 rows have no parent, and 484 have a variant parent. These counts are verified in both compiled data and the browser database.
- All 6,406 lexical articles included. PDF pages 1–18 contain front matter, title material, and blanks; printed lexical text runs pp. 3–373 (PDF 19–389). Headers, page numbers, alphabet ornaments and examples do not become forms or glosses. Hindi definitions, examples, scientific names and comparison prose remain recoverable in the article audit; English lexical definitions populate `Gloss`.

## Extraction and transcription

`data/other/forms/raw_data/nirmaan_mewari.py` uses PyMuPDF font spans, column positions and clustered baselines. Pypdf/pdfplumber text extraction gave wrong Devanagari mappings and was rejected; their rendered pages remained legible. No OCR was used. Source PDF and temporary renders are not committed.

The importer preserves page/column/article keys, raw extracted text, decoded text, native headword, homograph number, numbered senses, IPA, grammatical labels, candidate relationships and decisions in the JSONL audit. Keys depend on printed location and headword order within that column, not spelling or gloss content. Main and continuation lines remain in reading order across columns/pages. Header/alphabet fonts are excluded explicitly. Typography, not free-text digits in examples, splits senses.

Source IPA drives `Form` conversion, remains unchanged in `Phonemic`, and becomes `Original` through the normal raw-row reader. Devanagari goes in `Native`; homograph numbers remain in the audit and immutable article identity. The single slash alternative is expanded into two rows, with the complete printed IPA retained in the audit.

The dedicated profile `conversion/nirmaan-mewari.txt` is routed by bibliography key in `make_cldf.py`, with a filename fallback in `utils.py`. The alphabet on p. viii explicitly pairs ə/a, ɪ/i and ʊ/u with अ/आ, इ/ई and उ/ऊ; display mappings are therefore a/ā, i/ī and u/ū. Dental diacritics and affricate ties are normalized; retroflexes, aspiration and nasalization are preserved. Source `j` is the glide y. The exceptional advanced `t̟` and implosive `ɓ` are retained rather than silently regularized. Both NFC and NFD source forms have complete profile coverage.

Font repairs are mechanical and explicit: Krishna-font `Dd/Uu/Yy/Pp` become क्क/न्न/ल्ल/च्च, with a closed table for the remaining legacy tokens. This mapping never applies to English fonts. Duplicate vowel signs and doubled य in the PDF text layer are repaired. Invalid `न्`+vowel-sign sequences restore the visually verified न्द ligature; the two exceptional damaged heads आछ्यो and टड्डो have targeted repairs. Raw pre-repair text remains in the audit.

POS labels are parsed as grammar, including सं. = noun (not Sanskrit), क्रि.वि. = adverb, and source संखया = numeral. Numbered senses retain their own POS. Explicit parenthetical masculine/feminine/slang labels become `m`/`f`/`colloquial`. Examples and a definition describing a slang expression are not mistaken for labels.

## Language, provenance and relationships

All forms use canonical Mewari `mewari_dholpura`. A registered `mewari_kapasan` dialect tag represents the Kapasan-area basis of the dictionary, which was checked more broadly across Mewar. It does not claim a village for each attestation. The approximate map anchor is Kapasan town, 24.88775, 74.31232, quality C, from [the OpenStreetMap/GeoNames record displayed by Mapcarta](https://mapcarta.com/14881546); it is explicitly not an entry-level collection point. No new base language or clade was created.

Every row cites printed page, column and article ordinal, plus sense where applicable. The BibTeX record contains edition, editors, publisher, ISBN, URL, inclusion, importer/audit provenance and editorial credit. Its citation was formatted successfully with the existing Pybtex engine. `cldf/references.csv` was regenerated during the local rebuild.

The source marks free (`मु. रू.`), dialect (`बो. रू.`), spelling (`वर्त. रू.`), and a few English unspecified variants. These assert equivalence, not directional inheritance. The importer resolves exact normalized headwords, including explicit homograph numbers; a unique longest headword prefix handles appended Hindi disambiguating glosses. It never fuzzy-matches targets. Single-sense connections form a deterministic spanning tree of explicit source claims, preferring a fully defined article as root. Each child cites the article supplying the printed claim, even when it occurs in the opposite direction. Reciprocal statements therefore cannot create cycles. Multisense and ambiguous targets remain unresolved. Compound labels at the start of an article are distinguished from trailing synonym labels, but their three parent/sense ambiguities remain unlinked.

## Validation and deferred gates

- Source-specific tests: **8 passed**. Cover all counts/keys, re-emission from the audit, column/font/POS regressions, script and IPA separation, complete NFC/NFD profile coverage, registered dialect/reference, all variant endpoints and cycle freedom, and 6,535 persistent IDs assigned/reordered/corrected **in memory only**.
- In-memory `parse_file`: **6,535/6,535 converted**, zero replacement characters or conversion errors. The shared compiler now protects this source's immutable keys against article/homograph merging.
- Fresh final seeded image review: **0/20 material errors**, seed `20260913`. The saved sample records the reviewed keys. Earlier exploratory samples used seeds 20260911 and 20260912; targeted repairs were followed by the fresh final sample.
- Additional rendered checks: first and last entries; corrected legacy/conjunct heads; homograph numbering; numbered senses; wrapped variants; numeral POS; advanced t; slash/implosive pronunciation; explicit gender/register labels. Regression fixtures preserve positioned spans for difficult cases.
- Broader `tests/test_dialects.py`: **four passed, two failed** on pre-existing missing coordinates/quality. HEAD and current data both have 279 dialects without coordinates and 15 without quality; this import adds zero offenders. The new dialect passes the source-specific registration/coordinate test. These unrelated records were not changed.
- **Formerly deferred gates:** the authorized local rebuild now covers compilation, compiled row/edge/node reconciliation, deduplication and ID verification, persistent registry writes, generated references/concepts/alignments, browser DB and app QA. Broad test-suite exceptions remain explicit in the rebuild validation; ingestion is not described as having every gate green.
- **Not applicable:** OCR, donor-node construction, new language/clade registration, auxiliary cited etymological dictionaries, GPU/cluster jobs. Source text supplies lexical/variant evidence, not etymological ancestry.
- **Rights:** the PDF says all rights reserved, © Nirmaan Society 2017. No open redistribution licence has been established. Work is local data preparation; no publication, commit, push or deployment was performed. The source PDF is not bundled. Any later redistribution needs the rights gate settled.

## Reproduction

From the `data` repository, with PyMuPDF available to `python3`:

```sh
python3 data/other/forms/raw_data/nirmaan_mewari.py --pdf /tmp/mewari-dictionary.pdf --output /tmp/nirmaan-mewari-review
python3 data/other/forms/raw_data/nirmaan_mewari.py --pdf /tmp/mewari-dictionary.pdf --install
python3 data/other/forms/raw_data/nirmaan_mewari.py --pdf /tmp/mewari-dictionary.pdf --sample-seed 20260913 --output /tmp/nirmaan-mewari-sample
.venv/bin/python -m pytest -q tests/test_nirmaan_mewari.py
```

These commands prepare or check data; none invokes a build. The source hash/page/article assertions fail clearly on a different or absent PDF.

## Files

- `data/other/forms/20260911-nirmaan-mewari.csv`
- `data/other/forms/raw_data/nirmaan_mewari.py`
- `data/other/forms/raw_data/20260911-nirmaan-mewari-{audit.jsonl,manifest.json,fixtures.json,sample.json}`
- `conversion/nirmaan-mewari.txt`, `utils.py`, `make_cldf.py`
- `cldf/sources.bib`, `cldf/dialects.csv`, `tests/test_nirmaan_mewari.py`, `README.md`

## Unresolved relationships

Lexical data is included for every row below. Candidate keys and precise source statements are in the JSONL audit.

| Article key (after source prefix) | Headword | Review reason |
|---|---|---|
