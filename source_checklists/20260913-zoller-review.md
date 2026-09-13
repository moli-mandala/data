# Zoller 2023 ingestion review

Scope: Linguistic data I–III, pp. 519–1035. Checklist active; etymological/comparative,
dictionary/glossary and comparative-table addenda apply. OCR and online API addenda
are inapplicable: a user-provided digital PDF was decoded from glyph positions.

Source details, exclusions, editorial decisions, language metadata and reproduction:
[data package](../data/other/forms/raw_data/zoller_2023/README.md).

Installed **17,754 rows** representing resolved lexical attestations in **313 languages**.
All **4,377 source units** and **32,470 typographic candidates** have per-record decisions.
The author’s reconstructed/Sanskrit comparison material and unresolved candidates are
preserved in the audit rather than asserted as additional attested entries. The scope
includes the section's resolved comparison languages. No generic family is treated as
an attested language. Unsupported historical-stage, ambiguous abbreviation, transcription,
and gloss assignments remain explicit unresolved cases.

## Extraction, metadata and editorial review

- Pinned PDF checksum, page boundaries, font decoding, source keys and record reconciliation: passed.
- Canonical language/dialect registry and full-symbol preservation profile: passed.
  118 base-language registrations were added; final rows use 87 source-owned dialect
  registrations and reuse existing dialects. Three earlier draft dialect registrations
  remain reserved but unused. Modern language coordinates are quality C; unavailable
  historical/locality points remain blank, including Kassite.
- Transcription: NFC diplomatic preservation, including raised letters/digits and lowered
  diacritics; no inferred phonemic layer. Greek native script is retained in Native.
  Ambiguous notation has typed audit reasons; damaged lexical spans are withheld.
- Graph: 341 direct CDIAL edges (334 reflexes, 7 borrowings), plus 70 explicit variants.
  Qualified/rejected/component proposals and DEDR comparisons remain source-attributed
  analysis rather than inferred ancestry. Auxiliary unresolved citations remain in record audits.
- Visual audit: final seed 933, **0/20 material errors**. Earlier failed seeds and source
  regression tests retain evidence for fixed systematic classes. Claim audit 1926: 0/20.
- Focused importer/profile/dialect checks: **26 passed, 2 compiled checks deselected**
  before the full build. After compilation, both source-key/reference and graph/layer
  checks passed (2 passed, 9 deselected). The corrected Gandhari registry test passed
  separately. Final full-suite results are recorded below.

## Validation

Baseline before this import: **31 failed, 1,833 passed, 18 skipped**, full pytest run
using `--import-mode=importlib`; baseline log `/tmp/zoller-baseline-tests.log`.
These existing failures must be distinguished from any new source regression.

All seven generation stages completed. The final `make all` manual-survey check
failed on the same two baseline tests; this is **not a clean full-build gate**.
The full suite finished: **33 failed, 1,842 passed, 18 skipped** in 626.80 seconds
(`/tmp/zoller-final-tests.log`). Thirty-one failures match the pre-ingestion baseline.
The two additional failures were fixed-cohort Nihali study tests: they incorrectly
included five newly ingested Zoller forms in a previously reviewed 4,299-record batch.
Those tests now select the original five lexical sources, and the study report states
that boundary. A Zoller regression explicitly preserves the five new forms without
inventing provisional ancestry. The affected-module retest passed: **45 passed** in 22.66 seconds, including all
12 Zoller tests and 33 Nihali-study tests (`/tmp/zoller-post-suite-checks.log`).

The full suite was not repeated after this test-scope-only correction; no installed
forms or graph output changed. Repository-wide clean validation remains an open gate
because of the 31 baseline failures.

Compiled reconciliation: **17,754 source rows → 17,754 unique nodes**, no source
conversion errors, and **0 of 837,499 pre-ingestion IDs lost**. The compiled graph
contains all direct and variant relations expected by the importer.

| Artifact | Before | After | Change |
|---|---:|---:|---:|
| Forms | 837,499 | 855,253 | +17,754 |
| Edges | 403,090 | 403,501 | +411 |
| Source keys | 486,803 | 504,557 | +17,754 |
| ID aliases | 1,226,146 | 1,244,934 | +18,788 |
| Concepts | 3,295 | 3,297 | +2 |
| Form–concept links | 597,022 | 608,877 | +11,855 |
| Alignments | 2,143,644 | 2,146,109 | +2,465 |
| References | 631 | 632 | +1 |

Alias history includes superseded unpublished draft rows, so its increase exceeds
current source-node count. The identity registry contains 1,030,321 rows (831,016
active, 199,305 retired). An interruption exposed the existing non-atomic registry
writer; recovery combined the intact partial output, the existing snapshot and
already-written compiled IDs. Every pre-ingestion active ID and every pre-existing
alias remains represented. See `identity-recovery.json` for the recovery audit;
835 aliases from superseded, unpublished Zoller drafts lack restored registry records.

The generated catalog counts all nodes carrying any cited key (including CDIAL/DEDR);
its broad citation total is not this source's node count. The source-specific counts
above and `build-results.json` are authoritative.
Browser database refresh and app QA are inapplicable under checklist §13 because this
request is ingestion only. No commit, push or deployment was requested.


## Candidate reconciliation and unresolved work

32,470 candidates: 16,586 ingested candidates expand to 17,754 rows; 4,100
reconstructions, 4,345 Sanskrit etymon comparisons, 4,171 context/unglossed spans,
2,403 unresolved language assignments, 861 generic-family/comparison controls,
3 unresolved transcriptions, and 1 unresolved gloss remain accounted for in the audit.
No known systematic parser error remained after the final seeded visual review;
this does not turn the held cases into approved lexical or historical claims.

## Representative compiled entries

These are CLDF IDs for a later user-requested browser refresh; they have not been
inspected in a refreshed browser database.

- `f_p7nvhxwt44zlg` — Pr **atī**, “inside, indoor; innen, drinnen”; zoller2023[pp. 537–538, 18.1, entry 130].
- `f_cr6j7gjt4y5zg` — kw **gʰãːɖ**, “dense (forest)”; zoller2023[p. 994, 18.7.14, entry 24].
- `f_vhtino7yiwlfq` — mewari_basad **minᵃkī**, “cat”; zoller2023[pp. 839–840, 18.4, entry 148];dedr[3851].
- `f_uqmejr7tmjo5c` — Tampuan **tata̤m**, “bachelor, young man”; zoller2023[pp. 948–949, 18.7.12, entry 4].
- `f_r5yn6ixamln3i` — bhatr **nã̄-kᵛüṭ̚**, “girl”; zoller2023[p. 641, 18.1, entry 868].
- `f_fbllvkw3ozkhu` — Mang **cuaŋ⁴**, “foot”; zoller2023[pp. 1014–1015, 18.7.17, entry 6].


## User-requested browser refresh

The subsequent “rebuild db” request activated the browser gate. Rebuilt and staged
`.dbwork/jambu.db` / `static/db/jambu.db.zst` locally, cache version 35.
SQLite integrity: **ok**. All **17,754 Zoller IDs** resolve with exactly matching
transcriptions. Browser deduplication leaves **17,662 Zoller-cited nodes** across
313 languages; merged IDs remain accessible. Full browser corpus: 720,668 nodes.

The SQLite image is 124,502,016 bytes, above the older 97 MB advisory warning.
The packed artifact is **48,801,375 bytes**, passing the hard 50 MB guard.
Stronger compression exposed a five-byte offset-read bug in fzstd 0.1.1; a licensed
vendored decoder fixes that read. Exact browser-decoder SHA-256 round trips pass,
with two focused bit-boundary tests and frontend check **0 errors, 7 warnings**.
Staging now checks size and decoder equality before atomically replacing the asset.

Browser QA at `http://127.0.0.1:5193`: source metadata and 313-language distribution,
source search for `gʰãːɖ`, its Korwa entry and printed locator, Punjabi variant
`f_2jingzwiqef6w` → `āse`, Tampuan language (70 forms), FIREPLACE concept 185,
and linked Himachali `ghyānnā` (`f_sqfhrfikk74uy`, CDIAL 66). The source map
reports 310/313 languages located, with unavailable locations left blank.
No publishing, commit or push was performed. Data-side baseline failures above
remain separate from this successful browser refresh.

## West Pahari language split — explicit user revision, 2026-09-13

The user requested that all named West Pahari varieties be separate language records
under the existing `W. Pahari` clade. This overrides the checklist's default treatment
of named varieties as dialects. No new source extraction or linguistic ancestry claim
is involved; the dictionary/comparative-source evidence remains unchanged.

- Added 15 language records: Bangani, Deogari, Himachali, Khashdhari, Khashi, Padri,
  Bauri, Bushahari, Barari, Outer Siraji, Shoracholi, Kotguru, Shimla Siraji, Kotkhai,
  and Western West Pahari (the source's broad western attribution).
- Reassigned 2,680 source rows: 2,678 formerly under WPah, plus two Bangāṇī rows
  formerly under Garhwali. Retained 32 unspecified rows under WPah, now displayed
  as West Pahari (unspecified). There are 328 source languages after this split.
- All 17,754 source rows, keys, source transcriptions, glosses, locators, 341 direct
  CDIAL claims and 70 variants are unchanged. Only language assignment and redundant
  former umbrella dialect tags change. Old dialect registrations are retained for
  historical links. No new locations or Glottocodes were inferred.
- `west-pahari-languages.json` is the reviewed metadata; `west_pahari.py` applies it
  during ordinary replay and mapping regeneration. `west-pahari-migration.json`
  accounts for every changed row and proves every other field identical. New PDF
  extraction, OCR, source acquisition, and transcription audits are inapplicable to
  this metadata-only correction; the earlier scan audits remain valid.
- Focused raw-source/profile/dialect checks: 28 passed (two compiled checks deferred
  until the rebuild). Frontend check: zero errors and seven existing warnings.
- Full pipeline, compiled identity/graph assertions, full suite, and browser refresh
  results are recorded below once completed.

## West Pahari release validation (db-v36)

The completed full suite reports 1,846 passed, 18 skipped and 31 failed. Failure IDs
match the recorded pre-change baseline exactly: no new or resolved failures. The full
data generation completed, but `make all` still exits nonzero at the two baseline
manual-survey etymology checks. This is not a clean repository-wide test result.
All 17,754 source IDs survive browser compaction with the expected language and
transcription; SQLite integrity is OK. The 15 promoted languages contain 2,680
attestations; 32 unspecified source attestations remain WPah. Existing graph claims
and permanent IDs are unchanged. Browser-decoder roundtrip passed; packed bytes
48,797,602, expanded bytes 124,518,400. See the release audit for deployment/browser QA.
