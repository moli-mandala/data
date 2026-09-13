# SDML lexical survey ingestion review — 2026-09-11

Checklist active: `SOURCE_INGESTION_CHECKLIST.md`; survey and website/API addenda.

## Source and extraction

CC BY-SA 4.0 official lexical export, pinned with hashes and acquisition URLs under
`data/other/forms/raw_data/sdml_2026/`. Original 271 village rows and 73 prompt columns
are retained; empty supplemental JSON verified. The methods describe 277 surveyed
villages; six are absent from the released lexical export, whose coverage is pinned here. Structured CSV extraction, no OCR.
No grammatical or narrative material. Native forms absent from the export; not inferred.
Survey locality responses are Marathi contact-language data, not claims about speakers'
ancestry or the donor language of each word.

## Counts and editorial decisions

19,783 raw cells → 47,956 tokens → 47,317 installed rows. 606 missing tokens excluded;
33 unresolved alternative/parenthetical/malformed tokens held in audit. 269 populated
sites; all 271 source sites registered below Marathi with source coordinates, quality A.
No added base languages. 37 retained unusual-symbol forms marked `uncertain` with
transcription reasons in the audit. 2,254 tokens have unavailable or unmatched frequency
lists; raw lists retained and no individual counts inferred. All forms unlinked, no
borrowings, variants or cognacy inferred from appearance or shared elicitation prompt.

Sound profile preserves dental c/j versus palatal č/ǰ, aspiration, source vowel quantities,
and unusual symbols; normalizes ɡ and š and composes combining sequences. Original and
Phonemic retain the source's NFC transcription. Native is empty. Immutable pinned local
keys use village tuple, prompt and variant position; duplicate responses and prompt-specific
homonyms are retained. Later source revisions must reconcile those keys before changing
variant order. Source attribution uses `sdml2026` with locality/prompt/variant locators.

## Audit and validation

Seeded 20-record raw/output review: 0 material errors. First/last sites, malformed tokens,
missing markers, duplicate variants, absent frequency counts and erroneous frontend kinship
labels inspected. Glosses follow export keys/headings, with distinct stimulus locators.
Focused tests and build results are recorded below after completion.

Browser refresh is not activated by this ingestion request. No browser database build,
release, commit, push or deployment. PDF/OCR, etymological-reference matching, grammatical
tag extraction and graph-parent resolution are inapplicable to this lexical export.

### Completed validation

- Focused importer, corpus-profile and dialect tests: **22 passed** (25.49s).
- All seven data-generation stages of `make all` completed. The final separate
  manual-survey check failed on the two already-observed tests for Rajasthani
  source counts and duplicate source-owned overlay links. See build log.
- Final forms: **691,930 → 739,247**, exactly +47,317. All old rows/IDs unchanged.
  All new records are `unlinked`; all Original/Phonemic/Form/Gloss layers match
  importer and profile expectations; zero replacement characters/conversion errors.
- Source keys, aliases and durable identity rows each increase by exactly 47,317;
  no durable IDs lost. Edges and alignments are byte-for-byte unchanged.
- References: +1 (`sdml2026`), correct licence, coverage, editor and no OCR.
- Computed concept links: +20,184 net; catalogue 3,275 → 3,271. Existing lexical
  rows/glosses are unchanged. pysem's set-based matching lacks a concept-ID tie
  breaker for equal-scoring matches, so the complete rebuild changes some existing
  concept assignments; details are in the validation JSON. No lexical or graph
  claims were changed to force concept-count stability.
- Representative compiled entry: `f_2de45jadlcxlm`, **waṭi**, “utensil used for drinking
  water”, Gadmudshingi, Karvir, Kolhapur; source locator includes utensil/variant 1.
  It will be available at `/entries/f_2de45jadlcxlm` after a requested browser refresh;
  source reference `/references/sdml2026`. Browser rendering has not been claimed.
- Default full pytest collection stops on the existing duplicate
  `test_preintegration_contract.py` basename (SIL Bhumij/Noira). Full execution
  retried using `--import-mode=importlib`; result recorded below when finished.

Full suite with importlib: **1,762 passed, 18 skipped, 31 failed** in 552.77s.
All 31 failing test names occur in earlier ingestion logs (western surveys, Kharia,
Mewari or Vedda); the Shackle failure absent from the western-survey log is present
in the other earlier logs. No SDML test failed. Existing forms, edges and alignments
also match the immediate pre-ingestion baseline exactly. The failure-name comparison
is saved in `audits/20260911-sdml-test-comparison.json`.

The source is installed and source-specific validation passes. Repository-wide green
validation remains blocked by these pre-existing failures; the checklist's full-suite
gate is explicitly left open rather than reporting an entirely clean ingestion.

## Requested browser refresh — 2026-09-11

User requested rebuild and local serving. Transform and staging succeeded. SQLite integrity
check passed and staged Zstandard decompresses to the identical SQLite image. v33: download
45,313,027 bytes (<50 MB), expanded 102,776,832 bytes. All 47,317 distinct SDML locators
survive browser compaction into 8,175 display forms; 271 source dialect records are present.
Browser source search, sample entry, source map/filter panel, Marathi metadata and LATCH
concept page verified. Local Vite server: http://127.0.0.1:5193/ (session 41140).
`npm run check`: 0 errors, 7 existing warnings. No production deployment.
Details: `audits/20260911-sdml-browser-db.json`.
