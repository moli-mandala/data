# Bajjika, Lindgren and DravLex ingestion review

Checklist active: `SOURCE_INGESTION_CHECKLIST.md`, survey/comparative-table addendum for all sources;
external-dataset and comparative-source addenda for the Dravidian sources. No OCR was used.

## Scope and evidence

- Bajjika: Regmi, Prasain and Regmi (July 2014), *A Sociolinguistic Survey of Bajjika*,
  Tribhuvan University. Pinned 135-page university PDF; Annex D, printed pp. 103–110
  (PDF 115–122), 210 prompts × five sites = 1,050 cells, expanded to 1,208 slash alternatives.
  Every cell occurs in the per-record audit, including its complete unsplit source text.
  Questionnaire Nepali is a prompt, not Bajjika native-script evidence, and is not installed as Native.
  Publicly accessible report, no explicit licence found; lexical facts are extracted and attributed.
- Lindgren et al.: immutable Zenodo record 7668297, CC BY 4.0, `lindgren_2023.tsv`.
  All 5,881 IDs accounted for: 3,754 additional records installed and 2,127 reused DravLex
  records matched by source language, explicitly reconciled concept labels, and identical
  segmented IPA. The reused IDs and complete records remain in the audit, and their citations
  are attached to the corresponding DravLex observations. They are not independent fieldwork.
  Actual file has 243 concept labels and 34 doculects, despite the description's 231-item design.
- DravLex: pinned Git commit `37578075e5ccb09c43022f7f1282125e748de84d`, CC BY 4.0.
  All 2,127 forms, 100 concepts and 20 source varieties installed. Forms, parameters,
  language metadata, bibliography, transcription report and expert cognate annotations cached.
  `cognatesets.csv` does not exist upstream; `cognates.csv` and form Cognacy supply the annotations.
- SHA256 and acquisition URLs are recorded in `ia_dravidian_2026/snapshot.json`.
  Reproduction is offline with `data/other/forms/raw_data/ia_dravidian.py --output DIR`;
  `--install` installs the reviewed proposal. Bajjika extraction requires pdfplumber.

## Editorial decisions

- Bajjika sites are registered below existing Maithili (`Mth`), with Bajjika identified in each
  display label and Glottocode `bajj1234`; no claim that the survey itself uses the same language
  classification. Table 2.3 supplies exact site GPS, quality A.
- Existing canonical Dravidian languages reused. Koya retains the existing `Gondi`/`koya`
  registration. Separate Ollari Gadaba, Ravula and Pattapu base languages added, not forced into
  distinct related languages. Ollari/Ravula coordinates are explicitly quality-C Glottolog
  reference points; Pattapu has no invented coordinates.
- Ande, Mudu, Onti and Tappu Koraga, Byari and Madhwa Brahmin Tulu are named varieties.
  Other dictionary/fieldwork samples remain provenance, not artificial geographic dialects.
  No coordinates are invented for these named varieties.
- Bajjika SIL Doulos CID 1 decoded as combining tilde, CID 2 as ŋ, CID 3 as ɔ;
  Times New Roman private-use underdots decoded to combining dot below. Raised h distinguished
  from baseline h through font size. The item-137 continuation on the next page is rejoined.
  Word-internal line wraps removed; reviewed phrase-boundary exceptions retain spaces.
  Source capitalization and inconsistent aspiration are preserved in Original. The display
  profile maps c/j to č/ǰ and preserves ambiguous source distinctions rather than inventing IPA.
- Dravidian Original preserves the source Value/IPA field; Phonemic retains the upstream
  segmented analysis with word boundaries. Display conversion uses that analysis, with explicit
  affricate, length, gemination, retroflex and glide mappings. Less familiar segments are preserved.
  Upstream segmentation can omit source stem hyphens; those remain visible in Original.
- All observations remain unlinked. Source-local cognate sets are retained in Cognateset and
  raw annotations; no reconstructed form or direction of ancestry is inferred from group membership.
  DravLex loan labels become `loanword`; unspecified donors remain unspecified. First-edition
  DED numbers in annotations are not silently interpreted as revised DEDR numbers.
- Source-defined records, homonyms and prompt distinctions retain unique keys. Bajjika source
  anomalies such as `dur` under 'near' remain attributed and carry gloss uncertainty.
  Exact repeated alternatives remain source-defined separate records.

## Review and validation

- Seed 20260912: 20 output records per source compared with raw evidence, 0/20 material errors
  per source. Bajjika sample compared to rendered PDF row crops; additional first/last pages,
  legacy symbols, superscripts, phrase boundaries and the split item 137 inspected.
- Focused importer/profile/dialect checks: 28 passed. Reproduction, 210 complete prompts,
  unique keys, language registration, all 2,127 reuse matches, profile coverage and difficult
  glyph mappings tested.
- All seven CLDF build stages completed; `errors.txt` is empty. The final `make all`
  manual-etymology check has two existing failures, also present in the preceding SDML build.
  Gate 12 remains open until the repository-wide baseline is clean.
- Full suite with `--import-mode=importlib`: **1,773 passed, 18 skipped, 31 failed** in 539.13s.
  The 31 failing test names exactly match the preceding SDML run; there are no new failing
  test names. The ordinary pytest command still stops at the existing duplicate-module
  collection collision between the Bhumij and Noiri preintegration tests. Logs and the
  machine-readable comparison are in `audits/20260911-ia-dravidian-*.log` and
  `audits/20260911-ia-dravidian-test-comparison.json`. Pattapu is explicitly allowed to lack
  coordinates; the coordinate test's remaining six exceptions are the same baseline languages.
- Compiled forms: 739,247 → 746,336, exactly 7,089 additions. No previous form ID was lost,
  and no existing form row changed. Edges and alignments are byte-for-byte unchanged.
  Source keys, ID aliases and durable form identities each gained 7,089 rows with no losses
  or changes to their existing rows. All new source entry keys survive in compiled CLDF.
- All new records remain unlinked; 141 DravLex records retain the source's loanword label
  without an inferred donor. Concept mapping assigns 6,942 records to 7,318 memberships
  across 331 concepts, with no invalid target IDs. The remaining 147 source glosses are
  retained without an automatic concept match.
- All six directly cited bibliography keys resolve to formatted references with provenance,
  editor attribution and OCR=No. Full compiled evidence is in
  `audits/20260911-ia-dravidian-build-validation.json`.
- Browser refresh is not part of this ingestion request. No browser database was rebuilt;
  new app examples require the user's next explicit refresh request.

No commit, push or deployment requested or performed.

The pinned Lindgren thesis, printed pp. 27–29, independently confirms the source inventory
and explicitly warns that the exploratory distances do not establish historical relationships.
The released `F_Tulu_New` label has no clear counterpart in thesis table 4; it remains a
source doculect with no invented locality/social classification. All DravLex per-form expert
annotations are joined into its audit, including doubt, comments, and first-edition DED citations.
`sample-review.json` records the 60 reviewed entries and residual source-level limitations.

## Representative compiled entries

These are CLDF IDs; the currently served browser database predates this import.

| Source | Form ID | Language | Display | Source transcription | Gloss |
|---|---|---|---|---|---|
| Bajjika | `f_zearherxfrgck` | Mth, Bajjika site | deh | deh | body |
| Lindgren | `f_i6pyihbkicwro` | Koraga | iǰǰi | id͡ʒ:i | second-person singular, female |
| DravLex | `f_icy3paasnanja` | Badaga | na | na | I |

## Applicability and exclusions

- Survey/comparative-table addendum: applied to all three sources. Bajjika's English/Nepali
  prompt columns are concept evidence, not additional target-language observations.
  Nonlexical report chapters are outside the wordlist scope. No target wordlist cells were excluded.
- External-data addendum: applied to Lindgren and DravLex, with pinned snapshots and upstream IDs.
- Comparative-source addendum: group labels and expert annotations retained conservatively;
  no guessed ancestor nodes or DEDR links. Lindgren's 2,127 republished records are audit-accounted
  and cited on DravLex observations rather than installed again. Exact duplicate source analyses
  can match multiple preserved source-defined rows; their audit lists every target entry key.
  This occurs for six Lindgren IDs: two identical Kannada analyses each for hair, head and
  stone. DravLex distinguishes Andronov and fieldwork attestations, so both rows remain;
  the republished citation is attached conservatively to both matching targets.
- Dictionary/OCR addenda: not applicable. Bajjika uses its selectable embedded-font text layer;
  no OCR was used. No source supplies a separate target native-script layer here.
- New clades, new canonical grammar tags, inferred donor links and deployment: not applicable.
- Browser construction and QA: not applicable until explicitly requested, per checklist section 13.
