# Retrospective source-ingestion validation

Grammar-tag reparsing was validated 2026-08-23 against the canonical checklist. This record
applies to every unit in `source_checklists/manifest.json`; each source-specific checklist links
here for the repository-wide gates. The browser QA below records the prior 2026-08-21 run and was
not repeated because browser refresh remains user-triggered.

Data pipeline: BLOCKED after successful CLDF generation, reference linking, and graph unification.
The pre-existing retired Chattisgarhi identity `f_m5cb2il6e7l4k` remains referenced by
`data/etymology-assignments.csv`, so `assign_form_ids.py` stops before concept, alignment, and
reference regeneration. This same unrelated blocker was already recorded in the Khowar checklist.

- The grammatical-tag focused suite passes **44 tests** across the shared parser, Andersen,
  Sigiri, Thari, Berger, and source-checklist audit.
- The full suite reached **521 passing, 10 skipped, and 23 failing** against the intentionally
  incomplete pre-ID CLDF tree. Twenty-one failures are downstream stable-ID/edge checks caused by
  the blocker; the two pre-existing count failures remain 595 versus 597 marked origins and 14
  versus 19 Backstrom cloth forms.
- The 54 originally tag-empty units now contain 7,658 compiled grammatical-tagged rows from 8,155
  checked source-evidence rows. Only Dhivehi, Kvari, and Arora remain empty, with no source-supplied
  grammatical labels detected.
- All 99 installed ingestion units have non-empty forms, registered language/dialect identifiers,
  resolved bibliography keys, explicit sound-profile routes, audits, and focused regression
  coverage.
- `source_checklists/installed-record-audit.csv.gz` accounts for all **530,597** installed input
  rows with deterministic per-row SHA-256 digests.

Prior browser database and QA (2026-08-21): PASS with a non-blocking size warning

- The user-triggered refresh completed with 480,303 compact lemmas, 423 references, and 1,556
  cross-family comparisons.
- The resulting database is 83,017,728 bytes, 17,728 bytes above the 83,000,000-byte warning
  threshold; the builder reports this as a non-blocking size regression warning.
- SQLite `PRAGMA integrity_check` returned `ok`; `PRAGMA foreign_key_check` returned no rows.
- `npm run check` completed with **0 errors** and 6 pre-existing Svelte warnings.
- `scripts/test_build_static_db.py` passed all 8 tests, including the source-defined reconstruction
  record guard and the independently sourced identical-form merge with retained dialects,
  citations, aliases, and same-lect homonyms.
- All 99 source units occur in the compact browser DB; there are no source units without browser
  rows.
- The staged database was served from a fresh dev process. Final browser-console
  inspection found no errors.

## Representative browser QA

- Merriam source: `/references/merriam2026dravidiandb` renders the DOI, extraction scope,
  provenance links, all 6,672 records, and the expected six unetymologised reconstructions.
- Reconstruction level: `/entries/f_id647ogakr5je` renders Proto-Kurukh–Malto `*āq-` ‘to know’,
  its DEDR 17 parent, exact Merriam record locator, and Starostin attribution.
- Unlinked reconstruction: `/entries/f_kjmhk2nhrvvnu` renders Proto-South Dravidian II `*nā`
  ‘obl of 1sg’ with the source's DEDR 0 locator and no invented parent.
- Proto-language registry: `/languages/PKMDr` renders Proto-Kurukh–Malto in the Northern
  Dravidian clade with exactly 184 forms. Browser-console inspection found no errors or warnings.

- Torwali source: `/references/torwali2023student` renders the complete 2023 dictionary citation,
  snapshot digest and provenance, and reports all 1,943 installed forms plus the 326 audited
  pronunciation-less exclusions.
- Torwali dialects: `/languages/Tor` renders the source-specific `Bahrain (Torwali 2023)` and
  `Chail (Torwali 2023)` varieties separately from their SSNP namesakes, with 1,922 and 21 forms
  respectively. `/entries/f_sx6v4bpgws62y` displays *āfɣanistān* with noun, proper-noun, and
  Bahrain tags; `/entries/f_6bl4nlydtjcfk` displays *bʰumūdūr* ‘Bridegroom’ with noun and Chail
  tags. Browser-console inspection found no errors.

- Cross-family DEDR claim: `/entries/d50` displays “possibly borrowed from” CDIAL 1347 with medium
  confidence and its DEDR entry locator; the old `/Probably < IA` prose is absent from the reflex
  table.
- Cross-family CDIAL claim: `/entries/6087` displays “possible source of” DEDR 3559, while the
  reverse endpoint `/entries/d3559` displays “possibly borrowed from” CDIAL 6087.
- Uncertain comparison: `/entries/d126` displays “possible loan connection (direction unclear)”
  with low confidence, preserving the source's stated uncertainty rather than inventing a donor.
- Ordinary entry: `/entries/5131` renders *jambú*, its gloss, source, and derived terms.
- Borrowing chain: `/entries/n2571` resolves its durable alias and shows Proto-Nuristani
  `*iamarā` borrowed from Indo-Aryan `yamarāja`, including ancestry and alignment.
- Variant: `/entries/d4993-34` resolves to `muṛ̆uku`, marked as a variant of `muṛ̆ku` and linked
  through Proto-Dravidian `*muẓ-u-nk`.
- Alternate etymology: `/entries/10049` shows its accepted Proto-Indo-Iranian parent and the
  separate “also proposed” Indo-Aryan analysis, plus 64 reflexes across 46 languages.
- Unlinked OCR entry: `/entries/f_zgcmreutdcjxa` renders Ghatage `ādne dēsɨ` ‘to order’ with the
  OCR warning, verb tag, `Marati of Kasargod` dialect tag, and `ocr-review` tag.
- Ghatage source: `/references/ghatage-kasargod1970` identifies optical character recognition,
  renders 1,247 compact forms, supports word and `ocr-review` tag filtering, and exposes the
  exact `p. 137, entry 37` locator for the representative entry.
- Affected language/dialect: `/languages/M` renders `Marati of Kasargod` with 1,247 forms,
  quality B, and its Kasargod district description. A dialect map point is intentionally not
  asserted because the source does not supply a defensible elicitation coordinate.
- Concept membership: `/concepts/2398` reports MANGO with 202 forms and 98 unetymologised
  attestations, including Ghatage `ambo`.
- Toda: `/references/bhaskararao-toda2025?word=abak` renders the corrected dictionary source and
  the representative `abak` row after client filtering.
- Pinned legacy snapshot: `/references/patyal2` renders 66 forms from eight languages; its first
  page contains 50 linked, cited rows.
- Badaga source: `/references/hockings-pilotraichoor1992` renders the OCR provenance and all
  16,706 source rows, including 3,572 unetymologised rows.
- Reviewed Badaga entry: `/entries/f_hmjkffhyzp44y` renders *agaṭu madilu*, its noun analysis,
  DEDR 4692 link, and exact `p. 5, col. 2` citation.
- Review-pending Badaga entry: `/entries/f_id7i2lzuvr7ec` renders *Edekādu*, preserves source
  *Edeka:du*, carries the uncertainty marker, and links the printed DEDR 448/1438 claims.
- Badaga language: `/languages/Badaga` renders the existing S. Dravidian I registry record and
  does not invent a source dialect from the dictionary's pan-Badaga coverage.
- OCR workbench: `/dev/ocr` reports 9,973 pending and 20 reviewed articles, displays exact source
  crops and calibration fields, and distinguishes accepted from corrected decisions.
- Southworth Marathi: `/entries/f_cxg3ioej4emr2` renders `pʰaḷ` as an uncertain borrowing from
  Proto-Dravidian `*paẓ-V-`, with both the page 9 lexical locator and page 10 distribution locator.
- Southworth Old Marathi: `/entries/f_u4mq6fmjsatzq` renders `mecū`, its page 9 locator, and its
  borrowing chain to DEDR 4722.
- Southworth parent: `/entries/d4004` includes Marathi `pʰaḷ` among 98 reflexes in 24 languages.
- Southworth cross-family claim: `/entries/d1494` renders the paper's high-confidence DEDR 1494
  → CDIAL 3083 loan comparison with exact page/table evidence and a Southworth reference chip;
  `/entries/2639` renders the reverse endpoint as possibly borrowed from DEDR 1109.
- Southworth uncertainty: `/entries/d4004` renders the paper's `phaḷ` / CDIAL 9051 comparison at
  low confidence, preserving the printed controversial-origin marker.
- Corrected comparison: `/entries/5635` displays `taḍāga` ‘pool’, the printed Turner 5634, and the
  typed form-and-gloss correction; `/entries/f_cdqe2wzuzvmme` retains item 11 on Marathi `dāṭ`.
- Southworth source: `/references/southworth2005m` displays the partial page/table coverage,
  author-hosted URL, provenance, editor, and OCR extraction metadata.
- Burrow and Emeneau 1972: `/references/burrow-emeneau1972den1` renders its complete citation,
  DOI/stable URL, non-redistribution statement, provenance, 713 cited forms, and language
  distribution. `/entries/d512`, `/entries/d811`, `/entries/d800`, and `/entries/d4556` display
  the corroborated `iḷusan`, `talay-ēru`, `jicoṇa`, and `boḷi` reflexes with 1972 source chips.
- DEN I dialect/search/concept views: `/languages/Koraga` registers Mudu with coordinates and
  97 reflexes; `/reflexes?word=iḷusan` returns the sole Thiyya `iḷusan` row with merged DEDR,
  DEN I, and Roy citations; `/concepts/1575`, after expanding d22, displays Koraga `hakkaḷa`,
  its Mudu label, and the 1972 source. All inspected pages reported a loaded database.
- DEN II source: `/references/burrow-emeneau1972den2` renders its complete citation, DOI/stable
  URL, non-redistribution statement, provenance, 161 cited forms, and language distribution.
  `/entries/d49`, `/entries/d2121`, `/entries/d2728`, and `/entries/d3523` display `accu`, `koyk`,
  `sūri`, and `tōṛa (tōṛi-)` with the expected 1972 source chip and printed S² locator.
- DEN II homonym/dialect/concept checks: `/entries/d4375` displays Kodagu `pu·ḷɨ` ‘mist’, while
  `/entries/d4322` contains only the unrelated ‘sour’ set and no DEN II citation; `/entries/d4321`
  displays Koraga `boḷa` with its registered Mudu dialect; `/concepts/606` includes d1656 among
  the etyma for GRASS. All inspected pages reported a loaded database and no console warnings or
  errors.
- Buddruss Grangali source: `/references/buddruss-grangali1979` displays all 170 forms, the exact
  Stanford ILL provenance and scan digest, “Optical character recognition (OCR),” and
  “Etymologies: Supplied by the cited source.”
- Buddruss representative entries: `/entries/f_x35yadzjgygak` renders unlinked `ãʦ̣` ‘eye’ with
  Original `ãc̣`; `/entries/f_yrr2plsityoau` renders `gē` linked to CDIAL 4251;
  `/entries/f_ueqsxw4tzj5vs` renders `ṣal` ‘jackal (elicited for fox)’ linked to CDIAL 12578; and
  `/entries/f_3qgptx2uyrkzq` preserves the alternate mouth form `āsṭ` as its own source record.
- Grangali language: `/languages/Ning` renders the existing canonical Grangali registry record.
  No new dialect point is asserted because Buddruss names the same lect plus a consultant/session,
  not a contrasting stable variety with a defensible coordinate.
- DBIA canonical-CDIAL example: `/entries/f_rrab5sdrn3sqs` renders DBIA 1 as a form-less
  Proto-Dravidian loan-set entry with six real Dravidian-language forms and an expanded
  high-confidence “possibly borrowed from” comparison to CDIAL 991 *ahaṁkāra*. The comparison
  preserves the complete printed DBIA evidence and exact p. 9, no. 1 source locator.
- DBIA source-local IA example: `/entries/f_4ndxl2xxmlrm2` renders DBIA 10 as a two-language
  Dravidian set and, after the local dictionary loads, displays the low-confidence comparison to
  source-local Indo-Aryan *hasti-pippali*. Both inspected DBIA pages reported a loaded database.

## Remediation and residual review

- The Torwali PDF yields 2,269 deterministic headword records: 1,943 with explicit IPA are
  installed and 326 without IPA remain audit-only. The source's black `[چ]` marker assigns 44
  records to Chail (21 installed); every unmarked record is assigned to the named
  Sinkaen/Bahrain variety (2,225 records, 1,922 installed). These use source-specific dialect
  IDs rather than reusing SSNP tags. Printed POS labels map to the existing grammatical tag
  scheme; wrapped labels are parsed from their second baseline, while seven genuinely
  source-blank installed POS fields remain uninferred. Canonical Native remains blank because
  the PDF's ToUnicode mapping demonstrably corrupts some rendered Torwali headwords.

- Cross-family extraction installs 604 source-attributed pairs: 475 asserted by DEDR, 122 by
  CDIAL, and seven cross-table loan claims asserted by Southworth. The decision audit retains 914
  DEDR/CDIAL mentions without a conservatively resolvable pair and 10 explicit negative CDIAL
  citations; its deterministic 20-record installed sample has no material errors. All seven
  Southworth pairs are source-image verified; six are explicit high-confidence claims and the
  printed controversial `phaḷ` case is low-confidence.
- DEDR's ordinary reflex output has a net reduction of 360 rows, including all 192 OIA rows and
  other comparison-tail leakage, while retaining legitimate slash-separated lexical material and
  later labelled subsections. Removed 53 Dravidian comparison rows from CDIAL's ordinary reflex
  output. No transcription, language, dialect, sound-profile, or bibliography additions were
  required for these existing sources.
- Removed and audited 14 blank Rajasthani rows and 3 blank Tharu2 rows.
- Kept two replacement-glyph-only Toda heads audit-only; 7,558 readable dictionary records emit
  8,859 installed rows after variant expansion.
- Added a reproducible, SHA-256-pinned snapshot route with stable keys and audits for nine early
  hand-curated sources; repaired the malformed Patyal snapshot record.
- Ghatage is accepted as OCR-derived rather than claimed as perfect transcription: 129 records
  are source-image verified, 1,115 remain explicitly `ocr-review`, one corrupt alternate is
  audit-only, and the deterministic 20-record final sample passes. The bibliography and every
  displayed form are marked as OCR-derived.
- Hockings and Pilot-Raichoor is installed as 9,993 structurally accounted articles expanding to
  16,706 rows. Twenty scan-backed decisions are reviewed, 9,973 remain explicitly queued for
  transcription review, and 93 printed DEDR citations remain visible but conservatively unlinked.
- Southworth pp. 9–10 account for 25 Table 1 records and 23 Table 2 records, emitting 30 Marathi
  and Old Marathi forms, 25 comparison blocks, and seven cross-family comparison rows. All forms
  use printed DEDR borrowing targets; the Turner 5634/5635 correction and 3276–3278 range conflict
  remain explicit and auditable.
- Burrow and Emeneau 1972 is a conservative structural pilot: 1,154 numbered page segments and
  1,324 nested form candidates produce 709 independently DEDR-corroborated installed rows (713
  compiled attestations, 579 merged with DEDR). Its 304 uncorroborated transcriptions, six
  unresolved current-DEDR targets, one unsplit variant field, and all agent running prose remain
  audit-only pending diplomatic review.
- Part II extends that pilot across 119 numbered segments and 448 form candidates. It installs
  159 current-DEDR-corroborated forms (161 compiled attestations); 46 transcription-unreconciled
  DEDS forms, 25 target-unresolved DEDS forms, 152 DBIA loan-entry candidates, and all running
  prose remain audit-only. Printed S² labels are treated as DEN-II sequence numbers, and the
  language/form/gloss homonym guard prevents Kodagu ‘mist’ from being attached to d4322 ‘sour’.
- Buddruss's 167-number Atlas questionnaire installs 170 independent Grangali form records and
  audits 173 records including three explicit non-attestations. Nine difficult readings remain
  visibly uncertain. Only the source's direct assignments to CDIAL 4251 and 12578 become links;
  Turner numbers attached to secondary comparisons remain prose. The scan is not redistributed
  in the repository.
- DBIA now installs 337 form-less Proto-Dravidian grouping entries around 1,694 conservative
  language attestations instead of misclassifying each source article as Indo-Aryan. It emits 328
  typed loan comparisons with full printed evidence: 186 resolve to canonical CDIAL entries and
  142 retain a clearly identified source-local IA comparison term. Nine cross-reference-only
  articles remain comparison-unresolved and are enumerated in the audit.
- No release, deployment, commit, or push was requested or performed.
