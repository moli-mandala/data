# Retrospective source-ingestion validation

## SIL Pahari/Pothwari survey — 2026-08-28

- Source identity and completeness: PASS. Michael and Laura Lothers' 2010 *Pahari and
  Pothwari: A Sociolinguistic Survey* (SIL ESR 2010-012) is represented by the exact
  262-page official publisher PDF recovered from Internet Archive capture 2014-09-07
  (SHA-256 `e3695a807c4856118303eca74b68b192817ea69251fa8be62abb7b27e4c1ad6f`). Appendix
  B.1 spans physical PDF pages 153--208 (printed pages 147--202). The frozen positioned-text
  snapshot deterministically accounts for 217 prompts, 16 lists and all 3,472 cells.
- Inclusion policy: PASS. All 3,038 nonblank responses from fourteen target Pahari, Pothwari
  and Mirpuri locality lists install with immutable per-response keys. The complete audit retains
  and excludes 434 Abbottabad and Mansehra Hindko control cells, including eighteen source blanks
  at items 209--217. Eleven asterisked prompts remain lexical attestations but are explicitly
  marked as excluded from the report's lexical-similarity calculation. The corrected compiled
  CLDF preserves all 3,038 source locators as 3,038 distinct site attestations; a source-specific
  merge guard prevents identical shapes from silently losing their dialect provenance.
- Transcription and structure: PASS. No OCR was used. Appendix B.1 has a complete positioned
  Doulos SIL text layer, and content-stream order preserves the report's Indological Phonetic
  Script, length, nasalisation, palatalisation and breathy-voice marks. The literal `AUS` code
  printed fourteen times in the invariant ninth (Osia) row is retained in the snapshot and
  explicitly normalized to `OSI`; no source cell or symbol remains unparsed. The report's
  phonetic lexical-similarity judgements do not assert historical cognacy, so no graph edges were
  inferred.
- Language and dialect metadata: PASS. All target responses use canonical Pahari-Pothwari
  `poth` (`paha1251`), and all fourteen locality lists have language-qualified dialect tags with
  the report's reliability grade (eight A, six B). The report supplies regional maps but no point
  coordinates, so the site rows deliberately remain unlocated instead of inventing modern points.
- Focused validation: PASS. Eight source-specific tests plus both exhaustive profile/routing
  checks pass (**10 passed**), including exact 217-response compiled coverage for every target
  site. The complete data pipeline carries 3,038 rows through conversion, reference linking,
  graph unification, 623,639 durable IDs, 3,254 concepts, alignment and 504 generated references;
  it stops only at the pre-existing stale Kannauji expectation (1,991 accepted links versus 1,985
  expected). The final repository-wide suite completed with **1,054 passed, 10 skipped and 26
  failed** in 748.22 seconds. None of those failures is in the Pahari/Pothwari source tests; they
  are the already-recorded cross-source/build baseline (including stale Kannauji counts, legacy
  compiled-ID assumptions, global metadata expectations, Mudhili profile punctuation, and the
  source-checklist registry). Checklist gate 12 remains formally open because that repository
  baseline is not clean.
- Browser QA (20260828-sil-pahari-pothwari): PASS. The freshly rebuilt compact database contains
  568,165 post-merge lemmas, 665,657 lemma-reference links, 398 languages and 504 references;
  SQLite `PRAGMA integrity_check` returns `ok`, all eight browser-builder tests pass, and Svelte
  reports 0 errors with 7 existing warnings. `/references/lothers-lothers2010pahari` renders the
  exact source identity, provenance and all 3,038 cited forms. Exact search for `vǎ̤` returns one
  Dunga Gali record and visibly preserves the source transcription, gloss `wind`, and printed
  page 160 item 51. `/languages/poth` renders all fourteen ESR 2010-012 sites at exactly 217 forms
  each, with the intentional blank-coordinate metadata. The browser console has no warnings or
  errors (Vite connection debug messages only). The 112.0 MB database exceeds its historical
  83 MB warning threshold but is structurally valid.
- No release, deployment, commit, or push was requested or performed.

## SSNP volume 4: Pashto, Waneci, Ormuri — 2026-08-28

- Source identity and completeness: PASS. Daniel G. Hallberg's 1992 *Sociolinguistic Survey
  of Northern Pakistan, Volume 4. Pashto, Waneci, Ormuri* is represented by the exact
  194-page official SIL publisher PDF recovered from Internet Archive (SHA-256
  `83e2d833c06ecb4e40bfb0d316061d6b398b743bac299dc870c90c88a4b96f18`). Appendix B's
  inventory and lexical tables span physical PDF pages 97--164 (printed pages 79--146). The
  frozen positioned-text snapshot deterministically accounts for 200 printed prompts, 36 lists,
  7,200 cells and all 42 geometrically joined continuation lines.
- Inclusion policy: PASS. All 7,131 lexical responses from 34 Pashto locality lists plus the
  Waneci and Ormuri lists install with immutable entry keys. Sixty-eight cells printed `--` and
  the blank Bannu item 135 cell remain audit-only. The ten absent prompt numbers are explicitly
  excluded by the source. The unified CLDF preserves all 7,131 source locators on 2,723 safely
  coalesced reader-facing citation rows.
- Transcription and structure: PASS. No OCR was used. The complete SILDoulosNP text layer is
  decoded against the report's printed phonetic chart and high-resolution page renders, with
  content-stream order preserving overstruck diacritics. The volume-specific mappings retain
  `ɣ`, `ɸ`, retroflex `ɭ`, length, nasalisation and underdot distinctions; all 7,131 installed
  forms have explicit profile coverage and no replacement character. The source's lexical-
  similarity classifications are phonetic comparisons, not historical cognacy, so no
  etymological edges were inferred.
- Language and dialect metadata: PASS. Pashto and Ormuri reuse their canonical language rows;
  Waneci is registered as `wne` (`wane1241`). All 36 lists have language-qualified dialect tags,
  source subgroup and reliability metadata. Their map points are explicitly labelled modern
  locality or regional approximations rather than source coordinates; the sole list without a
  printed reliability code is quality C.
- Focused validation: PASS. Eight source-specific checks plus both exhaustive profile/routing
  checks pass (**10 passed**). The complete data pipeline carries all responses through
  conversion, reference linking, graph unification, durable IDs, concepts and 503 generated
  references; it stops only at the pre-existing stale Kannauji expectation (1,991 accepted links
  versus 1,985 expected). The repository-wide run completed with 1,045 passed, 10 skipped and 27
  failed before the final SSNP profile fixture correction; that source-local correction now
  passes, while the remaining failures are the previously recorded unrelated baseline. Checklist
  gate 12 therefore remains formally open.
- Browser QA (20260828-ssnp04): PASS. The freshly rebuilt compact database contains 565,127
  post-merge lemmas, 662,619 lemma-reference links, 398 languages and 503 references; SQLite
  `PRAGMA integrity_check` returns `ok`, all eight browser-builder tests pass, and Svelte reports
  0 errors with 7 existing warnings. `/references/hallberg1992pashto` renders the full source
  identity, provenance, three-language distribution and 2,723 cited forms. Exact search for
  house form `ɣãṇye / ɣãṛye` returns one Tirah Afridi record and visibly preserves source IPA
  `/ɣʌ̃ɳye / ɣʌ̃ɽye/`, gloss `spider`, and printed page 111 item 100. `/languages/Psht` renders
  all 34 new Pashto survey localities among 41 current dialects, and `/languages/wne` renders the
  199-form Harnai Waneci list. The browser console has no warnings or errors. The 111.1 MB
  database exceeds its historical 83 MB warning threshold but is structurally valid.
- No release, deployment, commit, or push was requested or performed.

## SIL Lahul Valley survey — 2026-08-28

- Source identity and completeness: PASS. SIL ESR 2019-006, *A Sociolinguistic Survey of
  Lahul Valley, Himachal Pradesh* by Brad and Wendy Chamberlain, is represented by the exact
  185-page official publisher PDF recovered from Internet Archive capture 2024-06-16 (SHA-256
  `17f8178505ef88879baecbd5d9fa6dd4f2bb885330722cbac21df70c71e47252`). Appendix A.4 spans
  physical PDF pages 46--87 (printed pages 38--79), with 210 concepts, 27 lect/site lists and
  6,206 audited response records. The frozen positioned-text snapshot and parser reproduce all
  records deterministically, including ten visually checked wrapped responses.
- Inclusion policy: PASS. All 5,027 nonempty responses from the 22 newly collected Chinali,
  Lahul Lohar, Pattani, Tinani, Bunan/Gahri and Bhoti lists install with immutable entry keys.
  Twenty-nine target cells printed `no entry`, 474 Standard Hindi/Lhasa Tibetan controls, and
  676 previously collected Tindi Pangi, Leh Ladakhi and Mane Spiti Bhoti responses remain
  audit-only. The unified CLDF preserves the new source as 3,293 reader-facing citation
  attestations after safe identical-attestation coalescing.
- Transcription and structure: PASS. No OCR was used: Appendix A.4 has a complete Unicode
  Charis/Doulos SIL text layer. Original and Phonemic preserve the field IPA, while
  `conversion/sil-lahul.txt` supplies a fully covered house display transcription with no
  replacement characters. The report's lexical-similarity group labels remain source notes and
  do not assert historical cognacy; the report's warning that the transcriptions have not
  undergone thorough phonological analysis is retained on every installed response.
- Language and dialect metadata: PASS. The six canonical language rows use current ISO and
  Glottolog identifiers where available, and all 22 survey lists are registered as
  language-qualified dialects. The report provides regional maps but no point coordinates, so
  each locality uses an explicitly labelled modern GeoNames, OpenStreetMap, or government point
  at quality C rather than presenting it as a source coordinate.
- Focused validation: PASS. The compiled Lahul module passes all eight tests, and the earlier
  extraction/profile run passes nine targeted checks. The complete data pipeline carries all
  5,027 input rows through conversion, reference linking, graph unification, durable IDs,
  concepts, 1,981,819 aligned segments and 502 generated references; it stops only at the
  terminal pre-existing Kannauji expectation (1,991 accepted links versus 1,985 expected).
  The full repository suite finishes with **1,038 passed, 10 skipped and 26 failed** in 693.90
  seconds. None of the failures selects `tests/test_sil_lahul_2019.py`; they concern unrelated
  legacy identity/count fixtures, older metadata expectations, incomplete retrospective profile
  fixtures, and the known Mudhili Gadaba `talːu` versus `tallu` assertion. Checklist gate 12
  therefore remains formally open despite the verified Lahul compiled output.
- Browser QA (20260828-sil-lahul): PASS. A fresh compact database contains 562,404 post-merge
  lemmas, 655,488 lemma-reference links, 397 languages and 502 references; SQLite
  `PRAGMA integrity_check` returns `ok`, all eight browser-builder tests pass, and Svelte reports
  0 errors with 7 existing warnings. `/references/chamberlain-chamberlain2019lahul` renders the
  full scope, provenance, six-language distribution and 3,293 cited forms. `/languages/lae`
  renders 1,067 Pattani forms and all eight quality-C locality records. Searching normalized
  `rəṇj.kriṇj` returns exactly one record and visibly preserves source IPA
  `/ɾəɳdʒ.kɾɪɳdʒ/`, gloss `spider`, Gushal-Pattani, and printed page 57 item 100.
  The browser console has no warnings or errors. The 109.7 MB database exceeds its historical
  83 MB warning threshold but is structurally valid.
- No release, deployment, commit, or push was requested or performed.

## SIL Malvi survey — 2026-08-28

- Source identity and completeness: PASS. SIL ESR 2009-011, *The Malvi-speaking People of
  Madhya Pradesh and Rajasthan: A Sociolinguistic Profile* by Bijumon Varghese, Mathews John,
  and Nelson Samuel (2009), is represented by the exact 280-page official publisher PDF recovered
  through Internet Archive capture `20150930085157` (SHA-256
  `e67e314974ab10eb8244b08dba56d08d1ce8cbf16eaef1be022071d49032a2dd`). The frozen
  snapshot contains all 8,798 printed Appendix B rows plus 114 explicit matrix cells for the three
  prompts removed by the source, for 8,912 audited records total.
- Inclusion policy: PASS. All 6,894 lexical responses from thirty target Malvi lists install with
  unique immutable entry keys. The audit retains and excludes 1,891 records from two Bhili, two
  Nimadi, Bhopali, Hindi, Gujarati and Marathi comparison/control lists, 37 target cells printed
  `By Name`, and 90 target cells for items 11, 23 and 24 which the source disqualified. The compiled
  CLDF safely coalesces identical attestations into 2,128 reader-facing rows while preserving all
  6,182 distinct source locators and all thirty dialect tags.
- Transcription and structure: PASS. No OCR was used. The embedded Type0 SAG-IPA font lacks a
  ToUnicode map, so 31 of its 34 used CIDs were identified from the report's printed IPA chart and
  the remaining `ɠ`, literal circumflex and combining square below from SIL's official
  `SAGIPA2Uni.map` plus rendered-page checks. Geometric extraction preserves raised `ʰ`, `ʱ`, `ⁱ`,
  `ᶦ`, and `ᵘ` and the below-line dental and square diacritics. No CID or replacement character
  remains. The later Unicode Thillorkhurd reprint in ESR 2012-002 independently agrees on 132
  responses across 126 concepts.
- Language and dialect metadata: PASS. The thirty lists are registered beneath Jambu's existing
  canonical Malvi base ID `mewari_basad` (`malv1243`) with source subgroup, locality,
  administrative unit and WordSurv code. The report supplies no point coordinates, so their
  coordinates remain blank rather than receiving invented centroids.
- Focused validation: PASS. Seven source-specific checks and three direct profile checks pass.
  The complete data pipeline succeeds through conversion, reference linking, graph unification,
  durable IDs, concepts, 1,981,831 aligned segments and 500 generated references. It then stops
  only at the pre-existing stale Kannauji assertions (1,991 accepted links versus 1,985 expected);
  no Malvi row appears in `errors.txt`. The full sound-profile test still includes the independently
  known Mudhili Gadaba `talːu` versus `tallu` baseline failure. These unrelated repository issues
  leave checklist gate 12 formally open.
- Browser QA: PASS. The rebuilt 108,625,920-byte compact database returns `ok` from SQLite
  `PRAGMA integrity_check`; all eight browser-builder tests pass; Svelte reports 0 errors and 7
  existing warnings. `/languages/mewari_basad` renders 35 current dialects, including all thirty
  new source localities. `/references/varghese-john-samuel2009malvi` renders the full
  inclusion/provenance and 2,128 coalesced source forms. Searching `khailo` returns the item-182
  entry with Original `/kʰʌⁱlo, ɠʰajlijo/`, Harsodan-Ujjaini, category note and printed p.149
  locator; `/entries/f_mkr5grtwox7mu` renders the same entry. The database exceeds its historical
  83 MB warning threshold but is structurally valid.
- No release, deployment, commit, or push was requested or performed.

## SIL Nimadi survey — 2026-08-28

- Source identity and completeness: PASS. SIL ESR 2012-002, *The Nimadi-speaking People of
  Madhya Pradesh: A Sociolinguistic Profile* by Kishore Kumar Vunnamatla, Mathews John, and
  Nelson Samuvel (2012), is represented by the exact 176-page official publisher PDF recovered
  through Internet Archive capture `20170810011221` (SHA-256
  `1a7e8daaeb2b967e2f9490292689e33a188caf47dc262c942a47136bb270d0d8`). The frozen
  positioned-text snapshot deterministically represents 4,019 printed response records, 72
  audit cells for the four standard prompts absent from the published appendix, and one target
  cell whose response row is absent, for 4,092 audited records total.
- Inclusion policy: PASS. All 2,826 nonempty target attestations from thirteen Nimadi lists
  install with unique immutable entry keys and survive as exactly 2,826 compiled citation rows.
  The 1,207 records/cells belonging to the Parya Bhilali, Malvi, Hindi, Gujarati and Marathi
  comparison lists remain audit-only, along with five target `no entry` records, 52 target cells
  for prompts 11, 23, 24 and 70 absent from the appendix, and two target cells without primary
  forms. The compact browser presents 1,236 source forms after safe identical-attestation
  coalescing while retaining all citations, dialect tags and ID aliases.
- Transcription and structure: PASS. No OCR or legacy-font reconstruction was needed: Appendix A
  has a Unicode Doulos SIL text layer. The importer handles its parity-shifted three-column
  geometry and the two-column long-predicate pages 95--99. Original/Phonemic retains the printed
  IPA, and `conversion/sil-nimadi.txt` provides a fully covered display transcription with no
  replacement characters. Three page-render checks document the fused category digit on item
  13, the blank primary plus category-2 alternate on item 40, and a spurious `(cid:1)` text-layer
  fragment absent from Gujarati item 98. Literal source capital `B` spellings remain in Original
  and normalize only in display. Similarity categories remain Notes, not cognacy claims.
- Language and dialect metadata: PASS. All thirteen lists are registered as language-qualified
  dialects beneath canonical Nimadi (`nima1243`) with the report's locality, administrative,
  community and WordSurv-code metadata. The report gives no defensible point coordinates, so
  coordinates are intentionally blank and the browser transparently reports all thirteen new
  survey dialects as not located (alongside the three pre-existing ESR 2018-011 Nimadi lists).
- Focused validation: PASS. Seven source-specific checks and four relevant profile/routing checks
  pass after the final build (**11 passed**). The complete data pipeline succeeds through
  conversion, reference linking, graph unification, durable IDs, concepts, 1,981,831 aligned
  segments and 499 generated references. It then stops only at the pre-existing stale Kannauji
  assertions (1,991 accepted links versus 1,985 expected); no Nimadi record appears in
  `errors.txt`. This unrelated repository baseline leaves checklist gate 12 formally open but
  does not affect the verified source extraction or compiled identity.
- Browser QA (20260828-sil-nimadi): PASS. The rebuilt 107,757,568-byte compact database returns
  `ok` from SQLite `PRAGMA integrity_check`; all eight builder tests and three Etymology Lab tests
  pass; Svelte reports 0 errors and 7 existing warnings. `/languages/Nimadi` renders all sixteen
  current dialects, including the thirteen new source localities and their per-list counts.
  `/references/vunnamatla-john-samuvel2012nimadi` renders the full inclusion/provenance record.
  Exact search returns `bhuklagi, bhuklagtithi` with Original
  `/bʱuklʌgi, bʱuklʌgti̪tʰ̪i/`, Sonipura-Balai, its source category note and the printed p. 94,
  item 184 locator; `/entries/f_l7dtpxqm4h4wy` renders the same unlinked source entry correctly.
  The database exceeds its historical 83 MB warning threshold but is structurally valid.
- No release, deployment, commit, or push was requested or performed.

## SIL Bareli/Pauri survey — 2026-08-28

- Source identity and completeness: PASS. SIL ESR 2018-011, *A Sociolinguistic Study of
  Bareli/Pauri and Related Languages* by Vinod Wilson Varkey and Kishore Kumar Vunnamatla
  (2018), is represented by the exact 197-page publisher PDF recovered from the Internet
  Archive (SHA-256 `02128358a61e175ba2a07b2862f6072167a3609cf71264e235ae21284fe2ceea`).
  Appendix C spans physical PDF pages 87--156 (printed pages 80--149) and contains 210 prompts
  across 33 lists. The frozen positioned-text snapshot and importer reproduce all 7,247 audited
  response records deterministically.
- Inclusion policy: PASS. All 6,320 regional attestations from 30 lists and eight canonical
  languages install with unique immutable keys. The 789 standard-language controls, 105 `NO
  ENTRY` cells, and 33 cells belonging to the source-disqualified `millet` prompt remain
  audit-only. The compiled CLDF retains exactly 6,320 citation-bearing rows; the compact browser
  presents 4,786 reader-facing forms after safe identical-attestation coalescing.
- Transcription and structure: PASS. No OCR was used: the PDF contains Unicode Charis SIL and
  Doulos SIL text whose page positions recover the three-column reading order. The source IPA is
  retained in Original and Phonemic, while `conversion/sil-bareli-pauri.txt` supplies the house
  display transcription with complete symbol coverage and no replacement characters. Two
  literal unmatched `[` characters are preserved rather than conjecturally repaired and carry
  typed uncertainty notes. Four parenthetical semantic annotations are separated into Notes;
  the wrapped two-clause item 184 remains one elicited response.
- Language and dialect metadata: PASS. All eight base languages are registered with verified
  Glottocodes and coordinates, and all 30 survey lists are language-qualified dialects with the
  report's exact locality and administrative metadata. Because the report gives no defensible
  site coordinates, dialect coordinates are intentionally blank and explicitly shown as not
  located (quality C) in the browser.
- Focused validation: PASS. Seven source-specific tests cover the pinned PDF/snapshot hashes,
  complete audit accounting, deterministic rebuild, exact language counts, difficult wrapped and
  uncertain forms, registry metadata, and compiled citation survival. Three relevant
  sound-profile/routing tests pass. The complete data build succeeds through conversion,
  reference linking, graph unification, durable IDs, concepts, alignment, and 498 generated
  references; it stops only at the pre-existing stale Kannauji expectation (1,991 accepted links
  versus 1,985 expected). The known unrelated Gadaba profile expectation (`talːu` versus `tallu`)
  is unchanged.
- Browser QA (20260828-sil-bareli-pauri): PASS. The rebuilt 107.3 MB compact database returned
  `ok` from SQLite `PRAGMA integrity_check`; all eight builder tests pass; Svelte reports 0 errors
  and 7 existing warnings. The source page renders the full citation, provenance, inclusion
  statement and 4,786 coalesced forms. Exact searches render Amalwadi `pats[` / source
  `/pats̪[/` 'five' and Tharadpura `bɦuklu ce, bɦuklu hato` / source
  `/bɦuklu tʃe, bɦuklu hʌtɔ̪/` 'hungry' with exact printed locators. The Rathwi Bareli language
  page renders all seven registered survey dialects and transparently reports that none has a
  source-supplied map point.
- No release, deployment, commit, or push was requested or performed.

## SIL Northern Pakistan Volume 2 — 2026-08-28

- Source identity and completeness: PASS. The pre-existing `20230416-northern.csv` installation
  is the complete CC-BY-4.0 Lexibank v1.1 release of Backstrom & Radloff 1992, not an incomplete
  OCR attempt. Its 11,343 raw and installed records cover 51 varieties and 1,233 parameter rows;
  the checked-in raw forms are byte-identical to release commit
  `377d157614c706b2fcb61eccd5c839f394b9aa6c` (DOI `10.5281/zenodo.13149113`). The release was
  manually digitized from the printed NAPA wordlists. No OCR was required.
- Inclusion policy: all 10,836 target-language records are retained. The 507 controls (Urdu 261,
  Pashto 246) remain installed and auditable but are intentionally excluded from compilation.
  The compiled CLDF contains 10,629 citation-bearing attestations after the established three
  Domaaki predicate splits and other compiler-level compaction. The browser presents 4,620
  reader-facing source forms across Shina, Balti, Burushaski, Wakhi, and Domaaki.
- Transcription: PASS. Upstream/source spellings remain in `Original`; released phonetic analysis
  remains in `Phonemic`; display forms use an explicit NFD sound-profile route. All target rows
  convert with no replacement characters. Reviewed regressions include `ṣ̌ʌq` → `ṣaq`, `ǰ̣oŋ` →
  `ʣ̣oŋ`, and `èí` → `eí`.
- Focused validation: PASS. Six source tests cover pinned hashes and release metadata, complete
  row/control accounting, updated Concepticon mappings, installed-form preservation, the three
  documented Domaaki splits, registry coverage, and exact/zero-replacement conversion.
- Full data pipeline: PARTIAL only at an unrelated repository baseline. Conversion, reference
  linking, graph unification, durable IDs, concepts, 1,981,831 aligned segments, and 497 generated
  references all succeeded. The terminal manual-survey check then stopped on the stale Kannauji
  expectation (1,991 current accepted links versus 1,985 expected). The legacy-restoration suite
  also has the unrelated 4,582-versus-4,586 assignment baseline; neither failure concerns this
  source's extraction, inclusion, transcription, or citation survival.
- Browser/database QA: PASS. The rebuilt 106.1 MB compact database returned `ok` from
  `PRAGMA integrity_check`; all eight builder tests pass; Svelte reports 0 errors and 7 existing
  warnings. The size exceeds the historical 83 MB warning threshold but is structurally valid.
  Exact searches show Balti `ṣaq /ʂɜq/` “blood,” Balti `ʣ̣oŋ /ɖʐoŋ/` “village,” and normalized
  Shina `eí`, each with Backstrom & Radloff citation. The source page and Balti, Burushaski, Shina,
  Wakhi, and Domaaki language pages were inspected successfully.
- No release, deployment, commit, or push was requested or performed.

## Bhattacharya 1968 Bonda Dictionary — 2026-08-28

The non-OCR source package is internally clean. All 2,881 structured SEAlang records are
accounted for: 2,716 Plains Bondo and 165 Hill Bondo records expand to 3,330 installed and
compiled form rows, while one exact repeated alternant is retained audit-only. The dedicated
importer and sound-profile run passed **17 tests** before an unrelated concurrent source appeared;
the importer-specific module continues to pass independently. The stratified 20-record review has
**0 material errors**, the offline rebuild exactly reproduces the installed CSV and audit, all
forms use canonical Remo (`re`) with registered Plains/Hill dialect tags, and no replacement
characters or unresolved citation keys occur.

The source's 550 explicit `ETY:` segments remain source prose rather than conjectural graph
ancestry. Of 36 gloss-level `see` records, 27 unique printed targets receive source-internal
variant links, one multiple-target record receives only the targets' shared gloss, and eight
ambiguous, malformed, or missing targets remain unlinked and glossless. Three Hill Bondo records
printed without a definition also remain intentionally glossless. Question mark is preserved as
a transcription symbol; only three terminal `(E?)` provenance/query markers are removed from
Form, retained in the audit, and tagged `uncertain`.

Data pipeline: the first post-install `make_cldf.py` run completed and carried all 3,330 Bonda
rows. `link_refs.py` and `make_refs.py` also completed; reference generation is clean after adding
the canonical Deccan College imprint. A later repository-wide rebuild is BLOCKED by an unrelated,
concurrently added `20260828-sil-mudhili-gadaba.csv` header row, which is read as a form with
language `Language_ID`; graph unification consequently stops at the synthetic edge
`Parameter_ID-1 → Parameter_ID`. The Bonda package introduced neither identifier. Current global
sound-profile/checklist tests also fail on that new source and other pre-existing SIL/Herin
expectations, while all Bonda-specific assertions pass. The full repository suite finishes with
**902 passing, 9 skipped, and 61 failing** in 281.16 seconds; none of the failures selects
`tests/test_bhattacharya_bonda_1968.py`. Most are expected consequences of the interrupted
pre-unification CLDF tree, with the remaining failures in unrelated source/profile/checklist
fixtures. No browser refresh was requested.

No release, deployment, commit, push, or browser-database refresh was requested or performed.

## Munda 1968 Proto-Kherwarian and Zide 1982 Sora–Juray — 2026-08-28

Both non-OCR source packages are internally clean. Munda contributes 920 Proto-Kherwarian
parameters and 2,919 Proto-Kherwarian, pre-Mundari, and Santali form rows from 2,768 structured
records. Zide contributes 2,057 Sora and Juray form rows from 1,750 records in 1,011 comparison
groups. The pre-unification compiler carries all 4,976 new forms, all 920 parameters, 110 Munda
grammatical-tagged rows, 70 Zide grammatical-tagged rows, and both resolved references. The
combined focused importer, reference, and sound-profile run passes **20 tests**; both offline
rebuilds exactly reproduce their installed files and audits. Neither source uses OCR.

Data pipeline: BLOCKED at the unrelated repository-wide graph-unification gate. `make_cldf.py`,
`link_refs.py`, and `make_refs.py` complete, but `unify_cldf.py` stops because the concurrently
changing Proto-Burushaski catalog names a Berger evidence key not emitted by the current Berger
input (`berger-entry-10004-dialect-1` in the last run). The guard correctly refuses to discard the
claim. The two Munda source checklists therefore leave gate 12 open; all other applicable gates
pass. No browser refresh was requested.

Full test suite: BLOCKED by unrelated repository state, with **903 passing, 54 failing, and 9
errors** in 278.53 seconds. Neither `tests/test_munda_proto_kherwarian_1968.py` nor
`tests/test_zide_sora_gorum_1982.py` appears among the failures. Most failures are consequences of
the pre-unification `cldf/forms.csv` (missing final Status/durable-ID/edge layers); the remaining
failures concern existing Berger/Burushaski, reference validation, SIL profile/checklist
expectations, and legacy source baselines.

No release, deployment, commit, push, or browser-database refresh was requested or performed.

## Buddruss 1996 Shina-Rätsel — 2026-08-28

The source package is internally clean: all 311 installed Shina rows have immutable source keys,
the 7 focused ingestion tests pass, and the Buddruss Shina sound profile converts every form
without replacement characters. The last successful unified build carried all 311 rows with
`buddruss-shina1996` citations. The source accounts for 296 analytical glossary units plus 15
explicit headline alternates; 149 rows carry 138 unique direct CDIAL links. The source's
explicitly unintelligible `wáaku` and all questioned, competing, component-only, and
comparison-only Turner claims remain conservatively unlinked.

Data pipeline: BLOCKED at the repository-wide completion gate. A fresh `make all` on the current
worktree reached graph unification, then stopped because the unrelated Proto-Burushaski catalog
still names three Berger evidence keys (`berger-entry-1000-dialect-2` through `-4`) that the
currently regenerated Berger form file no longer emits. The integrity guard correctly refuses to
silently discard that evidence. The partial rebuild leaves generated CLDF files pre-unification,
so compiled-ID checks are intentionally not treated as current completion evidence. The prior
fully unified build carried all five ILL sources and was the input to the successful browser build.

Full test suite: BLOCKED by 25 repository-level failures, with **912 passing and 10 skipped** in
396.25 seconds. None selects `tests/test_buddruss_shina_1996.py`. The failures concern existing
Berger cleanup fixtures, DEN I/II citation counts, DEDR/PDR and NeoJambu restoration baselines,
Kannauji counts, Gondi and Southworth identity aliases, Zargari expectations, other newly present
SIL source/profile/checklist state, and legacy CDIAL section metadata. They are outside this
source's scholarly scope and were not rewritten as part of the Shina ingestion.

Browser QA (20260828-buddruss-shina-raetsel): PASS with a non-blocking size warning

- A fresh compact database loaded 521,387 lemmas, 596,306 lemma-reference links, 87,526 text
  blocks, 843,038 permanent aliases, 1,556 cross-family comparisons, 3,246 concepts, 25,711 graph
  edges, and 1,101,445 reachable alignment segments.
- The database is 93,749,248 bytes, above the 83 MB warning target. SQLite `integrity_check`
  returned `ok` and `foreign_key_check` returned no rows.
- `npm run check` completed with 0 errors and 7 existing Svelte warnings; all 8 static-database
  regression tests passed.
- `/references/buddruss-shina1996` renders the full citation and scan provenance. The filtered
  `/languages/Sh?source_ids=buddruss-shina1996` view renders 311 forms and supports word filtering.
  Linked `áa~i` 'mouth', unlinked `čhii~ṣ` 'mountain', `agúl`/`hagúl` variants, the combined linked
  alternate `sáa~ity`, and uncertain `wáaku` were inspected. Concept 674 (MOUTH) reports 788 forms
  across 250 languages. The final browser console contains no warnings or errors.

The scan remains outside the repository and is not redistributed. No release, deployment,
commit, or push was requested or performed.

The repository was revalidated 2026-08-25 against the canonical checklist after installing the
Rezai Baghbidi Zargari article. This record applies to every unit in
`source_checklists/manifest.json`; each source-specific checklist links here for the
repository-wide gates. The browser QA below records the prior 2026-08-21 run and was not repeated
because browser refresh remains user-triggered.

Data pipeline: PASS. `make all` completed CLDF generation, reference linking, graph unification,
durable-ID assignment, concept generation, alignment, reference generation, and the mandatory
manual-survey etymology check. The build produced 510,735 durable form IDs, 341,067 typed edges,
320,905 aligned reflexes, 2,072,387 aligned segments, 3,220 concepts, and 428 references, and left
`errors.txt` empty. No durable form ID was retired: the run added 6,111 identities to
`data/form-identities.csv` and removed none.

Full test suite: BLOCKED by the same five repository-level failures recorded on 2026-08-24, none of
which selects the new source. The suite finished with **592 passing, 10 skipped, and 5 failing**:
two stale DEN I/II compiled-citation counts, one duplicated DEDR citation expectation, the
previously recorded Backstrom cloth count (14 versus 19), and one missing compiled PDR audit target
(`d583A`).

- The new Zargari focused suite passes **27 tests**, covering audit completeness, stable span keys,
  page and section locators, dialect registration and coordinates, curation policy (clauses,
  phrases and unglossed paradigms excluded; homographs and printed alternates retained), sound
  profile mappings and complete symbol coverage, compiled CLDF survival of all 522 rows, the 78
  variant edges, and the formatted reference. Its two text-layer tests need the uncommitted
  publisher PDF and skip without it.
- The new source contributes **522** compiled European Romani attestations under the registered
  `dialect:eur:zargari:Zargari` tag, all unlinked, all carrying canonical grammatical tags, and
  none emitting replacement characters or unresolved bibliography keys.
- All 103 installed ingestion units have non-empty forms, registered language/dialect identifiers,
  resolved bibliography keys, explicit sound-profile routes, audits, and focused regression
  coverage.
- `source_checklists/installed-record-audit.csv.gz` accounts for **531,075** installed input rows
  with deterministic per-row SHA-256 digests.

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

# SIL Western Arunachal / Monpa survey — 2026-08-28

- Full pipeline (`20260828-sil-western-arunachal-monpa`): PARTIAL — the source-specific build
  completes through references and graph unification, preserving all 9,279 installed source
  locators; the terminal repository-wide manual-survey check stops only on the unrelated stale
  Kannauji baseline (1,991 current links versus the expected 1,985).
- Focused Monpa importer, identity, registry, profile, bibliography, and conversion suite: eight
  passed. All 9,279 source forms convert without replacement characters. The importer recovers
  1,066 attestations absent from the upstream generated CLDF, including both adjacent `fat`
  concepts and Kho-Bwa headings whose spaces were lost upstream.
- Browser database refresh: PASS. The compact database passed `PRAGMA integrity_check`, eight
  builder tests, three etymology-guess tests, and the app's static checks (zero errors, seven
  pre-existing warnings). The database builder separately warned that the current corpus is
  106.1 MB versus its historical 83 MB warning threshold.
- Browser QA: PASS. Inspected Kalaktang Monpa, Tshangla, Dakpakha, Miji and Bugun language pages,
  their dialect summaries, `/references/abraham-sako-kinny-zeliang2018`, exact `ŋataŋ` and
  `hasokla` searches, and the concepts browser. The source page reports 6,319 reader-facing forms
  after intended compaction while retaining citation/site evidence; source `hasok1a` appears with
  display `hasokla`; and `FAT (ORGANIC SUBSTANCE)` remains distinct from `FAT (OBESE)`.
- No release, deployment, commit, or push was requested or performed.

# SIL Bangladesh Kuki-Chin survey — 2026-08-28

- Full pipeline (`20260828-sil-kuki-chin-bangladesh`): PARTIAL — every data-build stage completed
  through `make_refs.py`; the terminal repository-wide manual-survey check retained the same two
  unrelated Kannauji baseline failures (1,991 current accepted links versus the stale expected
  1,985). All 3,235 new source/site identities survive the compiled CLDF build one-for-one with
  zero replacement characters.
- Parsed all 306 Appendix A.3 prompts on printed pp. 50–88. The 2,565 printed response groups
  expand to 3,875 attestations: 3,235 Bangladesh attestations from ten sites install beneath
  Pangkhua (647), Bawm Chin (647), Mizo (645), Asho Chin (648), and Khumi Chin (648). The 307
  standard Bangla controls and 333 external Myanmar Khumi comparisons are audit-only; the latter
  account for all 53 explicit `no entry` records.
- OCR was unnecessary. The public 127-page PDF has a complete text layer whose subsetted
  `SAG-IPA-SILManuscript` font uses private-use code points. All 16,029 occurrences of the exact
  65 used glyphs were decoded from the embedded outlines and SIL SAGIPA2Uni mapping, with zero
  unparsed lines and zero unmapped symbols. The source PDF, deterministic transcript and glyph
  census are fingerprinted.
- All five languages have authoritative Glottocodes and the ten source localities are registered
  as quality-C dialects, using exact mapped points where available and explicitly approximate
  subdistrict coordinates otherwise. Similarity groups remain notes rather than etymological
  claims; this source intentionally adds zero graph edges.
- Focused validation: seven source extraction/import/audit/registration/compiled-identity checks,
  four relevant sound-profile/build-routing checks, eight browser-database unit tests, SQLite
  integrity, and Svelte checks passed. Svelte reported zero errors and seven existing warnings.
- Browser QA: PASS — rebuilt and staged the compact database; inspected `/languages/Pangkhua`,
  `/languages/BawmChin`, `/languages/AshoChin`, `/languages/KhumiChin`, and `/languages/Mizo`
  with their exact form counts and two dialects each; inspected
  `/references/kim-roy-sangma2011kukichin` and `/entries/f_hc6fe7vkp3xj2`; exact `rɨvan` search
  returned the Bilaichari and Konglak attestations; expanded unetymologised evidence on
  `/concepts/1732` (SKY) exposed forms from all five languages. The builder warned that the
  current repository-wide database is 103.8 MB versus its historical 83 MB warning threshold.
- No release, deployment, commit, or push was requested or performed.

# SIL Meitei (Manipuri) survey — 2026-08-28

- Full pipeline (20260828-sil-meitei): PARTIAL — every data-build stage completed through
  `make_refs.py`; the terminal repository-wide manual-survey check retained the same two
  unrelated Kannauji baseline failures (1,991 current accepted links versus the stale expected
  1,985).
- Parsed all 307 numbered Appendix B.3 prompts on printed pp. 45–68. The 1,219 printed response
  groups expand to 2,713 attestations: 2,406 Meitei attestations from six Bangladesh and two
  Manipur lists are installed, while all 307 Standard Dhaka Bangla controls remain audit-only.
  All 2,406 immutable source/site identities survive the compiled CLDF build one-for-one.
- The appendix has a complete text layer and did not require OCR. All 2,534 occurrences of 25
  legacy private-use glyphs were decoded with SIL's official `SAGIPA2Uni.map` v1.0; the PDF,
  page-text transcript, decoded TSV and used-glyph census are fingerprinted, with zero unparsed
  lines and zero unmapped legacy symbols. Rendered checks of printed pp. 45, 57 and 68 verified
  length, retroflexion, rhotics, nasalization, and below-letter diacritics against the page image.
- The eight site lists are registered beneath canonical Manipuri (`mani1292`); Mukabil is tagged
  as Pangal (`pang1284`) and the remaining lists as Meitei (`meit1246`). Report-route and town
  coordinates are explicitly approximate, quality C. No etymological relationship is asserted,
  so the graph contribution is intentionally zero edges.
- Focused Meitei extraction/import/profile/compiled-identity validation: 9 passed after the
  completed build. The wider sound-profile test file has one pre-existing Mudhili Gadaba
  geminate conversion failure (`talːu` versus expected `tallu`), unrelated to this source.
- Browser QA (20260828-sil-meitei): PASS — rebuilt and staged the compact browser database;
  its integrity/build checks and eight unit tests passed. Inspected `/languages/Manipuri`,
  `/references/kim-kim2008meitei`, `/entries/f_nvtvty23nlzlm`, an exact `noŋ` lexicon search,
  and `/concepts/1489` (CLOUD), where the unetymologised list exposes all four compacted Meitei
  cloud forms including `nōŋthāŋkupːaʔ`. The browser intentionally compacts 2,406 site-specific
  source identities into 1,189 reader-facing forms while preserving dialect tags and citation
  locators. All eight dialects render with their quality-C locality metadata. The builder warned
  that the current repository-wide database is 102.7 MB versus its historical 83 MB threshold.
- No release, deployment, commit, or push was requested or performed.

# SIL War-Jaintia survey — 2026-08-28

- Full pipeline (`20260828-sil-war-jaintia`): PARTIAL — every data-build stage completed through
  `make_refs.py`; the terminal repository-wide manual-survey check retained the same two
  unrelated Kannauji baseline failures (1,991 current accepted links versus the stale expected
  1,985).
- Parsed all 307 numbered Appendix B.3 prompts on printed pp. 57–87. The 1,690 printed response
  groups expand to 3,459 attestations: 2,030 records from seven War-Jaintia sites are installed;
  1,428 Pnar, Lyngngam, Khasi War and standard Khasi controls and one undefined printed `U` are
  audit-only. All 2,030 immutable source/site identities survive the compiled CLDF build
  one-for-one with zero replacement characters.
- The appendix has a complete text layer and did not require OCR. All 2,398 occurrences of 17
  legacy private-use glyphs were decoded with SIL's official `SAGIPA2Uni.map` v1.0; the PDF,
  page-text transcript, decoded TSV and used-glyph census are fingerprinted, with zero unparsed
  lines and zero unmapped legacy symbols. Rendered checks of printed pp. 57, 68, 70 and 87
  verified the conversion and both source anomalies: item 119 visibly prints undefined code `U`,
  while item 137 visibly uses `A` as the tenth similarity group after 1–9.
- The seven target lists are registered beneath canonical War-Jaintia (`warj1242`) with explicit
  quality-C approximate locality metadata. The shared `conversion/sil-bangladesh.txt` route has
  complete source-symbol coverage. The report asserts no etymological relationships, so the graph
  contribution is intentionally zero edges.
- Focused extraction/import/profile/compiled-identity validation: 10 passed. The wider
  sound-profile file retains one pre-existing Mudhili Gadaba geminate expectation failure
  (`talːu` versus `tallu`), unrelated to War-Jaintia.
- Browser QA (20260828-sil-war-jaintia): PASS — rebuilt and staged the compact browser database;
  SQLite integrity, three unit tests and Svelte checks passed (zero errors, seven existing
  warnings). Inspected `/languages/WarJaintia` (727 compacted forms, seven dialects),
  `/references/brightbill-kim-kim2007warjaintia`, `/entries/f_mahp3a3u3rpyo`, an exact
  `phli yaŋ` search, and `/concepts/1732` (SKY), whose expanded unetymologised evidence shows the
  War-Jaintia form. The browser compacts 2,030 site-specific identities into 727 reader-facing
  forms while preserving all site tags and source locators. The builder warned that the current
  repository-wide database is 103.0 MB versus its historical 83 MB threshold.
- No release, deployment, commit, or push was requested or performed.
# Bahl 1962 Korwa and Pinnow 1960 Juang — 2026-08-28

These two packages extend Munda coverage directly from structured SEAlang source data; neither source used OCR. The applicable checklist addenda were Dictionary/glossary, Website/API, and Etymological/comparative.

## Bahl 1962 Korwa

- Parsed 1,792 source records: 1,791 lexical records and one empty record retained in the audit only.
- Installed 1,830 form variants after source-faithful alternant expansion.
- Replaced 57 compatible legacy `BAHL` rows; retained 10 legacy rows whose identities could not be securely reconciled.
- Linked 57 source records (58 installed variants) to existing Rau Proto-Munda parameters using conservative form-and-meaning agreement.
- Classified all 50 source notes: 21 comments, 19 comparative notes, and 10 other notes.
- The seeded 20-row review sample had 0 material errors.
- All installed forms are covered by `conversion/bahl-korwa.txt`; the compiled source identities contain 0 replacement characters.

## Pinnow 1960 Juang

- Parsed 1,658 source records and 1,824 raw form variants.
- Excluded six exact repeated alternants and installed 1,818 distinct variants.
- Replaced 66 compatible legacy `PJDW` rows; retained seven unresolved legacy rows.
- Linked 66 source records (72 installed variants) to existing Rau Proto-Munda parameters using conservative form-and-meaning agreement.
- Classified 1,410 source notes: 1,400 comparative notes and 10 comments. Comparative prose is preserved as etymological evidence without inferring graph edges.
- Retained 185 intentionally glossless records: 33 source blanks and 152 source question-mark glosses. Source uncertainty and two parenthetical source-form markers are preserved in notes/audit fields rather than encoded as lexical forms.
- The seeded 20-row review sample had 0 material errors.
- All installed forms are covered by `conversion/pinnow-juang.txt`; the compiled source identities contain 0 replacement characters.

## Validation

- Focused importer suites: 10 passed (`test_bahl_korwa_1962.py` and `test_pinnow_juang_1960.py`).
- Sound-profile coverage and representative-conversion checks passed for both sources.
- The full CLDF pipeline completed through reference generation and graph unification. Exact source-key checks found 1,830 Bahl identities and 1,818 Juang identities in compiled output, with 58 and 72 linked pre-unification form IDs respectively.
- Repository-wide tests: 937 passed, 10 skipped, and 42 failed. Neither new source test module failed. The failures belong to pre-existing or concurrent work (including durable-ID expectations, older pre-unification schema assumptions, baseline counts, and incomplete SIL Jaunsari integration), so checklist gate 12 remains open for these packages.
- Browser refresh, release packaging, commit, and publication were not requested and were not performed.

# SIL Jaunsari survey — 2026-08-28

- Full pipeline (20260828-sil-jaunsari): PARTIAL — every data-build stage completed through
  `make_refs.py`; the terminal repository-wide manual-survey check retained two unrelated
  Kannauji baseline failures (1,991 current accepted links versus the stale expected 1,985).
- Focused Jaunsari importer and sound-profile validation: 8 passed after the completed build.
  The compiled graph retains all 1,619 installed Jaunsari source identities one-for-one.
- Browser QA (20260828-sil-jaunsari): PASS — rebuilt and staged the compact browser database,
  whose integrity/build checks and eight unit tests passed. Inspected `/languages/jaun`,
  `/references/john2008jaunsari`, `/entries/f_xhlsqcji4jfcg`, `/concepts/1480`, and an exact
  `çarir` search. The source page shows 1,278 reader-facing forms after the intended compaction
  of identical attestations while preserving their site tags and citation locators; the source
  `çʌɾiɾ` and display `çarir` are both visible. The database builder separately warned that the
  current repository-wide database is 102.1 MB versus its historical 83 MB warning threshold.
- No release, deployment, commit, or push was requested or performed.

# SIL Bishnupriya (Manipuri) survey — 2026-08-28

- Full pipeline (20260828-sil-bishnupriya): PARTIAL — every data-build stage completed through
  `make_refs.py`; the terminal repository-wide manual-survey check retained the same two
  unrelated Kannauji baseline failures (1,991 current accepted links versus the stale expected
  1,985).
- Parsed all 307 numbered questionnaire prompts on pp. 35–52. The source contains 746 printed
  response groups, which expand to 2,099 attestations: 1,801 Bishnupriya village attestations are
  installed, 298 Standard Bangla controls remain audit-only, and nine prompts with no printed
  response remain explicit in the audit. All 1,801 immutable source/site identities survive the
  compiled CLDF build one-for-one.
- Recovered all 947 occurrences of the 14 legacy private-use transcription glyphs and all 161
  superscript aspiration markers. The deterministic transcript hash and all 18 rendered source
  page hashes are pinned; every ambiguous aspiration placement was resolved against the page
  image. Item 267's visibly printed lowercase `o` is conservatively interpreted as the source's
  Standard Bangla code `0` and flagged in the audit.
- Focused Bishnupriya importer suite: 7 passed. Sound-profile coverage and representative
  conversions passed after the completed build. The source-specific extraction, identity,
  language/dialect, bibliography, and graph checks all pass.
- Browser QA (20260828-sil-bishnupriya): PASS — rebuilt and staged the compact browser database;
  its integrity/build checks and eight unit tests passed. Inspected `/languages/Bishnupriya`,
  `/references/kim-kim2008bishnupriya`, `/entries/f_xsbyjhn5odruu`, an exact `megh` search, and
  `/concepts/1489` (CLOUD). The browser intentionally compacts the 1,801 site-specific source
  identities into 773 reader-facing lexical forms while preserving all village tags and source
  locators. The builder warned that the current repository-wide database is 102.3 MB versus its
  historical 83 MB warning threshold.
- No release, deployment, commit, or push was requested or performed.
