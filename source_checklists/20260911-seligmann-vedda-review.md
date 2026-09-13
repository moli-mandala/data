# Seligmann & Seligmann 1911: Vedda vocabulary review

The complete SOURCE_INGESTION_CHECKLIST.md is active. Applicable addenda:
dictionary/glossary, OCR-heavy, and survey/locality labels. Etymological comparison
prose is retained as source evidence, not installed as accepted graph analysis.

## Coverage and evidence

- Canonical source: Charles Gabriel Seligmann and Brenda Zara Seligmann, *The Veddas*,
  Cambridge University Press, first edition, 1911. Internet Archive Toronto scan
  `veddas__00seliuoft`; immutable downloaded PDF SHA-256 in the manifest.
- Publisher's current chapter catalogue confirms the vocabulary chapter; the 1911
  original, not the later reprint's typesetting, supplies the actual forms.
- Scope: all 185 numbered articles, 196 sense/supplement records → 502 form rows.
  All numbered articles are represented; no illegible heads are dropped.
- Excluded: comparative-language words in etymological prose, narrative chapters,
  songs, index and following Kaele-base appendix. Raw commentary remains in the
  checked page OCR and per-form audit. It has not been normalized into linguistic
  claims. This is a lexical-attestation ingest, not a new etymological analysis.
- PDF: 636 pages; printed vocabulary pp. 424–450 = PDF pp. 592–618. The abbreviation
  key is printed p. 423 / PDF p. 591. Every vocabulary page and both endpoints were
  visually inspected. No duplicate or displaced vocabulary pages were observed.
- Extraction: existing OCR plain text and pypdf's PDF text layer were inspected;
  both have serious italic-headword OCR substitutions. All installed forms, heading
  glosses and labels were transcribed/checked against page images rendered by
  `pdftoppm -scale-to 1500`. The source image, not a guessed OCR repair, decides spelling.
- Reproduction: checked `verified.tsv` + `pages.json` → deterministic importer,
  preview, installed 15-column CSV and per-form JSONL audit. Scan/render artifacts
  remain under ignored `tmp/pdfs/seligmann-vedda/`; the manifest provides acquisition
  URLs/hash and render commands. Missing checked inputs fail clearly.
- Rights: original published in 1911; public domain in the United States. Only
  lexical facts, transcription and original-source OCR are redistributed here.

## Editorial decisions

- Stable keys use source + printed entry/subentry + printed group/form position.
  They do not depend on corrected spelling/gloss or global input ordering.
- `Original` and `Form` preserve the broad field transcription described on
  pp. xii–xiii. `c` already denotes the affricate. No unsupported inference of
  vowel length, retroflexion, nasality, or phonemic analysis. Explicit profile
  handles NFC and NFD; Native and Phonemic are blank.
- Headword grammar `(v.)` becomes `verb`; explicit imperative and masculine/feminine
  labels are structured. Merely proposed imperatives in etymological discussion
  do not become unqualified grammatical assertions.
- Co-listed expressions remain distinct attestations; commas alone do not create
  variant edges. No etymon, borrowing or derivation edges are inferred. All 502
  forms are unlinked. The donor/matching and ranked graph gates are inapplicable
  to this deliberately lexical scope; zero matching claims were made.
- Vedda is a distinct base language, Glottocode `vedd1240`. Glottolog and the work's
  historical Sinhala-dialect account differ in classification; using the existing
  `Other` clade avoids resolving that dispute in an importer.
- Eleven explicit place labels are registered as Vedda dialects. Their historical
  coordinates have not been established and are blank, as permitted by the
  checklist. No modern language point is misrepresented as an elicitation site.
- `O.` is the consultant Wannaku of Uniche and stays provenance. `T.` is printed
  among locality labels but defined as Tamil on p. 423, alongside `Tk.` Tamankaduwa.
  Forty-four forms retain this ambiguous label in audit with `dialect-mapping`
  uncertainty. They are retained under the volume's Vedda vocabulary scope.
- Additional uncertainty: `adane` (entry 3), explicitly questioned by the source;
  `kanda arini` (17.i), whose bambara identification the source questions. Total:
  46 uncertain form rows, not 46 unreadable OCR heads.
- Notes retain only actual usage/consultant information. Exact printed page and
  item locators appear in every citation. Raw OCR and review flags remain in audit.
- Source bibliography includes authors, edition, publisher, URL, precise included
  portion, provenance, editor attribution and OCR=Yes. Comparative language
  abbreviations are not silently turned into invented bibliographic references.

## Audit and validation

- Every emitted form has a checked per-record audit with raw page OCR, immutable key,
  source label, image locator, parsed form/gloss/tags and typed unresolved cases.
- Seed 19110911 selects articles 9, 14, 16, 28, 65, 77, 78, 82, 85, 87, 88, 93, 97,
  106, 115, 142, 151, 154, 176, 179. Output compared against original images:
  0/20 material errors. All pages additionally received direct headword review.
- Edge cases checked: first/last page, split entries and words, bee/deer/lizard
  subsenses, scientific names, multiword forms, grouped locality labels, occasional
  diacritics, source-questioned spelling, and same-form/different-sense records.
- Focused source tests pass (5 tests before compilation). Full source corpus: all
  502 forms convert with zero unmapped symbols. NFD coverage was repaired during
  focused validation; installed transcription was unaffected.
- Registry tests now permit paired missing coordinates and no quality grade for
  unlocated historical sites, consistent with the ingestion checklist. Existing
  legacy source-reliability annotations remain accepted. Present coordinates retain range checks.
- All seven data pipeline scripts ran successfully. The `make all` final manual-survey
  gate fails two unrelated Rajasthani/Mewari assertions (15,887 source nodes versus
  stale expectation 15,876; source-owned forms still present in the existing
  etymology overlay). Neither source nor overlay was edited by this ingest.
- Final focused suite: 23 passed, including compiled-source survival, persistent
  IDs, symbol coverage, and dialect registry checks. All 502 source keys resolve
  to 502 separate unlinked Vedda nodes; no new unmapped input appears in errors.txt.
- Full suite: the literal `uv run pytest -q` command hit six existing collection
  errors (duplicate test basenames and import-path issues). Retried the complete
  suite using `uv run python -m pytest -q --import-mode=importlib`: 1,708 passed,
  18 skipped, 32 failed in 551 seconds. No Vedda source-specific test failed.
- The full run had loaded the checklist module before its new source registration
  was added. After refreshing generated records, all four checklist tests pass.
  Thus that freshness failure is resolved; 31 failures outside the Vedda tests
  remain. OCR and base-coordinate snapshot expectations were updated for Vedda;
  their fresh rerun now identifies only pre-existing Irula/Gadaba metadata and
  Noiri/Dungra Bhili/Kodaku/Vasavi/PKher/PreMu coordinate exceptions.
- Remaining failures concern other source snapshots, curated etymologies,
  reference metadata and graph expectations. The full log is checked in at
  `source_checklists/audits/20260911-seligmann-vedda-full-tests.log`;
  subsequent focused/checklist/existing-gate logs are adjacent. No unrelated
  scholarly assignments were altered to make these tests pass.
- The mandatory full-validation gate remains open: this source is installed and
  its own checks pass, but the overall ingestion must not be labelled fully
  validated until the repository-wide failures are resolved.
- Generated reconciliation against the initial working tree: forms +502, source keys
  +502, references +1, aliases +508. Zero pre-existing IDs lost; Vedda is the only
  language with a changed node count. Six additional alias records were produced during identifier reconciliation;
  all original live IDs remain present and every source-key alias resolves.
  Graph edges and alignments are byte-for-byte identical to the initial working tree.
- The concept mapper varies between independent processes. A same-process
  with/without-Vedda diagnostic found 444 Vedda links and zero removed or added
  pre-existing links. Thus the aggregate concept-link count decrease in the saved
  build is a reproducibility issue in the existing mapper, not lexical/graph loss.
  The diagnostic and exact saved-build counts are in the build-validation JSON.
- Browser database build and app inspection: not requested; per checklist §13 no
  refresh is performed during routine ingestion. App examples will require the
  next user-triggered refresh. No release, commit, push or deployment requested.

## Representative compiled entries

- `f_aw5odjy4i3h32` gaigedi “Areca-nut” versus `f_g26fhejvsptd4` gaigedi “Coconut”.
- `f_26m3xiszzhwvu` dia “Tears” versus `f_6sjop4tp2fyzg` dia “Water”.
- `f_s7nb3bzswo3cq` naidaṇḍa “Nose”: source diacritics preserved.
- `f_7biclweyctray` kanda arini: source-questioned bee identification.

These IDs are compiled and verified locally; they are not yet in the browser DB.
