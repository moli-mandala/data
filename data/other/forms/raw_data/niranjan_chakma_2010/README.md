# Niranjan Chakma (2010): source preparation

**Current state, 2026-09-27:** whole-source files are installed: **3,662 rows**,
**5,232 accounting records**, all **742 historical keys** preserved, and **486
typed-uncertain rows**. A fresh independent 20-unit original-output review
passed; 40 focused tests and the actual canonical parser pass. Two consecutive
guarded installations produced identical bytes. Evidence and exact hashes are
in `whole-installation/source-stage-handoff.json`. The canonical route is
`whole-installation/install_whole_source.py --install`; it reproduces revision 2
from the immutable historical baseline and requires hash-bound acceptance.

Known-language forms with defensible provisional glyph encodings are retained
with uncertainty. Unresolved language identities, specifically damaged readings,
explicit blanks and controls remain fully audited. Three source-analysis records
preserve their entire printed component/origin prose but do not receive inferred
language assignments or ancestry edges. Main reference etymology provenance is
`source`; auxiliary works are separately registered as secondary references.

Full CLDF/database, compiled identity/graph/reference, full-suite and browser
gates remain deferred. No database build or publication was performed.

## Earlier preview and historical-stage evidence

**2026-09-27 whole-source preview:** the full 110-page source is now accounted
for in `whole-source-assembly-preview.jsonl` and `whole-source-export-audit.jsonl`.
The frozen fifteen-column preview has 3,660 rows, preserves all 742 installed
keys, and includes grammar, reverse vocabulary, homophones, named varieties,
all comparative sections and the final basic list. Documentary pages and
unresolved/control witnesses remain explicitly accounted for. See
`WHOLE_EXPORT_READINESS.md` for exact exclusions, accepted uncertainty policy,
validation and remaining gates. Independent integrated review is in progress;
the canonical CSV still contains the historical 742-row stage.

Revision 2 in `whole-export-r2/` contains **3,662 rows**: the independently
verified Prakrit and Vedic comparisons omitted by the first identity mapping
are now included, with explicit unspecified-variety/stage notes. All 3,660
earlier rows are unchanged. The original freeze and its failed accounting
review are preserved. Revision 2 also passes the actual generic parser with
no field differences; independent original-output review remains required.

The new route `whole-installation/install_whole_source.py --check` reproduces
the frozen CSV/audit from an immutable 742-row baseline. Its `--install` mode
requires a hash-bound root approval and matching independent review. The
original `import_source.py` and the reproduction instructions below describe
the historical thematic stage; do not use that installer to overwrite a
future whole-source installation. Full database/build/browser gates remain
deferred under the current source-stage scope.

## Historical thematic-stage record

**Main-wordlist source files installed; broader source review and compiled validation remain incomplete.** This package does
not run a CLDF or browser database build.

Niranjan Chakma, *The Chakma Vocabulary & Terminology*, first publication
February 2, 2010, Language Wing, Education Department, TTAADC, Khumulwng,
Tripura. ISBN 978-93-82172-08-6. The introduction describes the June 9–12,
2009 workshop. The 110-page image PDF is pinned in `source-manifest.json`.
The public scan is at
<https://chakmamaadi.wordpress.com/wp-content/uploads/2022/05/2010-the-chakma-vocabulary-terminology.pdf>.
No open redistribution licence has been verified; the PDF remains in local
scratch. Public availability is not treated as a licence grant.

The current extraction covers the main Chakma–Bengali–English thematic wordlist,
printed pp. 19–46 (PDF 21–48). Chakma is printed in Bengali script. Bengali is
a gloss/control column, not an additional language to ingest from this section.
No Roman transcription, phonemic analysis, dialect assignment, or etymological
relationship has been inferred.

Later Bengali–Chakma, homophone, geographic-variation, and comparative chapters
remain pending. In particular, the Roman heading “Brok-Skad” must not be
silently interpreted as Bru/Reang. This partial extraction does not claim to
represent the whole book.

## Reproduction

From the `data` repository, using its existing Python environment:

```sh
.venv/bin/python data/other/forms/raw_data/niranjan_chakma_2010/extract.py \
  --pdf ../tmp/pdfs/chakma-discovery/niranjan-2010.pdf \
  --output ../tmp/pdfs/chakma-discovery/niranjan-main-ocr-v4 \
  --boundaries data/other/forms/raw_data/niranjan_chakma_2010/column-boundaries.json \
  --bengali-model-dir ../tmp/pdfs/chakma-discovery/tessdata-best
.venv/bin/python data/other/forms/raw_data/niranjan_chakma_2010/preview.py \
  --output ../tmp/pdfs/chakma-discovery/niranjan-preview
```

PDFium renders one page at a time at scale 4; Tesseract uses Bengali/English
column models, PSM 6, and one thread. The manifest records the version and
actual crop coordinates. Full page renders and raw TSV files remain in scratch.
The Bengali model is the official `tessdata_best` model, pinned by URL and hash
in `review-summary.json`. Its embedded English sublanguage is disabled explicitly:
leaving it enabled introduces Latin-script guesses into Bengali-script cells.
The checked-in raw line ledger preserves word boxes, confidence values and text.
PDFium opens the scan normally with no password; pdfminer fails on its permission
integer, and the sampled pages have no native text layer.

The first fixed-crop experiment clipped words on shifted pages. Per-page crops
recover words such as “Current”, “Root”, and “Moonbeam”, but are still provisional.
Headers may cross crop boundaries. All 28 pages have now received a first visual
review; native-script uncertainties and a fresh acceptance audit remain open.

## Review accounting

- 2,313 raw OCR lines across all three columns; 758 are English-column lines.
- Provisional structure excludes 26 English header/noise lines and joins one
  wrapped botanical name. Visual review recovered “Dancing hall” on printed p.26
  and “Linguistic” on p.35, omitted entirely by full-column English OCR, producing **733 candidate records**,
  not accepted forms. Isolated-cell OCR and original-page coordinates document
  the recovery in `structure-review.json`.
- 89 native-column lines remain unpaired and are retained separately. Their
  individual exclusions in `unpaired-line-review.jsonl` identify section headings,
  column headings and page footers; no lexical row is excluded by that ledger.
- The first pass lacked 14 aligned OCR readings. All 14 were visually confirmed
  to contain printed text; eight are Chakma cells and six are Bengali controls.
  The Bengali-only best-model pass recovers all 14; current alignment has zero
  missing cells. Earlier `v2-*` evidence is retained and English anchors are
  unchanged. This proves improved alignment, not correct spelling.
- Raw candidates retain their original unreviewed status; visual decisions live
  in the separate review ledger. Candidate row numbers are review
  locators only, not finalized stable entry keys. The first seeded transcription audit found two material errors in 20 entries;
  both were corrected. A fresh sample (seed 2026092109) found no new material
  errors in 20 entries, including one already-flagged unresolved spelling.
  This does not certify uncertain readings or an installed importer, the 742-row source CSV is installed, and no build has been run.

`structure-review.json` preserves the provisional line classification;
`candidate-cells.jsonl` and `unpaired-lines.jsonl` are reproducible review outputs.
`missing-cell-review.json` preserves the follow-up cell OCR and original-page
crop boxes. Source spelling and English misprints must remain distinguishable
from OCR errors. No blind transliteration or spelling correction is authorized
by this package.

`visual-review.jsonl` records 733 reviewed headword readings on printed pp.19–46,
including 91 remaining conjunct/letter uncertainties. A second enlarged-crop
review of all 144 flagged readings resolved 53 and retained uncertainty in 91;
`second-glyph-review.jsonl` preserves the previous readings and decisions. It preserves the raw OCR
beside each reading. These decisions feed source rows; uncertain readings remain explicitly flagged. The grammar
introduction explicitly notes problems writing Chakma in Bengali letters;
`transcription-review.json` therefore proposes source-script preservation with
blank Phonemic and no inferred transliteration. Publisher location alone does
not establish the wordlist's dialect.

Four English-gloss discrepancies remain separately recorded: printed “Worm”
beside Bengali গরম (warm), “Spout” beside অঙ্কুর (sprout), and “Torm” beside
কাঁটা (thorn), plus “Admission text” beside ভর্তি পরীক্ষা (admission test).
Suggested corrections do not overwrite the printed evidence. Printed p.33's
two occurrences of শুলোনি “Pain” remain separate source records.

Remaining source work is itemized in `review-summary.json`. All remote work and
database builds are explicitly deferred by the user's instructions. Compiled
integration and browser checks therefore remain deferred, not passed.

Fifteen focused preparation, importer and lightweight parser tests pass: original-page crop coordinates,
deterministic review-output reproduction, and retention of missing OCR cells
and the wrapped botanical gloss, plus unchanged English anchors between models.
Additional checks protect the recovered whole row and bind visual review to its
original OCR cells without dropping uncertainties, and account for every unpaired
line exclusion. A second-review check preserves reading history and prevents
unresolved flags from being cleared. These are evidence checks, not a lexical
acceptance audit.

Multiple-form review records 11 pairs of independent coequivalents and one
unresolved slash scope in the eye-ball expression. The pairs do not justify
variant edges. The unresolved expression must not acquire an inferred repeated
prefix. See `multiple-form-review.json` for literal segments and scan evidence.

## Source-level draft importer

`import_source.py --output <scratch-directory>` emits a 15-column `draft.csv`,
`audit.jsonl` and counts, without invoking a build; add `--install` to write the canonical source CSV and YAML. Frozen
physical source-cell keys live in `entry-locators.json`; spelling corrections
and review-order changes do not change them. `row-policy.json` records the
field mapping and explicit withholding decisions.

Current reconciliation: 733 source cells minus two withheld cells, plus 11
additional coequivalents, gives **742 source rows**, of which **97 carry
uncertainty flags**. The damaged warm headword and unresolved eye-ball slash
scope remain in the checked-in `draft-audit.jsonl`, alongside every emitted
row's raw OCR and review evidence. Three editorial gloss corrections use the
recorded Bengali controls and retain printed English in the audit. No variant,
borrowing or ancestry edges are inferred. Source script is retained in Form
and Native, with Phonemic blank.

Bibliography and source-preserving settings validate locally; the complete
55-character inventory is recorded in `symbol-inventory.json`. Later sections
of the book remain in scope for review. Database and remote work remain deferred.

## Source installation checkpoint

The main-wordlist CSV and settings are installed as
`data/other/forms/20260921-niranjan-chakma.csv` and `.yaml`; the bibliography
is registered under `niranjan2010chakma`. Chakma's registry clade is now Eastern,
with evidence in `language-review.json`; no publisher-derived dialect is added.
The source-output sample (seed 2026092110) found 0/20 new material errors,
including one already-flagged uncertain spelling. Its exact rows match the
installed CSV. Lightweight `parse_file` checks preserve all 742 spellings with
zero conversions. This does not establish compiled survival, database validity
or whole-book completion. Those gates remain explicitly open/deferred.
