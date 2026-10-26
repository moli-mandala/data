# Whole-book recovery — 2026-09-27

The original 110-page PDF and 742 installed main-wordlist rows remain unchanged.
The source checklist is active with dictionary, comparative-table and OCR addenda.
This reopening is required by the user's whole-source instruction; no stage-only
extraction will be represented as complete ingestion.

Direct review of PDF6 (contents) and boundary pages confirms these sections.
Printed page +2 is the observed PDF one-based offset. Ranges use the contents
and observed transitions; every interior page still needs structural inspection.

| Printed pages | Scope | Current disposition |
|---|---|---|
| Front matter | Title, imprint, introduction, contents | Bibliographic/context review; check any lexical examples |
| 5–18 | Chakma grammar | Newly recognized omitted lexical and grammatical examples; recover all |
| 19–46 | Chakma–Bengali–English thematic wordlist | Existing 742 source rows; retain keys and recheck whole-book overlaps |
| 47–59 | Bengali–Chakma reverse wordlist | Recover both physical columns; Bengali supplies source gloss |
| 60–68 | Homophones | Independent census:197 groups/427 cells; printed numbering reaches198 but skips146. Preserve distinct meanings and physical attestations |
| 69–74 | Gojha/group-based pronunciation differences | Independent census:70 groups/145 cells; named groups require identity review, not automatic geographic/dialect equivalence |
| 75–76 | Comparative language abbreviations | Resolve labels for the following comparisons; no invented identities |
| 77–87 | Similarities with other languages | Numbering reaches101; comparisons alone do not establish ancestry |
| 88–90 | Chakma–Kokborok | Chakma target, comparator separately accounted; distinct gloss scopes occur |
| 91–93 | Chakma–Brok-Skad | 72-row owner census; source Ladakh attribution and Glottolog support existing Brokskat `bro`, subject to independent mapping review |
| 94–97 | Santali–Chakma | Source explicitly distinguishes some two-language meanings; target Munda evidence must be accounted |
| 98–101 | Further cross-language comparisons | Interior layout and language inventory still to review |
| 102–105 | Languages, families and populations | Documentary table, not lexical attestations; verify all four pages |
| 106–107 | Basic wordlist | 99 physical cells, two blocks per page; full owner first reading retains16 explicit transcription uncertainties and source gloss conflicts |
| PDF110 | Back cover | ISBN/price only in viewed original |

Viewed PDF pages:6–8,49,61–62,70–71,78–79,89–90,92–93,95–96,105–110.
The boundary review is not a full transcription audit. It also shows why the
remaining sections cannot be excluded as mere duplicates: grammar, homophones,
named lects and distinct comparative meanings add analyses and attestations.

`recover_remaining_ocr.py` reuses the pinned scan and existing Bengali/English
models. It recovers every PDF page outside21–48 with one worker, atomic per-page
TSV caches, pinned model hashes, and at most one temporary rendered page. No raw
OCR is accepted as a lexical reading without the subsequent structural audit.

Remaining gates: every-page/record census, original-reading reconciliation,
rich field and language/dialect mapping, whole-book key/deduplication review,
source profile/reference revision, fresh independent audit, focused checks and
canonical installation. Full compiled pipeline, full suite, database and browser
gates remain deferred under the user's current boundary.

## Whole-chapter recovery update, 2026-09-27

The 82-page remaining OCR inventory is complete (4,511 raw lines), but raw OCR
is not accepted transcription. Subsequent original-page work is recorded in the
following separate, reproducible proposals; the existing 742 canonical rows
have not been replaced by these proposals.

- `brokskat-first-reading.jsonl`: all 72 paired rows on printed 91–93,
  144 witness cells. Six rows have separate language-specific meanings; 21 rows
  carry review notes. Complete independent original review is assigned.
- `santali-first-reading.jsonl`: all 96 paired rows on printed 94–97,
  192 witness cells. Fourteen rows require language-specific sense scope; 48
  carry review notes. Santali is the left column, Chakma the middle column.
  Complete independent original review is assigned. Both are target languages.
- `kokborok-first-reading.jsonl`: all 68 paired rows on printed 88–90,
  preserving 68 Chakma target cells and 68 audit-only Kokborok comparator cells.
  Seven rows require separate language meanings. Repeated headers on printed
  89 and 90 are excluded explicitly; rows above those headers remain included.
  The referential Chakma definition on printed 90 row 1 retains its full raw
  context, including the preceding Kokborok purpose, for subsequent resolution.
- `further-first-reading.jsonl`: all 84 paired rows on printed 98–101, plus
  five additional comparator forms embedded in definitions. The 43 literal
  comparator labels await identity mapping; one repeated abbreviation leaves
  the two-language meaning allocation explicitly unresolved.
- `abbreviations-first-reading.jsonl`: all 88 printed declarations on pages
  75–76, preserving both occurrences of Bengali and 87 distinct literal
  abbreviation strings. No declaration is a lexical row. The declaration
  `চা` → `চাক` conflicts with the contextual use of `চা.` for Chakma meaning
  scopes elsewhere, so every occurrence requires contextual resolution.
- `numbered-group-raw-inventory.jsonl`: every group 1–101 on printed 77–87
  now has separately allocated raw evidence. Original word coordinates split
  56 OCR lines that combined opposite columns; 621 resulting column fragments
  (1,557 OCR word occurrences) are assigned exactly once. Fourteen cross-column
  or cross-page groups reconcile with the original layout. Headless group 48,
  blank comparison cells in 7/19, and explicit-origin groups 65/87 remain
  specially marked. This is structural raw recovery, not accepted literal
  transcription or a count of lexical witnesses.
- `basic-reviewed-reading.jsonl`: all 99 cells on printed 106–107 reconciled
  with the independent full review, including all 97 nonblank Bengali controls.
  Three corrections are carried forward without changing the frozen first
  reading. The One, Dog and Liver forms retain explicit transcription holds.
  English/Bengali semantic conflicts and the two blank control cells remain.
- `homophones-first-reading.jsonl` and `named-groups-first-reading.jsonl`:
  the section owner reports complete first readings of 427 and 145 cells,
  respectively. Fine-glyph review and independent certification remain.
- `reverse-chapter-scope-review-20260927.json`: the independent whole-chapter
  boundary census proposes 741 paired cells on printed 47–59; exhaustive keyed
  transcription must reconcile that provisional count before freezing it.

The reverse-wordlist owner subsequently completed all 741 keyed cells and pinned
the 13-page first reading in `reverse-first-reading-freeze-20260927.json`.
Independent glyph review remains. `TRANSCRIPTION_REVIEW_POLICY.md` records the
newly detected source-wide hasanta/disjoining omission class and the full-scope
review required before any first-reading proposal can be installed.

Comparative row keys are based on chapter, printed page and physical row, with
separate witness keys. Slash/comma alternatives and source parentheses are kept
literal at this stage. Comparison does not imply cognacy, borrowing or a variant
edge. The companion `*-status.json` files pin input and output hashes and list
remaining review work; none constitutes approval of canonical installation.

`further-language-mapping-review.json` now inventories all 84 main comparison
witnesses and all five embedded comparisons, keyed to their immutable witness
IDs. It proposes 43 direct base-language mappings and six context-supported
Brokskat mappings, retains 14 comparisons outside the requested language
families, and leaves 26 identity assignments unresolved. These are identity
decisions only: literal readings, source-specific varieties and historical
attributions remain separate review requirements. Generic Munda, unspecified
Prakrit, Kayethi, Shabar and other ambiguous labels have not been silently
assigned to a more specific registry language. The reproducible inventory
validates all proposed IDs against the current registry and pins both inputs.
No canonical rows were changed by this inventory.

The Brokskat and Santali full reviews and subsequent hasanta sweeps are now
reconciled reproducibly in `reconcile_brokskat_santali_reviews.py`: 168 rows,
336 witness cells, and 39 changed witness proposals. Repeated evidence is
retained without double-counting changed cells, and the one superseded
Brokskat proposal remains traceable. Root inspected the original Brokskat
images and adjudicated seven keyed changes in
`brokskat-root-glyph-adjudication-20260927.json`. Five remain accepted;
approval of two হ-related hasanta readings was withdrawn after an independent
font-control challenge. Four other glyph holds remain. Santali root review
similarly accepts 30 proposed changes and holds two হ-related proposals.
Ordinary হ itself can carry a lower stroke, so this class needs further control
comparison before adding a hasanta. This is transcription adjudication, not
installation approval.

`reverse-reviewed-reading-proposal.jsonl` now reconciles the independent review
of all 741 reverse-wordlist cells (both target and Bengali prompt). The script
rechecks all 13 frozen page TSV hashes and exact source strings before carrying
forward 95 findings: 37 reviewer corrections, 26 proposed corrections, and 32
typography uncertainties. There are 63 changed field proposals; ten involving
হ্ are explicitly held under the subsequent font-control caution. These
categories preserve the historical review's wording and are not root approval.
No raw reading, source qualifier, alternate, or control-language attribution
has been replaced in the immutable first-reading files.

The numbered chapter now has an independent full second reading in
`independent-numbered-complete-review-20260927.json`: all 101 groups, 115
page/column segments and 620 original lines reviewed against all 22 original
half-page images. Every original line key is accounted for, including all
14 continuations, the two explicit blank comparator fields and headless group
48. The 621 OCR fragments reconcile to 620 original lines because group42's
last line has overlapping OCR fragments. Ten groups have additional root
findings; all earlier unresolved readings remain explicit. Structured witness,
definition, language-label and reference extraction still remains, so this
review does not approve canonical installation.

`numbered-structured-review.jsonl` now structures all 101 groups and accounts
for every one of the 620 source lines: 100 heads, 395 comparison slots (392
nonblank witnesses awaiting identity mapping, two explicit blanks, and one
unresolved label/form segmentation), headless commentary, and two explicit
source analyses. Components, ordered developments, cited works and their
quotations remain separate; group87's লম্পক and লম্বক are not harmonized.
Comparator-specific meanings, repeated labels on one baseline, head qualifiers
and the group52 Charyapada citation retain their distinct scopes. Six focused
regressions passed (`test_niranjan_chakma_numbered_structure.py`). This validates
structural preservation only; canonical language mapping, residual glyph
adjudication, final lexical expansion and whole-source installation remain.

All four demographic-table pages (printed102–105/PDF104–107) have now been
independently inspected in full. `documentary-pages-review-20260927.json`
accounts for 94 language-metadata rows and zero lexical entries, including the
wrapped Hindi locality cell and large blank layout gaps. They are explicitly
documentary rather than silently omitted. Their historical population and
genealogical claims are not imported into the language registry.

The full thematic audit covers all 28 original pages, 733 physical cells and
742 installed rows. Root independently inspected fresh original-row crops for
all 14 proposed corrections and accepted them at transcription stage in
`thematic-root-glyph-adjudication-20260927.json`: twelve explicit disjoining
repairs and two letter/vowel corrections. Nine separate uncertain candidates
and the earlier damaged/scope-held cells remain explicit. The validator confirms
the canonical CSV and all existing keys are unchanged; whole-book installation
will apply the accepted corrections together with the recovered sections.

The homophone and named-group second readings now reconcile all 572 cells
into `homophones-reviewed-reading-proposal.jsonl` and
`named-groups-reviewed-reading-proposal.jsonl`. These preserve 18 corrections,
15 proposed corrections, 32 uncertainties and ten explicit confirmations.
Despite the historical `independent-*` report filenames, the reviewer also
authored these first readings: the reconciled records explicitly disclose
same-author rereading and do not claim independent transcription certification.
Root correction adjudication and an independent fresh audit remain. All
homographs, distinct definitions, missing labels and Tanchangya attributions
stay separate.
