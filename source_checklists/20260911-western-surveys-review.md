# Western surveys ingestion review — 2026-09-11

The full `SOURCE_INGESTION_CHECKLIST.md` is active. Addenda: survey/comparative
table for Varli, glossary and OCR for the two Ghatage sources. This installation
is reviewable, but the OCR review backlog and non-clean full validation prevent
claiming every checklist gate complete.

## Coverage and exclusions

| Source | Raw units | Installed forms | Exclusions |
| --- | --- | --- | --- |
| Dadra and Nagar Haveli (2003), Varli table | 414 prompts × 6 lect columns = 2,484 cells | 816 | 1,656 comparison cells; 53 blank target cells |
| Ghatage, South Kanara Konkani (1963) | 21 pages; 1,439 OCR candidates | 1,389 | 50 unresolved/damaged candidates |
| Ghatage, Kudali (1965) | 56 pages; 2,034 OCR candidates | 1,902 | 133 unresolved/damaged/nonlexical candidates |

Varli has 775 nonblank target cells and 41 additional slash-separated responses:
430 Davar and 386 Dungar forms. Dhodia, Kokni/Kokna/Kukna, Gujarati and Marathi
comparison columns are excluded. Kudali has one entry with separately labelled
feminine/masculine heads, adding one output row. Ghatage counts are recognized
OCR candidates, not an independently enumerated census of printed entries.
Pinned whole-page OCR retains ignored layout context for future recovery.

All 4,107 forms are unlinked. The included tables make no etymological claims;
inferred cognacy, borrowing, derivation and artificial variant edges are
inapplicable. Exact page/item or page/column/entry locators and unique source keys
preserve homographs. Audits retain raw evidence, exclusions and review status.

## Acquisition and reproduction

- Varli: Census of India PDF, 101 pages; printed 80–87 / PDF 94–101.
  SHA-256 `23024c757475badedfa8f84807701ad7adf7dbf32e991504e248ce74c9864d53`.
  Ranjita Pattanaik and M. K. Koul receive the investigation/report credit;
  Ahirwal is the supervisor. The source is dated 2003 despite later map dates.
- Konkani: Internet Archive scan, 149 pages; printed 120–140 / PDF 128–148.
  SHA-256 `72ddf107fdeef7d427a24d15addda1311c6731c6fc64f44d297a488baca5c4c5`.
- Kudali: complete 2022 Wayback publisher scan, 161 pages; printed 95–150 /
  PDF 105–160. SHA-256
  `f94ccc31234e1037feb5be0525f04340ed733ca28074b8569f854631674f8ea3`.

Truncated 2023 Wayback representations were rejected. The alternate Konkani
plain text loses diacritics and is not the preferred transcription. No verified
publisher open licence is asserted, and an Internet Archive uploader licence
is not treated as a publisher grant. Scans are not redistributed in the repo.

`data/other/forms/raw_data/dadra_varli_2003.py --install` reproduces the manual
Varli layer. `ghatage_western/import_glossaries.py --install` reproduces both
glossaries from pinned compressed word boxes and separate image corrections.
The package includes 300-DPI rendering, deskew/OCR and seeded-audit scripts.
The manifests record acquisition URLs, hashes, page ranges and output accounting;
`data/other/forms/raw_data/ghatage_western/README.md` gives detailed reproduction.

## Language, transcription and sound profiles

Davar maps to Bhili (`dava1244`); Dungar is a separate base language, `Varli`
(`varl1238`, Marathi-Konkani). The distinction follows the source discussion and
Glottolog. South Kanara Chitrapur Sarasvat Konkani maps to existing `Ko`, and
Kudali collected at Vengurla maps to Marathi `M`, following the source's historical
classification. Four language-qualified dialect tags are registered. No new clade
or consultant dialects are invented. Dialect locations are meaningful descriptions
with blank coordinates. The new base Varli point is Glottolog's 20.5635 N,
73.2975 E, explicitly marked as a modern approximation, quality C, not the 2003
elicitation site.

Varli's defined capital codes D/N/T/R/M/S → ḍ/ṇ/ṭ/ṛ/ŋ/ś and E/O → ɛ/ɔ;
colon length becomes macron length. Undefined L/C are preserved with typed
uncertainty. Three clipped Davar alternatives retain visible prefixes and source
truncation flags. Slash responses receive separate keys without inferred variant
relationships. Item 403's explicit masculine plural label is tagged. An incidental
manual ŋ/n misreading in item 67 was corrected after image magnification.

The Ghatage profile preserves č/c, ǰ/j, nasalization, schwa, retroflex letters and
capital morphophonemic notation. Only š → ś and vowel-colon length are normalized.
Ambiguous OCR symbols remain under review. Original is preserved, and separate
Native/Phonemic fields are blank rather than fabricated. NFC and NFD symbol
coverage passes with no replacement characters. Profiles and protected source
keys preserve source-defined homographs and IDs through corrections/reordering.

## Image audits and remaining review

Varli: all eight pages, blank targets, alternatives, clipped cells and grammatical
labels inspected. Fresh seed 20260911: **0/20 material transcription errors**.
There are 24 uncertain compiled rows: three clipped responses and 21 readings
containing undefined source L/C.

Ghatage: three seeded rounds plus chapter boundaries and parser-edge checks.
The first sample found 11/20 material errors in Konkani and 9/20 in Kudali,
plus a recoverable audit-only Kudali head. Corrections remain separate from raw
OCR. Later rounds found further character errors. After gutter/continuation
fixes, the third fresh sample has **0/20 structural alignment errors per book**;
this does not mean zero image-to-transcription errors. The ordinary fresh
zero-material-error target remains unmet for both glossaries.

The checklist explicitly permits structurally reliable unreviewed OCR with raw
evidence, exact locators and typed review markers. Every unreviewed installed row
has `ocr-review uncertain`: **1,276 Konkani and 1,792 Kudali**. Unread/unparsed
heads remain excluded. A fully proofread edition still requires checking these
heads and definitions against scans, recovering damaged candidates and repeating
the fresh material-error audit. These are not presented as fully reviewed texts.

## Validation and shared-workspace limitations

- Final source suite: **14 passed** in 10.17 seconds, including current compiled
  retention, references, dialect metadata, transcription and source graph checks.
  All three generated source checklists match current inputs.
- All seven build-generation stages ran; `errors.txt` is empty. `make all` exits
  nonzero at two pre-existing manual-survey-etymology assertions. The baseline
  already contains 15,887 Rajasthani forms (test expects 15,876), and 1,605
  source-bearing overlay assignments prohibited by the other assertion.
- Default pytest stops at duplicate `test_preintegration_contract` module names.
  Importlib mode reports **39 failed, 1,722 passed, 18 skipped**. A concurrent
  Kharia ingestion changed shared generated files during this run, including
  checklist inputs. This is not an isolated clean result, nor a claim that all
  39 failures are proven pre-existing. No western-survey test failed.
- The western-only build snapshot added exactly 4,107 forms and three references,
  changing no existing compiled form rows or IDs. Its legacy-alias check found
  zero old aliases removed or retargeted. The new files append after existing
  inputs using named temporary IDs, avoiding numeric alias reuse.
- A final current-data comparison still finds zero baseline forms changed or
  removed, all 4,107 survey forms unlinked and zero survey graph edges. The
  concurrent Kharia task added another 521 forms and 68 variant edges. Every
  baseline edge survives; alignments remain byte-identical. Later shared registry changes, including
  Legacy_ID churn is distinguished from the earlier western-only snapshot.
- Complete rebuilds also expose existing Concepticon tie nondeterminism:
  identical probes under different Python hash seeds map “snap” to BREAK OF ROPE
  versus SNAP OF STICK and “to stop” to CEASE OR FINISH versus STOP DOING. The
  shared mapper was not changed. Concept counts/deltas are retained in the report.
- Browser DB refresh and representative app QA are inapplicable: no refresh was
  requested. No browser DB, release, deployment, commit or push was performed.

## Representative compiled entries

| Source | Form | Gloss | Durable form ID |
| --- | --- | --- | --- |
| Davar Varli | hɔwa | air | `f_bqlyi4u2dfsco` |
| Dungar Varli | vārā | air | `f_btjh5oyiz73yu` |
| South Kanara Konkani | əccu | mould | `f_2yhamq3he7k42` |
| Kudali | əkkal | wisdom | `f_t5lzjt77r3xbe` |

These are verified compiled entries, not claims about an unrefreshed app database.
See the [validation JSON](audits/20260911-western-surveys-build-validation.json),
[focused tests](audits/20260911-western-surveys-focused-tests.log),
[build log](audits/20260911-western-surveys-build.log), and
[full test log](audits/20260911-western-surveys-full-tests.log).

## Requested browser database refresh

The user subsequently requested adding the surveys to the DB. The current v32
local SQLite and staged Zstandard artifact already contain all 4,107 forms.
Verified integrity `ok`, exact staged/local SHA-256 agreement, and 44,172,750-byte
compressed size (under 50 MB); expanded size 100,040,704 bytes matches metadata.
Representative entries, source search, grammar/dialect/OCR rendering, affected
language views and AIR concept counts were inspected locally. Svelte check: zero
errors, seven warnings. OCR review remains outstanding. No release/deploy was
performed. See [browser validation](audits/20260911-western-surveys-browser-db.json).
