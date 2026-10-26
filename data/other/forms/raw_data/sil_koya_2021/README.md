# SIL Koya survey (JLSR 2021-029)

## Current source-stage installation — 2026-09-27

All 124 PDF pages have a recorded scope decision. The independently accepted
whole-report revision installs **1,665 rows**: 1,437 wordlist alternatives,
205 narrative/question units, five explicitly named terms and 18 comparative
attestations. `whole-audit.jsonl` accounts for 2,119 units. Seven existing
Gondi locality/dialect registrations remain unchanged. Thirty-six printed
alternative relationships remain closed within the source.

The source supplies 1,401 responsive target wordlist cells; 69 missing target
slots and 420 Telugu/Oriya control slots are explicitly accounted for.
Untranslated documentary matter, numeric scores, Telugu test narratives,
English questionnaires and the undefined festival name remain contextual
controls, with their scope decisions in the all-page census. Four explicitly
named but undefined narrative terms retain blank glosses and source context.
All handwritten glyph uncertainties remain visible in Notes/tags/audit.

The final metadata revision restores 165 sentential and 40 fragment/multiword
classifications, preserves all 74 test-question/answer boundaries, and retains
the literal stutter annotation in Original. Wordlist numerals/interrogatives
and pronouns receive only source-supported labels; the printed unqualified
you/we prompts do not support invented register or inclusive distinctions.
Literal wordlist single dots, colons, raised letters and uncertain marks are
preserved; Phonemic stays blank. Comparative G follows the source's explicit
voiced-fricative definition. Original and display conventions remain separate.

`install_whole_source.py` is the only current installation route. Without
`--install` it verifies acceptance, both frozen generations, and actual parser
and metadata behavior in a small sandbox. With `--install` it archives prior
files and writes the approved rows/profiles, routing YAML, one source-key alias,
and the single bibliography scope update. Historical `import_koya.py` is an
immutable reconstruction fixture: **do not run its --install mode against the
working repository**. Historical first readings, failed freezes, audits and
legacy installed files are retained without rewriting their evidence.

The incorrect UTN198 v2 key is retired into v1. The durable identity review
establishes that v2 has no separate persistent ID; the surviving JAG198 ID must
not be redirected. Only the approved source-key alias CSV is installed. No
identity registry or compiled alias file is edited. The old unrelated positional
alias is not evidence of Koya ownership.

Fresh independent final audit: **20/20 units, zero material errors**.
Guarded regeneration reproduced all reviewed installable bytes. **26 focused
tests pass**, including actual canonical parsing of all 1,665 rows, source keys,
Original/Notes/citations/tags, closed variants, profiles under NFC/NFD, metadata,
reference unchanged fields, and historical reconstruction. Evidence is in
`whole-source-recovery/literal-integration-metadata-r2/`,
`whole-source-recovery/whole-regeneration-validation-20260927.json`, and
`whole-installation-result-20260927.json`.

Full CLDF/database builds, compiled identity/graph validation, full tests and
browser QA are **deferred**, not passed. This is complete source-file coverage
at source stage, not a completed production ingestion or release. The next
explicitly authorized build must verify UTN198/JAG198 IDs against the recorded
baseline. No new remote operation or database build was run.

## Historical Appendix E installation evidence

The following earlier account describes the archived wordlist-only stage;
its counts and commands are historical, not current installation instructions.


## Whole-source recovery in progress (2026-09-27)

The historical canonical files below still contain Appendix E only. The
`whole-source-recovery/` directory now holds a complete page census and a
**review-only** proposal for 1,666 rows: the existing 1,438 wordlist rows,
205 narrative/question units, five explicitly named terms, and 18 comparative
attestations. No recovered additions have been installed yet.

All seven target narratives were compared with the original pages. Their 74
expected test answers are separate audit fields; narrative translation
parentheses remain intact. Narrative spelling is preserved, while comparative
`G` has the source's explicit voiced-velar-fricative interpretation. Separate
file-default profiles keep these conventions apart from wordlist IPA.

The old wordlist regenerates exactly. Its previously hidden ingestion Notes
are preserved in a relocation audit so new genuine residual notes can be
public. Existing keys and all other wordlist fields remain unchanged in the
proposal. Current parser checks cover all 1,666 rows without conversion errors;
five focused recovery tests pass. A frozen 20-unit independent sample and 12
additional edge/control units are under review. Bibliography changes are
prepared separately and will accompany accepted source-stage installation.

The first independent audit failed: 13 of 20 sampled units lacked response-scope
tags, and two had legacy wordlist transcription defects (a dropped length mark
and a raised-vowel glide flattened to a consonant). The failure report is
`whole-source-recovery/independent-whole-output-audit-20260927.json`.
These classes require review and repair before a fresh disjoint sample; the
frozen proposal remains evidence of the failed pass. Separately, all 1,666 rows
pass current language/dialect, per-file profile-routing and reference checks in
`whole-draft-metadata-validation.json`.

Running `import_koya.py` without `--install` now writes its CSV, audit and manifest
only to preview paths under `data/tmp/`. It cannot overwrite the installed audit
or manifest. A focused regression verifies this isolation.

Full CLDF, compiled identity/graph/reference checks, the full suite, and
user-triggered browser-database/browser checks remain deferred under the
current execution constraints. The source is not yet complete.

## Historical Appendix E installation

This package extracts Appendix E of *A Sociolinguistic Survey of Koya
Dialects*. The survey was conducted in 1985–1986 and published by SIL
International as **Journal of Language Survey Reports 2021-029**.

Canonical record: <https://www.sil.org/resources/archives/88873>

Canonical PDF: <https://www.sil.org/system/files/reapdata/10/79/91/107991682663957550005886358528752432250/JLSR2021_029.pdf>

PDF SHA-256: `a6541e0d2397849ce7c36961b3849f3b2c1f1c267036cfa1a3f6025796e14e7d`

The PDF is 124 pages. Ordinary text extraction works for the report, but the
phonetic alphabet and wordlists on PDF pages 80–123 (printed pages 75–118) are
raster images. `extract_ocr.py` verifies the pinned PDF, extracts and enhances
the embedded scans, and can reproduce `tesseract_raw.txt`. OCR is retained only
as comparison evidence. It was not used to install an unreviewed form.

`manual_review_data.py` is the authoritative cell ledger. Every one of the
1,840 cells actually printed in the seven target and two control lists was
visually inspected against the enhanced source image. It also records 50
conceptual list slots omitted by the report: Malakanagiri items 61–80 and
Jaganathapuram, Chintoor, and Podia items 201–210. Empty strings are confirmed
ruled blanks or identified omissions, never OCR failures.

`import_koya.py --install` expands the ledger to:

- `data/other/forms/20260828-sil-koya.csv`: 1,438 rows representing 1,401
  attested target cells; source slash alternatives are separate stable rows.
- `data/other/forms/raw_data/20260828-sil-koya-audit.csv`: all 1,890 conceptual
  site × item slots, including 420 excluded Telugu/Oriya controls and 69 missing
  target slots.
- `data/other/forms/raw_data/20260828-sil-koya-manifest.json`: pinned source,
  counts, review policy, and uncertainty summary.

The source gives lexical similarity group numbers, not etymological claims.
All installed forms therefore remain unlinked. The western wordlists overlap
the later Beine/Rama digitization; those values are recorded only in the audit
as alternate evidence after manual source-image checking.

SIL's publication page states a fair-use policy for scholarly research and
instruction. This package checks in extracted lexical facts and audit metadata,
not the copyrighted PDF or page images.

Run the focused checks with:

```sh
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_sil_koya_2021.py
```

Shared registry/build integration is intentionally described in
`INTEGRATION.md` and deferred to the coordinating agent.
