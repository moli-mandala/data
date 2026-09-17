# Ho, Bhumij and Dhurwa shared integration — 14 September 2026

The three source packages are installed in the shared lexical inputs, bibliography,
dialect registry and conversion routes. **The full database ingestion is not yet
complete:** the repository-wide build/full test suite remain deferred. A bounded
source-only compilation and graph check validates these inputs. Browser refresh
was not separately requested and is outside this routine integration.

## Counts and exclusions

| Source | Audited cells | Installed input rows | Target dialects | Bounded compiled nodes |
|---|---:|---:|---:|---:|
| Varenkamp 2024, Ho, Appendix D.3 | 5,670 | 2,900 | 14 under `ho` | 2,900 |
| Bailey & Maggard 2015, Bhumij, Appendix B.3 | 3,780 | 2,100 | 10 under `mu` | 2,100 |
| Joseph & Joseph 2021, Dhurwa, Appendix B | 1,000 | 809 | 4 under `Parji` | 809 |
| Total | 10,450 | 5,809 | 28 | 5,809 |

Ho excludes 2,730 control/republication cells, 38 target blanks and two ambiguous
target readings (item 167/HKE and item 179/HKA). Its third ambiguous reading belongs
to an excluded control. Bhumij excludes 1,680 control cells and 46 target blanks;
46 expanded alternatives retain explicit source-local variant-parent keys. All
1,050 cells from five same-elicitation lists reprinted in Ho remain audit-only in
that later publication. Dhurwa excludes three target blanks and all 200 cells of
the unidentified fifth column, including 199 responses and two blanks.

The sources make no historical cognacy or borrowing claims. The bounded graph
contains 5,763 unlinked nodes and 46 Bhumij variant nodes/edges, with no ancestry,
borrowing, derived or component edges. Cross-source merging in the complete
database has not been tested; these are source-only node counts.

## Source, transcription and metadata decisions

The exact publisher PDFs, acquisition manifests, manual ledgers and exclusion
audits remain frozen in `raw_data/sil_ho_2024`, `sil_bhumij_2015` and
`sil_dhurwa_2021`. Original extraction authority is cell-by-cell manual review of
rendered pages; no OCR supplies installed readings. Only extracted lexical facts
are included; the PDFs are not redistributed and no open-data licence is asserted.

`integrate_manual_surveys_2026.py` verifies the staged file hashes and adapts the
headerless rich CSVs without changing entry keys. Source ISO `unr` maps to Jambu
Mundari `mu`; `pci` maps to existing Parji/Duruwa `Parji`. No new parent languages
or clades are created. All 28 dialects retain source labels and qualified tags;
coordinates and dialect Glottocodes are blank because precise locality evidence
is absent. The existing language-level Glottolog points are not copied to sites.
Udala's explicit `Mundari? Bhumij?` label remains unresolved, flagged `uncertain`
with a typed dialect-mapping reason in the integration audit.

Ho uses an explicit NFC preservation profile, including commas, question-mark
glottal notation, underlined dental letters, superscripts and word boundaries.
Bhumij uses the frozen `sil-bhumij` profile. Dhurwa uses its complete reviewed
profile (`ʈ/ɖ/ɳ → ṭ/ḍ/ṇ`, `dʒ → j`, `j → y`, colon length → `ː`). Source forms
remain exact in `Original` and `Phonemic`; those fields intentionally preserve the
source's phonetic transcription independently of display conversion. Similarity
group labels stay in audit data rather than Cognateset or ancestry edges.

The integration restores the Bhumij source's two lexical `small` qualifiers and
its legible item 195/LAD `(?)` qualifier; the latter has typed source uncertainty.
Extraction instructions and redundant page/site prose are removed from lexical
Notes but retained verbatim in the per-record integration audit. Source-local
variant keys are preserved; comma punctuation is not expanded a second time.

Primary cover checks corrected Dhurwa's second author to **Selvi Joseph**. The
already frozen citation key `josephmichael2021dhurwa` is preserved. Ho's printed
item 93 is **tail**, not the staged **tall**: all fourteen target glosses are
corrected with the original gloss and exact page evidence retained in the audit.
No lexical transcription or historical interpretation was guessed.

## Audit and validation

- Existing extraction and frozen-contract tests: **117 passed**.
- Final focused integration, dialect-registry and profile checks: **19 passed**
  in 1.46 seconds; **136 focused checks passed in total**. The one deselected
  dialect test reads the entire compiled corpus and remains part of the deferred
  shared-database validation; all new rows have independent registry coverage.
- Fresh seeded frozen-ledger-to-installed audit: **0/20 material errors for each
  source**, seed 20260915, after the source-page gloss correction. Reproduce with
  `python data/other/forms/raw_data/integrate_manual_surveys_2026.py --audit-seed
  20260915 --audit-output source_checklists/audits/20260914-manual-surveys-sample.json`.
  This tests integration against the authoritative manual ledgers; it is not
  presented as a fresh independent re-transcription of all sampled PDF cells.
- Primary images inspected separately: covers of all three sources; Ho physical
  pp. 72, 102 and 141; Bhumij pp. 34, 52 and 76; Dhurwa pp. 17 and 21. These check
  authorship, first/last-page boundaries, source qualifiers, retained script
  distinctions, repeated-list handling and the Ho item-93 correction.
- Focused integration tests exercise the real `make_cldf.main()` on a bounded
  fixture with empty unrelated lexical inputs, followed by `unify_cldf.main()`.
  All 5,809 keys survive; 46 variant parents resolve; no replacement characters
  or conversion errors occur. Stable IDs survive row reversal and source/display
  transcription and gloss corrections in a temporary registry. The shared
  persistent identity registry is never rewritten by these tests.
- The three new input files are appended after all existing build inputs, using
  filename-based temporary IDs, so their addition cannot renumber old legacy
  source aliases. Every source has explicit profile routing and corpus coverage.
- `make_refs.py` produces three complete references without changing existing
  reference rows (632 retained, 635 total). Dialect registration adds 28 rows
  while preserving all 2,056 existing rows. Manual provenance, inclusion boundaries, editor credit,
  `OCR=No` and `Etymology_Provenance=none` are explicit.

## Deferred gates and scope

The workspace's 8 GB resource policy directs full builds/full suites to existing
CI or an authorized remote runner, and says to defer full gates when no suitable
runner is available. The repository's available CI only runs on pushes/PRs; no
runner for these uncommitted workspace inputs is configured. Initial local free
space was about 5.8 GiB. No full local rebuild, full suite, publication or new
remote infrastructure was started merely to offload validation.

Therefore `make all`, the repository-wide pytest suite, shared compiled-source
survival/diff/graph checks, and retrospective global manifest regeneration remain
open. The existing shared compiled forms, edges, source-key registry, ID aliases,
identity registry are unchanged against pre-integration hashes. This task has not
rewritten concepts or alignments. The browser database
and live app are not refreshed by this routine integration; browser integrity,
size and representative app QA remain deferred. No commit, push or deployment.

Representative installed records for the next app check: `ho2024-hth-i001`
(*homo*, body), Ho item 93 (*tail*), `bhumij1989-ladhiramsai-i195-a01`
(the source-questioned response), `bhumij-mundari1989-udala-i001-a01`
(mixed language label), and `dhurwa2021:p017:i001:TIR:a1` (*men*, body).
Reference IDs are `varenkamp2024ho`, `baileymaggard2015bhumij`, and
`josephmichael2021dhurwa`. These are installed-input examples, not claimed live
app entries.

Survey-wordlist and manual/OCR-review addenda apply. Dictionary, website/API,
upstream CLDF and etymological-source addenda are inapplicable; the preserved
lexical variant relationships do not assert etymological ancestry.

## Checklist gate accounting

The following applies to each of the three sources, with source-specific counts
and exceptions above. The frozen package checklists retain the extraction-stage
evidence; this review records the subsequent shared integration.

| Checklist section | Status and evidence |
|---|---|
| 1. Source and scope | Passed: pinned PDF versions and hashes, full cell counts, exact appendices and excluded controls; only extracted lexical facts included. |
| 2. Extraction | Passed using frozen manually verified image transcriptions; PDF text/OCR supplied no installed reading. Publisher PDFs and temporary renders are not added to source control. Website/API snapshot and upstream CLDF extraction are inapplicable. |
| 3. Files and identifiers | Passed: explicit adapter installation, dated rich CSVs, immutable keys, generated per-row integration audits and preserved identity registry. |
| 4. Languages and dialects | Passed: existing canonical parents, 28 registered qualified sites, source aliases and locations; unsupported coordinates/Glottocodes blank. No new parent language or clade. |
| 5. Rich schema | Passed: all 15 columns, NFC forms, separate source/display layers, clean glosses, structured citations, residual lexical qualifiers in Notes. |
| 6. Structured information | Passed for source dialects, locators, variants and uncertainty. No source grammatical/register labels or explicit donor claims require new tags. Source similarity groups remain audit-only. |
| 7. Sound profiles | Passed: three explicit routes, complete installed-corpus conversion checks and difficult-symbol regression cases. |
| 8. References | Passed: all three source records and formatted references, exact inclusion boundaries, manual provenance, editor credit and OCR flags. |
| 9. Graph | Passed on the bounded source fixture: 46 variant edges and no historical claims; all other rows unlinked. Full-database graph validation deferred. |
| 10. Audit | Passed for integration: exhaustive frozen cell ledgers plus 5,809 adaptation rows, exclusions reconciled, seeded 0/20 per source and primary-image edge checks. The fresh sample checks frozen raw manual transcription against installed output. |
| 11. Focused tests | Passed for extraction contracts, metadata, profiles, all-record compiler survival and stable IDs. |
| 12. Full pipeline | Partial: dry proposal reviewed, explicit installation and focused checks completed; full build, full suite, global generated diffs and cross-source deduplication checks deferred under the resource policy. |
| 13. Browser QA | Inapplicable to this routine ingestion because no separate browser refresh was requested. App DB remains unchanged. |
| 14. Handoff | Counts, exclusions, unresolved cases, transcription decisions, audit, tests and changed-file routes recorded here. Live app examples await a requested refresh. Publication is outside scope. |

The survey addendum is satisfied by the complete concept/site matrix, exact
blank/control dispositions and variant accounting. The manual/OCR addendum uses
the frozen image-review ledgers and retained non-authoritative OCR evidence;
there are no unreviewed OCR readings installed by this integration.

## Files changed and reproduction

- New inputs: `data/other/forms/20260914-sil-{ho,bhumij,dhurwa}.csv`.
- Adapter: `data/other/forms/raw_data/integrate_manual_surveys_2026.py`;
  adjacent `20260914-sil-*-integration-audit.csv` files and the seeded audit
  under `source_checklists/audits/` account for every installed record.
- Shared routing: `make_cldf.py`; new `conversion/sil-ho.txt` and
  `conversion/sil-dhurwa-2021.txt`. Existing `conversion/sil-bhumij.txt` is unchanged.
- Metadata: `cldf/dialects.csv`, `cldf/sources.bib`, `cldf/references.csv`.
- Regression coverage: `tests/test_manual_surveys_integration.py` and
  `tests/test_sound_profiles.py`; existing source extraction tests are unchanged.
- Discovery and handoff: `README.md`, `audit_source_ingestions.py`, the SIL
  source census and the three packages' README/CHECKLIST/INTEGRATION notices.
  Frozen manifests, staged inputs and manual ledgers remain unchanged.

From the data repository, reproduce the focused final checks with:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_manual_surveys_integration.py tests/test_dialects.py \
  tests/test_sound_profiles.py::test_every_installed_source_has_an_explicit_sound_profile \
  tests/test_sound_profiles.py::test_sound_profiles_have_unique_graphemes \
  -k 'not every_form_language'
```

The same adapter's explicit `--install` flag reproduces all three shared inputs
and integration audits. Source extraction remains reproducible through each
frozen package's existing importer. Run `make_refs.py` after bibliography changes.
