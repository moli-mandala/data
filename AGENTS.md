# AGENTS.md — moli-mandala/data

Agent-facing notes for the CLDF data repo. Read `README.md` first for the raw-data layout and
column definitions; this file covers the things that bit us and aren't obvious from the code.

## What this repo is

The **source of truth** for Jambu's etymological data. Raw dictionary data (CDIAL, DEDR, Munda,
DBIA, "other") lives under `data/`; the pipeline compiles it into a CLDF wordlist under `cldf/`.
The sibling `../jambu-static` consumes `cldf/` to build the browser DB. **Nothing here runs at
serve time** — it's all an offline build.

## Mandatory source-ingestion checklist

For every request to ingest, import, add, OCR, snapshot, re-ingest, or substantially reparse a
lexical/linguistic source, read `SOURCE_INGESTION_CHECKLIST.md` completely before taking ingestion
actions. Announce that it is active, use its standing editorial policy and applicable source-type
addendum, and treat its definition of done as mandatory. Do not silently skip gates. Record
inapplicable or explicitly deferred gates in the audit/final handoff, and do not call an ingestion
complete until the applicable focused tests, full data build, and browser-database QA pass.

Pure source discovery does not activate the installation/build portion until the user selects a
source for ingestion.

## The pipeline (run order matters)

`make all` runs stages 2–7 below in order (then `make_refs.py` and the manual-survey etymology
test gate). Per-source importers, including `make markodi`, are separate targets and are never run
by `make all`. The stages, for when you need to run one by hand:

1. `data/cdial/parse.py` — regenerate `data/cdial/cdial.csv` from the CDIAL HTML.
   **Only when CDIAL parsing logic changes.** Slow; caches to `data/cdial/cdial.pickle`.
   (DEDR has a parallel `data/dedr/parse.py` + `get_params.py`.)
2. `make_cldf.py` — raw `data/**` → `cldf/{forms,parameters,languages,references}.csv`.
   Per-source settings (transcription profile, dedupe-by-key, alternates splitting, excluded
   languages, gloss handling, audit-only notes, reference metadata) come from the YAML beside each
   source CSV (`data/other/forms/<stem>.yaml`; dictionaries: `data/<dict>/source.yaml`) via
   `source_meta.py`. Do not add `if source_key == …` branches; add a YAML key. Validate with
   `uv run python source_meta.py`. Appended-source ordering is declared there too
   (`defaults.identity.legacy_ids: stem` + `append_order`).
3. `link_refs.py` — resolve `<smallcaps>` cross-references in the descriptions to
   `<a data-entry="ID">` markers; also touches `derivation.csv` / `merges.csv`. Idempotent.
4. `unify_cldf.py` — fold `parameters.csv` (etyma) + `forms.csv` (reflexes) into ONE unified
   `cldf/forms.csv`, then **delete `parameters.csv`**. Applies the section-restructure, borrowed
   forms, and merges.
5. `assign_form_ids.py` — replace order-dependent IDs on attested forms with persistent `f_…` IDs,
   rewrite graph references, preserve old IDs in `cldf/form-id-aliases.csv`, and apply the curated
   etymology sidecars (`etymology_assignments.py`: one `data/other/forms/etymologies/<source>.csv`
   per lexical source, plus `data/<dictionary>/etymologies.csv` for dictionary-entry children).
   Rows in the `_pending.csv` inbox are filed under the source that owns their child on every
   build. Its committed `data/form-identities.csv` registry is
   identity state: do not regenerate or discard it during a re-ingestion. Rich importers' immutable
   `Entry_Key` values reach this pass through generated `cldf/form-source-keys.csv`.
6. `align.py` — phonetic final-origin→child alignments → `cldf/alignments.csv`. Approximate/computed
   layer, tuned for Indo-Aryan. Reads unified `Origin_ID` relationships, so it **must run last**.

Then, in `../jambu-static`: `npm run db:transform` reads `../data/cldf` directly.

## Importers, scratch, and research passes

- Regenerate a source CSV with `make ingest SOURCE=<stem>`; `make sources` lists the sources whose
  YAML declares `defaults.importer.commands`. Do not add per-source Makefile targets.
- Validate an etymology-lab research pass with `make check-pass DECISIONS=<json> PASS=<name>` and
  save it with `make save-pass DECISIONS=<json> PASS=<name> NOTE="…" AUTH="…"` (both wrap
  `etymology_lab.py`). The helper performs the historical `*_save.py` checks, files rows into the
  per-source sidecars, and records sha256 ledgers instead of copying overlay files into `backups/`.
  Do not write new `*_save.py` scripts.
- `make check-sources` validates the per-source YAML settings and sidecar placement; run it before
  a build and after saving a pass.
- `make clean-scratch` at the workspace root shows (and with `FORCE=1` deletes) task scratch under
  `tmp/`, `data/tmp/`, and stale `.dbwork` builds; `make prune-lfs` drops superseded LFS blobs.

## Run incantations

The repo env (`pyproject.toml`/`uv.lock`) carries everything the pipeline and both dictionary
parsers need, lxml included, so plain `uv run python …` works; `make` targets wrap the common
paths:

```bash
make all                      # parsers (only if stale) + every stage → cldf/
make forms                    # parsers + make_cldf → assign_form_ids: the content-only build (~3× faster; no alignments/refs)
make cdial | make dedr        # re-parse one dictionary (only when parse.py / helpers / the page snapshot changed)
make parser-diff P=cdial      # what a parser change did: glosses filled/blanked/edited vs HEAD, in seconds
cd data/cdial && uv run python parse.py --entry 10132 134   # one entry, rows to stdout, <1 s
cd data/dedr  && uv run python parse.py --entry 360 a12     # DEDR appendix entries are a<number>
```

The parser CSVs (`data/cdial/cdial.csv`, `data/dedr/dedr_new.csv`) are make targets with real
dependencies, and each parser writes to a temp file and renames on success, so a crashed parse
never leaves a stale CSV that `make all` would quietly build from. Iterate on a parser with
`--entry` and `make parser-diff`; run `make forms` to see the result in `cldf/forms.csv`; run
`make all` once at the end.

### Gotcha: parse.py silently drops the HTML wrapper without lxml
BeautifulSoup **without** `lxml` installed falls back to a parser that strips the outer
`<html><body>` wrapper, so each CDIAL entry's `Description` starts with `<number>` instead of the
full entry HTML. This is silent — the run "succeeds" — and later manifests as **all CDIAL
etymology vanishing** on the site. lxml is pinned in `pyproject.toml` for this reason; if you run
the parser outside the repo env, add `--with lxml`. (Downstream, `unify_cldf.py`
guards with `is_html = header.lstrip().startswith("<")`, but don't rely on that; parse it right.)

## Data-model invariants (edge model, 2026-08)

- **One row per node** in `cldf/forms.csv` (content only: no graph columns). Parentless nodes
  carry `Status` — `entry` (curated etymon) or `unlinked` (unetymologised import); attested
  rows have an empty Status. `Redirect` (addendum → main) stays on forms.csv.
- **All ancestry/derivation relations live in `cldf/edges.csv`**: `(Child_ID, Parent_ID, Kind, Rank, Pos, Source,
  Note)` with Kind ∈ {reflex, borrowed, variant, component, derived}. Rank 1 = the accepted
  etymology (≤1 per node across reflex/borrowed/variant); Rank ≥2 = alternate hypotheses
  (`Note` carries `review:*` markers for auto-classified ones). `Pos` orders compound members
  on `component` edges. A variant's rank-1 edge points at its **true target** (parent or
  sibling); the etymon is reached transitively — there is no separate Variant_Of pointer.
- Nuristani/CDIAL editorial groupings use the existing rank-1 `reflex` slot with
  `grouping:cdial` in the edge Note and `etymology-group` in Tags. They assert neither inheritance
  nor borrowing. `nuristani_grouping.py`, called after curated assignments, keeps mapped PNur
  reconstructions and attestations as siblings under CDIAL; obsolete blank PII nodes redirect.
- Source-attributed article comparisons that do not assert an accepted ancestry relation live in
  `cldf/comparisons.csv`, never `edges.csv` or ordinary reflex rows. Their direction and confidence
  describe the printed claim and may remain explicitly undetermined/low.
- `unify_cldf.py` still synthesizes the legacy columns internally; `edges_build.py` is the
  serialization boundary (classification rules + invariants live there, cross-checked by
  `tests/test_edges.py` via `unify_cldf.py --legacy-cols`). Read edges with `edges_util.py`
  (`rank1_map`, `effective_etymon`, `aligned_parent`, `attach_legacy_graph` shim).
- The etymology overlay is keyed `(Form_ID, Etymon_ID)` with `Kind/Rank/Status/Source/Notes/Pos`
  and split into per-source sidecars next to the source CSVs (`data/other/forms/etymologies/`);
  a row belongs to the sidecar of the source that owns its *child*. Read them with
  `etymology_assignments.read_assignments()`, write with `write_assignments(rows, SidecarResolver())`,
  and audit placement with `uv run python etymology_assignments.py check`. `Status=rejected` deletes a
  generated hypothesis edge.
- `make_cldf.py` merge-joins are **order-deterministic** (dict.fromkeys) — set-based joins
  once re-minted ~650 durable f_ ids per rebuild; `reconcile_form_ids.py` repaired the
  historic drift once and is a no-op on current builds.
- **Section forms**: CDIAL entries with numbered sub-headers (`2. *kṣata-²`, `3. …`) are promoted
  to their own entries with IDs `{cdial-id}-{n}`, derived from the head via a `derivation.csv`
  edge (their `Origin_ID` is NULL — they're etyma). Reflexes are re-homed onto them by the `info`
  half of their cognateset (see below). ID collisions get an `x` suffix appended.
- **Non-CDIAL reflex IDs** are `{param}-{n}` (e.g. `m1-1`, `d1-1`), NOT `{file}-{row}`. But CDIAL
  numeric etyma (`re.fullmatch(r"\d+[a-z]?", epid)`) keep `{file}-{row}` — otherwise "other"-source
  reflexes hung on numeric CDIAL etyma collide with section-form IDs like `3643-2`.
- **cognateset = `subnum:info`**. `subnum` is `parse.py`'s paragraph counter; `info` is the form
  number and is what re-homes a reflex to its section. Borrowed rows are tagged
  `subnum:<parent-lang> →`. Non-numeric `info` carries forward under the most recent numeric section.

## Tags & Sanskrit eras

`tags.py` lifts a leading `;`-delimited run of structured tokens out of `Description` into `Tags`,
**only when the whole field is known tokens** (so prose is never mangled). Categories: gender,
grammatical, **source** (every Sanskrit-work abbreviation in `sanskrit.txt` + a few lexicographers),
and **era** (a cited work also contributes Early-Vedic / Late-Vedic / Epic / Classical / Medieval).

- `sanskrit.txt` — the abbreviation list (all become source tags).
- `sanskrit_meta.tsv` — hand-authored table: FullName, [Author], era, genre (~285 rows).
- `sanskrit_works.tsv` — **generated** abbrev→era mapping. The generator did diacritic-fold +
  sandhi (aupaniṣad→opaniṣad) + prefix + author matching to reach ~74 rows.
- The era tag set here must stay in sync with `../jambu-static/src/lib/tags.ts` (`ERA_TAGS`).

## `cldf/languages.csv` is a hand-edited source, not generated

`make_cldf.py` **only reads** `languages.csv` (Clade column included) — it never rewrites it. To
change clades (e.g. the "Early NIA" grouping of the Old {Punjabi, Bengali, Assamese, Maithili,
Awadhi, Hindi, Marwari, Gujarati, Marathi, Sinhala} lects), edit the CSV directly. Clade **names**
must match `../jambu-static/src/lib/clades.ts` + `cladeTree.ts`.

## Shipping to prod

The compiled DB is a **GitHub release asset**, not committed:
1. In `../jambu-static`: `npm run db:transform` → `.dbwork/jambu.db`.
2. Upload that as `jambu.db` on a fresh release of the `jambu` repo (the deploy workflow's
   `STATIC_DB_URL` points at `releases/latest/download/jambu.db`).
3. Commit + push `cldf/` here **only when the user asks.**

## Local resource budget (8 GB RAM laptop)

- Default to focused tests and small smoke checks locally. Run required full data builds,
  full test suites, production prerendering, and maximum-compression packaging in existing
  CI or an authorized remote environment. Relocate required gates; do not silently skip them.
- Before starting expensive work, inspect existing jobs and reuse verified artifacts/checks
  when their inputs are unchanged. Batch source changes into one full rebuild.
- Run at most one heavy local job at a time across this workspace. Do not overlap database
  generation, full tests, compression, and production builds. Do not stop another task's jobs
  without establishing ownership or authorization.
- If a heavy local run is necessary, explain why and run it sequentially with one worker/thread
  where supported. Avoid automatic all-core compression and high-memory compression settings
  locally. If required asset size/codec gates need those settings, package remotely instead.
- Prefer streaming reads, scoped SQL queries and bounded samples over loading multiple complete
  datasets into memory. Reuse one dev server and browser tab; avoid duplicate database loads.
- CPU priority (`nice`) does not limit RAM, and Node heap limits do not cap total process memory.
  Do not promise a memory ceiling without measuring and enforcing it.
- Use existing authorized CI for remote work; do not invent a cluster destination, incur new
  paid infrastructure, or publish unfinished changes merely to offload a check. If no suitable
  runner is available, report the deferred full gate and continue lightweight work.
