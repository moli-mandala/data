# AGENTS.md — moli-mandala/data

Agent-facing notes for the CLDF data repo. Read `README.md` first for the raw-data layout and
column definitions; this file covers the things that bit us and aren't obvious from the code.

## What this repo is

The **source of truth** for Jambu's etymological data. Raw dictionary data (CDIAL, DEDR, Munda,
DBIA, "other") lives under `data/`; the pipeline compiles it into a CLDF wordlist under `cldf/`.
The sibling `../jambu-static` consumes `cldf/` to build the browser DB. **Nothing here runs at
serve time** — it's all an offline build.

## The pipeline (run order matters; it is NOT in the Makefile)

The `Makefile` only knows `make_cldf.py`. The current pipeline is a sequence of scripts you run
**by hand, in this order**:

1. `data/cdial/parse.py` — regenerate `data/cdial/cdial.csv` from the CDIAL HTML.
   **Only when CDIAL parsing logic changes.** Slow; caches to `data/cdial/cdial.pickle`.
   (DEDR has a parallel `data/dedr/parse.py` + `get_params.py`.)
2. `make_cldf.py` — raw `data/**` → `cldf/{forms,parameters,languages,references}.csv`.
3. `link_refs.py` — resolve `<smallcaps>` cross-references in the descriptions to
   `<a data-entry="ID">` markers; also touches `derivation.csv` / `merges.csv`. Idempotent.
4. `align.py` — phonetic etymon→reflex alignments → `cldf/alignments.csv`. Approximate/computed
   layer, tuned for Indo-Aryan. Reads `parameters.csv`, so it **must run before `unify_cldf.py`**.
5. `unify_cldf.py` — fold `parameters.csv` (etyma) + `forms.csv` (reflexes) into ONE unified
   `cldf/forms.csv`, then **delete `parameters.csv`**. Applies the section-restructure, borrowed
   forms, and merges. **Always last.**

Then, in `../jambu-static`: `npm run db:transform` reads `../data/cldf` directly.

## Run incantations

Use `uv run --with …` — the repo's own env is stale. The exact sets that work:

```bash
# make_cldf.py
uv run --with segments --with unidecode --with tqdm python make_cldf.py

# data/cdial/parse.py  —  lxml is REQUIRED (see gotcha below)
uv run --with beautifulsoup4 --with lxml --with tqdm python parse.py

# link_refs.py / align.py / unify_cldf.py — plain, stdlib-only (align may want tqdm)
uv run python unify_cldf.py
```

### Gotcha: parse.py silently drops the HTML wrapper without lxml
BeautifulSoup **without** `lxml` installed falls back to a parser that strips the outer
`<html><body>` wrapper, so each CDIAL entry's `Description` starts with `<number>` instead of the
full entry HTML. This is silent — the run "succeeds" — and later manifests as **all CDIAL
etymology vanishing** on the site. Always include `--with lxml`. (Downstream, `unify_cldf.py`
guards with `is_html = header.lstrip().startswith("<")`, but don't rely on that; parse it right.)

## Data-model invariants (the unified `cldf/forms.csv`)

- **One row per node** in the etymon graph. Etyma have empty `Origin_ID` and empty `Relation`.
- `Origin_ID` — self-referential FK to a form's parent/etymon.
- `Relation` — `reflex` (daughter-language reflex), `variant` (same-language non-headword form),
  `borrowed` (cross-dictionary loan; `Origin_ID` points at the **source reflex**, not the etymon,
  and `Borrowed_From` records it — so ancestry chains recurse correctly), or `''` for etyma.
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
