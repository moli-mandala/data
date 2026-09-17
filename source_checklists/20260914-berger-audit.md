# Berger audit — 14 September 2026

**Berger needs repairs to lexical identities, parsing and grammatical structure.**
The importer is reproducible, but it reproducibly preserves errors that the current
tests do not catch. This was an audit only: source rows, importer, translations,
identity registry, cognate catalog and compiled graph were not changed.

## Findings, in priority order

### 1. High: reused source keys attach different words to accepted cognate sets

The identity crosswalk uses page-order alignment with form similarity as low as
0.55. It can give a new, unrelated article an old key. The curated cognate catalog
then resolves that key successfully while retaining the old article's meaning.
All four examples below were checked against the source image and the current
compiled graph. These are existing accepted grouping errors, not hypothetical
failures on a future rebuild.

| Source key | Current word / meaning | Accepted grouping still means | Compiled form ID |
| --- | --- | --- | --- |
| `berger-entry-1712` | ćhanjíśs — an udder disease | back of the head | `f_z5xvu6csjlvds` |
| `berger-entry-2920` | gayú — partridge | part of a mill | `f_dmns7ggrc6lne` |
| `berger-entry-7051` | phúliś — a flowering herb | feather | `f_3hucpcn7vqt32` |
| `berger-entry-7630` | qhurdá — bad, superfluous, unsuitable | a head cold | `f_xfdmmpvymgge2` |

The stored spellings in this table may themselves need OCR repair. The images
establish that the current lexical meanings differ from their grouping meanings.

Among the 1,261 Berger keys used in the cognate catalog, **157 have a different
normalized form from the pinned legacy record**, spanning 149 sets. Of those,
87 have character similarity below 0.8. These are review candidates, not 157
proven errors: genuine OCR corrections and boundary repairs also change forms.
Across the complete 7,173-row crosswalk, 652 matches score below 0.8.

The 597 compatibility rows preserve absent old evidence keys, but do not protect
an old key that has already been reassigned to a different article. See
[`sequence_map`](../data/other/forms/raw_data/berger_cleanup.py#L751),
[`build_identity_map`](../data/other/forms/raw_data/berger_cleanup.py#L794), and
[`catalog_preservation_rows`](../data/other/forms/raw_data/berger_cleanup.py#L1143).

**Repair:** review the questionable mappings against article position, complete
form and meaning; restore the old identity to its lexical referent and update
the affected catalog bindings. Preserve public IDs and aliases deliberately.
Merely regenerating keys or raising the similarity threshold will not repair
the already-accepted graph.

### 2. High: class, plural and POS labels are lost or applied to the wrong form

**None of the 10,703 installed Berger rows has a Burushaski noun-class tag.**
The current compiled Berger forms also have none. The German OCR audit preserves
some labels, but the source CSV has no complete grammatical-note field analogous
to the repaired Yoshioka data.

The grammar function searches the first 350 characters of an entire article.
A plural paradigm or later verbal subentry can consequently change the headword's
tags. See [`_grammar_tags`](../data/other/forms/raw_data/berger.py#L501).

Image-verified examples:

- **trin**, printed Y-class noun “favour”, is tagged `verb` and translated “please”.
  The article's later **trin ét-** subentry supplies the misleading verbal evidence.
- **aadát** and **agón** receive `pl` because their articles print plural forms.
  Those labels describe the listed plurals, not a plural-only headword.
- **ámin / ámis / ámit** and their plural forms have explicit human/X/Y class
  distinctions in the source. The main row is reduced to `noun pl`; those class
  distinctions are not represented in canonical tags.
- **halċík** is explicitly Nager, but the installed row is tagged Hunza only;
  its printed inflectional string is absent from notes.

There are 750 plural-tagged rows and 225 singular-tagged rows, but those counts
must not be read as validated grammatical analyses. A simple source-text screen
finds class-label candidates in 827 non-excluded audit records; this is a locator
count and includes possible non-headword labels.

**Repair:** retain the complete printed grammar, extract full attested paradigm
forms, and scope POS, number, class and dialect tags to each form. Keep suffix-only
paradigms as source notes rather than constructing unprinted forms.

### 3. High: fresh image review finds lexical and translation errors

A fresh deterministic sample of 20 installed primary source articles (seed
**20260914**) has **11 confirmed material errors**, two cases needing closer glyph
review, and seven without a confirmed material error. Material errors include
incorrect forms, meanings, grammatical scope or dialect assignment. This is a
diagnostic sample, not an estimate of the corpus-wide error rate.

Examples:

- **dumm** is German prose inside another entry's definition, but is installed
  as a Burushaski word (`berger:p097:c1:e014`).
- **phaláan** loses its “so-and-so / thingummy” definition; its gloss consists
  of alternate forms, punctuation and a broken fragment (`berger-entry-6704`).
- **luthúri gaíṅ**, a grape name, is split: only the first word remains in Form,
  while the second is stranded in Gloss (`berger-entry-5560`).
- **gaáśng** includes the source's `ng.` dialect label in the headword
  (`berger:p164:c1:e011`).
- **zanqán** absorbs the next **zanzán** index entry (`berger-entry-10437`).
- A printed target's superscript **2** becomes a question mark, losing its
  homonym identifier and adding false uncertainty (`berger-entry-5076`).

The full sample and per-entry decisions are in
[sample.csv](audits/20260914-berger/sample.csv). The older saved 20-entry sample
records zero errors, but it does not establish that the remaining import is clean.

**Repair:** address entry boundaries, compounds, affixes, dialect labels and
homonym numbers before retranslating. Then review translations against the German
definitions and run a fresh sample after the fixes.

### 4. Medium: 78 source relations point to excluded entries

There are **36 variant and 42 derivation relations** whose targets are absent from
the installed rows: 78 affected children, pointing to 59 distinct missing parents.
All 59 parents are present in the old audit with status `excluded`.

`resolve_key` searches parsed entries, while `import_rows` subsequently filters
some of those entries out. The resulting relation can therefore name an entry
that was never emitted. In the current compiled graph, 72 affected children have
no accepted parent; six have another `reflex` parent. The source-local variant
graph contains no cycles, but endpoint preservation fails.

See [missing-relations.csv](audits/20260914-berger/missing-relations.csv) and
[`resolve_key` / `import_rows`](../data/other/forms/raw_data/berger_cleanup.py#L1112).

**Repair:** resolve relationships against accepted emitted entries, then review
excluded-parent cases. Restore a parent only when its source reading is supported;
otherwise record the unresolved relation explicitly.

### 5. Medium: reference-only glosses and incomplete audit reporting

- **1,674 rows** have glosses beginning with a see/equivalence expression. This
  is a screening count, not 1,674 independently verified resolvable references.
  Printed targets and homonym numbers should be preserved before matching them.
- One hand-entered row, `berger:gold:cdial11406:balanees-man`, has a blank gloss
  and `?` in Notes. It has no matching row in the 20260828 main audit. The older
  39-row grammar audit is a separate artifact, so this is a main-audit coverage gap.
- The generated checklists say “unresolved cases: none detected”; the source audit
  actually records **939 excluded records**, including missing/stale translation
  and damaged-form cases. Those states should be surfaced in the review summary.
- The source is deliberately marked uncertain: 10,702 of 10,703 installed rows
  carry `uncertain`. That records risk but does not resolve any of the issues above.

## What passes

| Check | Result |
| --- | --- |
| Installed rows | 10,664 auto + 39 hand-entered = 10,703 |
| Auto compatibility tranche | 597 of the 10,664 auto rows |
| Raw language codes | 9,249 Bur; 1,454 Werch (Yasin) |
| Source coverage accounting | 9,700 units; 11,039 parsed records; 11,636 audit rows including compatibility records |
| Page inventory | Printed pp. 9–486; duplicate spread and appendix exclusions match the manifest |
| Cached regeneration | Exact match to both installed files; zero changed rows or keys |
| Keys and schema | All 10,703 keys unique; all rows 15 columns and NFC |
| Cognate evidence presence | All 1,261 Berger catalog keys exist; presence does not prove correct identity |
| Compiled identity coverage | All 10,703 active Berger source keys have compiled forms |
| Sound-profile token coverage | No uncovered forms; passthrough OCR characters still require linguistic review |
| Focused tests | **33 passed** |
| Source mutations during audit | None; nine source/artifact hashes verified unchanged |

The tests were `test_berger.py`, `test_berger_cleanup.py`,
`test_burushaski_cognates.py` and `test_burushaski_comparisons.py`. They currently
validate structure and selected historical examples, not the failures found here.

## Recommended next work

1. Repair the incorrect article identities and cognate bindings first; these
   already affect accepted groupings in the compiled graph.
2. Repair segmentation and grammatical/paradigm extraction, preserving class and
   plural labels; then review translations and the 939 exclusions.
3. Repair missing relation targets and resolve reference-only index entries with
   source evidence. Add regression checks, repeat fresh image QA, and run the
   later integration build as one batch.

No importer fix, catalog reassignment, full build, full-suite run, browser refresh
or publication was performed for this audit. The source PDF hash matches the
20260828 manifest. The scan is not redistributed. Machine-readable evidence,
source hashes and reproducible diagnostic scripts are in the
[audit package](audits/20260914-berger/README.md).
