# Sheth integration review — 14 September 2026

**A resolved subset is installed in shared inputs; the dictionary ingestion is
not complete.** The full shared CLDF build/full test suite, unresolved article
segmentation and auxiliary reference catalogue remain open. The app database has
not been refreshed.

## Counts and installed scope

| Unit | Count |
|---|---:|
| Pinned DDSA digital pages | 952 |
| Audited source articles | 41,638 |
| Articles represented in shared inputs | 31,501 |
| Articles held audit-only | 10,137 |
| Original draft candidate rows | 63,741 |
| Installed headword/sense/alternate rows | 42,118 |
| Source-only compiled nodes | 42,118 |
| Explicit alternate-head variant edges | 2,268 |
| Parentless unlinked nodes | 39,850 |

The installed language counts are **41,670 Prakrit (`Pk`), 443 Apabhramsha (`Ap`),
and five Ashokan Prakrit (`As`)**. Existing dialect tags identify 174 Shauraseni,
29 Magadhi and 15 Paishachi rows. No new language, dialect, coordinate, clade or
Glottocode is created. Unsupported/multiple source labels stay under review.

The complete digital snapshot is audited, including the five no-starting-headword
pages 546, 658, 678, 679 and 687. Web page numbers are preserved as web locators;
they are not silently relabelled printed pages. The digital endpoint ends at
page 952 with *hrāsa*. Completeness of the linked printed edition and supplement
is explicitly not claimed.

The selector holds out entire articles when the current parser cannot cleanly
separate their senses, compounds, paradigms, quotations, malformed labels or
untagged citations. This deliberately includes some valid but complex articles;
“audit-only” does not mean illegible. Overlapping exclusion counts include 5,309
articles with unsegmented sentences/abbreviations, 2,947 embedded-subentry or
reference cases, 2,609 morphology cases, 2,347 etymology-scope cases, 1,754
unsegmented quotation cases, 926 outside-sense prose cases and 686 numeric or
parenthetical reference-review cases. There are 119 invalid native/roman head
pairings. Exact reasons and complete raw markup remain in every article's audit.

## Source and editorial decisions

Source: Hargovind Das T. Sheth, *Paia-sadda-mahannavo*, Calcutta, published by the
author, 1923–1928; DDSA digital transcription pinned on 11 September 2026. The
acquisition manifest contains all 952 original page hashes and the known original
PDF artifact hashes. The compressed integration audit preserves every raw lexical
article and is also a reproducible input snapshot. This imports lexical facts;
no open-data licence is asserted for DDSA's transcription and the printed PDFs
are not redistributed. The original edition is retained rather than substituting
the different modern ISJS English translation.

The `sheth-ddsa` profile explicitly preserves the 41 observed romanization symbols
plus word-boundary handling. It keeps long ē/ō, retroflex letters, anusvara,
combining candrabindu and the printed bound-form marker °. No new phonological
interpretation or expansion of an abbreviated bound head is imposed. `Original`
and `Form` retain DDSA's romanization; `Native` retains Devanagari; `Phonemic` is
blank because the source does not supply a separate IPA analysis. No new OCR is
used; upstream transcription errors remain possible and are not silently emended.

Numbered senses and distinct homographs retain immutable page/article/sense/
alternate keys. Printed alternate heads retain explicit within-sense variant
parents. Grammar and language labels belong to their source scope; notably a
whole `<reference>(अप)</reference>` is a language label, while `(अप १२)` remains
a work citation. The existing dialect registry is reused for शौ, मा and पै.
Layout braces joining alternate heads no longer leak into glosses or Notes.

Whole “see/see above” instructions stay in scoped source-reference notes with a
blank lexical gloss when no definition is supplied. They are not automatically
converted to variant, ancestry or borrowing edges. There are 7,992 rows with
source cross-reference notes, including 7,584 blank-gloss rows. Source usage
quotations stay in Notes. The audit retains all cited-work strings, raw markup,
etymological segments and unresolved material. The printed Sanskrit equivalents
or source etymology labels are retained as prose on 35,252 rows; **no historical
match or borrowing is inferred**.

Work-level source tags are now resolved where the surviving frontmatter identifies
the abbreviation (see the source-tag follow-up below). Edition-level auxiliary
bibliography remains unresolved. The earlier frontmatter review
found duplicated scan pages and missing printed reference pages 10–11; edition-
specific abbreviations cannot safely be assigned invented identities. Every
installed row cites the correct primary Sheth DDSA locator. Secondary work
strings and exact locators are preserved in the per-article audit. Unverified codes
are not expanded. The complete reference-catalogue gate remains open.

## Audit and validation

A seeded raw-markup versus proposal review (seed 20260915) found **2/20 material
errors**: an untagged work citation in *tallēsa*'s gloss and “see above” as the
meaning of *aṇōvamiya*. Whole untagged See instructions now become scoped source
notes; unresolved numeric/parenthetical citation text is quarantined. Cross-
references inside one numbered sense are also confined to that sense, and
unscoped See material outside numbered senses is held for review.

The fresh installed-scope sample, seed **20260916**, has **0/20 material errors**
against pinned DDSA markup. It covers native/roman heads, grammar, sense
boundaries, lexical glosses, examples, cross-references, language and primary
locators. This is an integration audit of the digital transcription, not a claim
that twenty printed-book transcriptions were independently verified. Development
failures, resolutions and final checked keys are recorded in
`source_checklists/audits/20260914-sheth-integration-audit.json`; the complete final
sample is in `raw_data/sheth_2026/sample.json`.

Additional edge checks cover the initial homographs and bound heads on digital
page 1, the final page 952, source language labels, typed versus untyped See
instructions, alternate-head braces, native/roman corruption, multiple source
senses, compound etymology scope and the five installed Ashokan rows.

Focused tests exercise the real compiler on every installed row, then the real
source-only CLDF compilation/unification path. All **42,118 nodes and 2,268 variant
edges** survive, with no conversion errors, replacement characters or lost keys.
Persistent IDs survive row reversal and transcription/gloss changes in a temporary
identity registry. These checks do not substitute for global cross-source
reconciliation. The regression set also includes the earlier Ho/Bhumij/Dhurwa
integration to check that appending Sheth has not disturbed it.

Final result: **37 focused tests passed in 5.52 seconds**. The formatted-reference
update adds Sheth while preserving all 635 existing rows (636 total). Shared
compiled forms, edges, source-key records, aliases and the persistent identity
registry match their pre-integration hashes. The installed CSV, compressed audit
and sound profile match the integration manifest. Python syntax and whitespace
checks pass, treating the repository's CSV CRLF line endings as intentional.

## Checklist gates

| Gate | Status |
|---|---|
| Source/version/licence/scope | Digital snapshot pinned and subset/exclusions explicit; printed-supplement reconciliation open. |
| Extraction | Reproducible sequential processing of hash-checked DDSA pages or frozen article audit. No new OCR; scan transcription addendum inapplicable. |
| Files and keys | Dated rich CSV, full article audit, immutable article/sense/alternate keys and explicit installation. |
| Languages/dialects | Canonical existing parents and existing registered literary dialects; no unsupported new geography. |
| Field separation | Validated for selected records; structurally unresolved articles held out. |
| Linguistic structure | Scoped grammar and alternate-head links passed; compounds/paradigms and ambiguous See targets remain open. |
| Sound profile | Explicit preservation route and full installed-corpus coverage passed. |
| References | Primary BibTeX/formatted reference and locators passed; auxiliary catalogue resolution deferred. |
| Graph | Source-only variant and unlinked checks passed; no inferred historical edges. Full-database graph/deduplication checks deferred. |
| Audit | All 41,638 articles accounted for; final fresh 0/20 on installed scope, plus edge checks. |
| Focused tests | Parser, metadata, source coverage, compiler survival, graph and stable-key regression checks pass. |
| Full pipeline | Deferred: full build, full suite, shared compiled diffs and global audit-manifest regeneration. |
| Browser | Not requested for this routine integration; unchanged. |
| Handoff/publication | Counts, exclusions, transcription and unresolved gates recorded; no commit/push/deployment. |

The dictionary/glossary and website-snapshot addenda apply. The source-comparison
addendum applies to preserved etymological prose, with accepted ancestry mapping
and the post-full-build audit still open. Survey-specific addenda are inapplicable.

The workspace resource policy directs full builds/full suites to existing CI or
an authorized runner and permits deferral when no suitable runner is available.
The existing CI only accepts pushes/PRs and cannot run these uncommitted workspace
inputs. No full local build or publication was started to bypass that policy.
Source-only checks are bounded fixtures with unrelated lexical inputs empty;
shared compiled forms, edges and persistent identity state are not rewritten.

## Files and reproduction

- `data/other/forms/raw_data/sheth_integrate.py`: strict selection and explicit installer.
- `data/other/forms/raw_data/sheth_parse.py`: repaired source-language, brace and See scope parsing.
- `data/other/forms/20260914-sheth.csv`: selected shared inputs.
- `data/other/forms/raw_data/sheth_2026/`: all-article audit, manifest, counts, sample and README.
- `conversion/sheth-ddsa.txt`, `make_cldf.py`: dedicated route and append-only input placement, protecting old aliases and homographs.
- `cldf/sources.bib`, `cldf/references.csv`: primary Sheth bibliography and provenance.
- `tests/test_sheth_integration.py`, `tests/test_sound_profiles.py`, `tests/test_manual_surveys_integration.py`: integration and compatibility coverage.
- `README.md`, `audit_source_ingestions.py` and this review: discovery and gate accounting.

Reproduce proposals/installations using the package README. Run focused tests with:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_sheth_parser.py tests/test_sheth_integration.py \
  tests/test_manual_surveys_integration.py \
  tests/test_sound_profiles.py::test_every_installed_source_has_an_explicit_sound_profile \
  tests/test_sound_profiles.py::test_sound_profiles_have_unique_graphemes
```

Representative keys for a future requested app refresh: `sheth1923:p1:e1:s1:v1`
(*a*, first alphabet letter), `sheth1923:p10:e15:s1:v1` (Apabhramsha *antraḍī*),
`sheth1923:p137:e36:s1:v2` (*uarōha*, explicit alternate head), and
`sheth1923:p932:e2:s1:v1` (*sēhara*, numbered sense with a source quotation).
These are input keys, not claimed live app entries.

## Quoted-source tags follow-up — 14 September 2026

The dictionary, website snapshot and source-comparison checklist addenda remain
active. The verified catalogue contains 195 entries manually transcribed from
physical PDF pages 3–9 and 12 of the [DDSA frontmatter](https://dsal.uchicago.edu/dictionaries/sheth/frontmatter/frontmatter.pdf).
The duplicated scans leave printed reference pages 10–11 unavailable. No expansion
is guessed for missing or uncertain abbreviations.

30,147 of the 42,118 installed records now carry quoted-work tags, with 219 distinct
tags emitted (including commentary and nested drama distinctions). The per-row
claim audit records 41,850 resolved and 14,678 unresolved claims; these counts
include repeated claims on alternate heads, not unique quotations. One unscoped
reference on an accepted article remains audit-only. The existing 10,137 excluded
articles remain excluded. All first 14 CSV columns and every pre-existing tag are
unchanged: forms, glosses, languages, source keys and variant relationships match
the pre-tag baseline across all 42,118 rows.

Matching uses only explicit source-reference markup scoped to the corresponding
sense. It excludes etymologies and language labels. Semicolon continuations retain
the work and exact locator; टी/भा indicate commentary, whereas टि marks a variant
reading. The नाट glossary has a separate namespace because its abbreviations can
name different works. Source tags do not infer Sanskrit periods or historical links.
Full work names are registered for frontend pills and source filters. Primary
Sheth bibliography and sound-profile decisions remain unchanged; edition-level
auxiliary bibliography is still unresolved.

The seeded 20-record raw-markup/tag audit found 0 material tag errors. Final installed
claims match the reviewed sample exactly; an additional regression covers numeric
continuations with commentary qualifiers. Evidence and sample are in
[audits/20260914-sheth-source-tags.json](audits/20260914-sheth-source-tags.json).
45 focused parser/integration/profile tests pass. Frontend `npm run check` reports
0 errors and 6 pre-existing warnings. The focused integration tests cover source
compilation and variant relationships. Full CLDF build/full suite remain deferred
under the workspace resource policy: no authorized remote runner is available for
these uncommitted inputs. The browser database is unchanged; browser refresh and
browser QA are not part of this input-only update. No new OCR, language/dialect,
clade, etymology edge, or publication is applicable.

Implementation: `sheth_sources.tsv`, `sheth_sources.py`, scoped parser and installer,
`tags.py`, `tests/test_sheth_source_tags.py`, and frontend `shethSourceLabels.json`
plus `tags.ts`. Regenerate frontend labels with `python3 sheth_sources.py` after
catalogue edits; the focused test verifies that both catalogues agree. Manifest
hashes cover the final inputs, compressed article audit, catalogue and resolver.

## Requested local rebuild — 14 September 2026

The user subsequently requested a full database rebuild and local serving. Ran
make_cldf, link_refs, unify_cldf, assign_form_ids, concepts, align and make_refs
sequentially, then browser SQLite transform/compaction. CLDF retains 42,118 Sheth
rows, 30,147 tagged rows and 2,268 source-local variant edges. The browser source
page displays 42,115 forms after its standard identical-record collapse. Database
quick_check passes; all 219 emitted Sheth tag names are present. The complete
browser DB has 765,603 nodes and 636 references.

Local zstd -6 -T1 packaging produces 69,026,419 bytes from 134,578,176 bytes;
restoration through the browser decoder matches SHA256. Exact dbMeta sizes were
updated, fixing the initial stale-size loader rejection. This local artifact is
over the production size target; production packaging/publication was not requested.
The existing port-5173 server was restarted in tmux. Browser QA passed for the
Sheth source table, expanded work names, separate commentary tags and a real
देशीनाममाला filter returning 4,449 forms. Representative entry:
http://127.0.0.1:5173/entries/f_d6h7tkmrqxfhc .

The post-build focused run passed 33 tests but failed two broader manual-survey
expectations: Rajasthani compiled count 15,887 versus expected 15,876, and existing
source-owned overlay rows. These remain recorded, not silently waived or edited.
Full-suite and full-ingestion completeness remain open. Evidence:
`audits/20260914-sheth-local-build.json`; workspace execution logs:
`../tmp/sheth-local-20260914/{build,database,frontend-check,server}.log`.

## Sanskrit counterpart follow-up

See [the Sanskrit extraction review](20260914-sheth-sanskrit-review.md) for the
26,191 counterpart records, 27,484 source-attributed comparison links, exclusions,
transcription policy and validation. These structure the printed equivalents
without asserting unproven historical ancestry.
