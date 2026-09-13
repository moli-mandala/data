# Selected surveys ingestion review — 2026-09-11

Selected by the user: Angika (Regmi 2017), Majhi (Chalise 2014), The Koraga
Language (Bhat 1971), and Linguistic Survey of India: Orissa (2002). The standalone
Gondi–Kui annotated Swadesh source is excluded. No Gondi or Kui rows are added.

The mandatory SOURCE_INGESTION_CHECKLIST.md is active, with survey/comparative-table,
dictionary/glossary, OCR-heavy, and etymological/comparative addenda as applicable.
The canonical data are installed. The full repository validation gate remains open
for the pre-existing test failures described below.

## Source scope and provenance

- Angika: report dated April 2017, by Ambika Regmi; comparative wordlist pp. 83–90,
  210 prompts at Omnagar, Darahiya, Chhitha, Pandittol, and Pokhariya. English and
  Nepali elicitation prompts are controls, not additional lexical attestations.
- Majhi: report dated July 2014, by Krishna Prasad Chalise; comparative wordlist
  pp. 71–77, 210 prompts at Manthali, Kunauri, Rajagaun, Seleghat, and Sitkha.
  This is a different collection from the existing Page Majhi/Bote material;
  reuse the Kunauri dialect registry entry, retaining collection-specific keys.
- Koraga: D. N. Shankara Bhat, 1971, Deccan College, Poona. Chapter 7 vocabulary,
  pp. 88–118, 1,192 entries. Onti, Tappu, and Mudu reuse the existing registered
  dialects. Grammar examples and the Belari appendix pp. 119–124 are outside this
  vocabulary ingest. Comparator languages remain source comparison prose, not
  newly asserted cognates or donor nodes. The author's caution about contact and
  resemblance is respected: no ancestry is inferred from resemblance.
- Orissa: Office of the Registrar General, India, Language Division, 2002;
  comparative vocabulary pp. 192–233, prompts 1–1013, five targets: Standard
  Oriya, Sambalpuri, Bhatri, Desia, Relli. Other descriptive chapters, sentence
  lists, narratives, and non-target families are excluded. The separate Dravidian
  chapter examples are not part of this parallel vocabulary table.

Pinned URLs, PDF hashes, page counts, acquisition date, and extraction hashes are
in `data/other/forms/raw_data/selected_surveys_2026/snapshot.json`. The original
government Orissa PDF was compared with the compressed PARI mirror; the original
has substantially clearer 300 dpi images and drives extraction. Both scans have
615 PDF pages. Printed-page offsets differ in earlier chapters; vocabulary locators
use the verified pp. 192–233 range, not a book-wide offset assumption.

No explicit redistribution licence was identified. Complete PDFs and render caches
remain ignored local inputs. Only extracted lexical facts, necessary transcription
evidence, bibliographic metadata, and audits are included in the data package.

## Extraction and editorial decisions

`data/other/forms/raw_data/selected_surveys.py` stages the proposal offline and
installs only with `--install`. Native positioned PDF text drives Angika and Majhi.
Font-aware decoding handles superscript aspiration and documented legacy font
glyphs. Physical wraps have an explicit reviewed interpretation; word boundaries
in phrases are preserved. Two Angika WHICH cells and one Majhi cell with a printed
missing-glyph box are excluded without guessing the missing reading.

Koraga has a page-by-page manual collation of all 31 vocabulary pages, retained
alongside original OCR lines and coordinates. All 857 short-i headword glyphs were
visually inspected to distinguish plain i and barred ɨ; this is an explicit glyph
overlay, not a rule replacing word-final i. Four records without a usable lect
siglum remain under canonical Koraga with typed dialect uncertainty, including
the source's undefined `k` label. Source comparisons and “see” references are
retained without inventing graph relations; some references are semantic neighbors,
not equivalent forms. Homographs retain distinct page/entry keys.

Orissa uses detected/deskewed table geometry and two cached Tesseract passes,
`eng` and `script/Latin`. TSV parsing disables CSV quote interpretation. Rules are
removed only when detected at cell boundaries; fixed-width edge erasure was rejected
because it damaged italic initials. The serial-number column and page transitions
validate row alignment independently of English OCR. Category headings, merged
header bands, and footer bands are explicitly excluded and audited. Source-empty
cells can produce hallucinated OCR; image review distinguishes these from real forms.

The English OCR model often loses nasalization. A Latin-pass tilde is retained when
the base characters otherwise agree exactly; other disagreements receive explicit
image-collated overrides or remain typed OCR uncertainty. Grammar labels, human/
nonhuman qualifiers, elder/younger kin terms, and definition qualifiers are separated
from forms. Commas/slashes inside parentheses do not split a lexical reading.
Unreviewed but structurally reliable readings retain both raw passes, exact locators,
and `uncertain` tags under the standing editorial policy.

Each source has an explicit `selected-*` sound profile. Clear retroflex, affricate,
aspiration and length correspondences are converted in `Form`; `Original` retains
the extracted source notation. Koraga ɨ remains distinct. Orissa O is open o (ɔ),
T/D are retroflex, and c/j are palatal affricates, following the source's phonology.
Other ambiguous capital symbols are preserved and flagged. Neither invented native
script nor a redundant invented phonemic transcription is supplied.

Angika (angi1238, Bihari) and Reli (reli1238, Eastern) are new canonical languages.
Reli's identity is corroborated by the source's Relli chapter, pp. 112–135, and
https://glottolog.org/resource/languoid/id/reli1238. Telugu-looking lexical material
does not cause reassignment to Telugu. Source regions are recorded without invented
point coordinates. Existing language/dialect coordinates are preserved.

## Audit and validation

Per-record JSONL audits reconcile all source cells/entries and every emitted row.
The raw records, extraction scripts, manual reading overlays, seeded samples, and
crop-rendering scripts are retained in the source package. All four sources are
unlinked lexical attestations; no new borrowed, reflex, variant or derived edges
are inferred. Source-defined records are protected from destructive deduplication.

Fresh seeded native-text/scan samples: Angika 0/20, Majhi 0/20, Koraga 0/20 material
transcription or structural errors. Orissa's initial OCR sample exposed nasalization,
character and boundary-noise errors; these repairs and subsequent samples are recorded
separately rather than presenting the corrected initial sample as a fresh audit.

| Source | Raw cells/entries | Excluded | Installed rows | Compiled distinct nodes |
|---|---:|---:|---:|---:|
| Angika | 1,050 | 2 | 1,076 | 1,076 |
| Majhi | 1,050 | 1 | 1,055 | 1,055 |
| Koraga | 1,192 | 0 | 1,369 | 1,369 |
| Orissa | 5,065 | 22 | 5,713 | 5,713 |
| Total | 8,357 | 25 | 9,213 | 9,213 |

The 8,332 retained source units expand to 9,213 readings (+881) through explicit
alternate forms and shared lect attestations. All 9,213 are unlinked; zero ancestry,
borrowing or variant edges are asserted. Orissa's 22 exclusions comprise 21 empty
cells and one printed missing-glyph head. Its 49 nonlexical grid bands are separately
audited. No control-language forms, book grammar examples, or Gondi/Kui entries
were emitted. The batch adds two base languages and fourteen dialect registry rows;
Kunauri and the three Koraga dialects reuse existing IDs.

Orissa audit history is **5/20, 2/20, 2/20, 3/20** on successive independently seeded
samples (2026091102–2026091105), with all detected errors corrected. The final sample
had no row, lect, or prompt-alignment error; its remaining errors were an extra OCR
letter, nasalization misrecognition, and case misrecognition. Targeted reviews covered
annotation/symbol cases, 59 residual suspicious cells, 92 nasal-disagreement/empty
candidates, and 61 acute-mark/colon cases. These are overlapping review sets, not a
claim of that many unique reviewed cells. All 1013 extracted English prompts were
read, with damaged/abbreviated prompts collated against the scan. The checked-in
reading overlay records the corrections and special scope decisions.

**Documented legacy-source exception:** the damaged italic Orissa scan did not reach
the ordinary 0/20 transcription target. Under the checklist's explicit allowance for
structurally reliable unreviewed OCR and malformed legacy residual exceptions, it is
installed with exact cell coordinates, both OCR passes, and typed OCR/transcription
uncertainty. This is not a claim that unreviewed spellings are error-free. There is no
known remaining table/parser alignment error; individual glyph/noise errors can remain
in unreviewed cells. No further random seeds were selected merely to obtain a clean
sample. The other three sources have 0/20 fresh transcription/structural samples.

Focused checks: **31 passed before the build**; the deferred compiled-survival test also passes in the final full suite. The
independent post-build validator verifies all 9,213 exact source keys, distinct nodes,
Original/Gloss/Language/Tags/Source preservation, exact profile output, empty Native
and Phonemic, unlinked status, absence of graph edges, and complete formatted reference
metadata. All four profiles have zero unmapped forms. `errors.txt` is empty.

All seven data stages of `make all` passed through reference generation. Its final
manual-etymology check failed the same two baseline assertions (compiled-graph count
and duplicate source-owned overlay rows). Full-suite comparison: **31 failed, 1823 passed, 18 skipped in 578.64s (0:09:38)**. No new failures compared with the saved pre-ingestion baseline. The complete list/comparison is in `20260911-selected-surveys-test-comparison.json`.

Generated changes: forms +9,213 (794,270 total), source keys +9,213, aliases +9,213,
durable identities +9,213, concepts +3, concept memberships +7,919, references +4.
Edges (374,684) and alignments (2,049,443) are byte-identical to baseline. Every old
form ID and all its fields/citations are unchanged; only the 9,213 selected-source
nodes were added. Concept-membership differences are reviewed separately in the
`20260911-selected-surveys-generated-diff.json` audit. There are 7 newly active and
4 inactive concept IDs (net +3), 7,813 memberships on the new forms, and net +106
memberships among old forms. In total 2,029 old forms (780 distinct glosses) changed
automatic memberships. Every changed concept ID is accounted for among the existing
mapper's equal-ranked candidates, including capitalization variants used by the
batch mapper. `pysem` 1.3.0 sorts a set by similarity/POS/frequency without an ID
tiebreaker, then takes one match; separate hash-seed probes reproduce changed winners.
The affected rows and candidate evidence are retained in the generated-diff,
mapper-ties and mapper-seeds JSON audits. No concept-mapping code was modified for
this batch. These automatic memberships should not be interpreted as editorially
reviewed semantic changes. All prior source keys, alias pairs and durable identity
records survive exactly, with only 9,213 additions each.

Validation artifacts are under `source_checklists/audits/20260911-selected-surveys-*`.
The compiled validator is reproducible with
`python data/other/forms/raw_data/selected_surveys_2026/validate_compiled.py`; add
`--baseline /tmp/jambu-selected-before` when the local pre-install copies are available.
The first full-suite run was interrupted by `ENOSPC` during pytest capture near
completion. Its log is retained as `20260911-selected-surveys-full-tests-disk-interrupted.log`.
Only this batch's disposable OCR image/TSV cache, already-audited baseline CSV copies,
and failed Git LFS temporary file were removed to recover disk space. The source PDFs,
complete raw OCR JSON, reading overlays and completed before/after audits remain intact.
The final run uses `--capture=sys --basetemp=/private/tmp/jambu-selected-pytest`.
Source catalogue entries link each installed file, extractor, profile, audit and test;
the global full-pipeline box remains unchecked.

The checkout was already dirty. Baseline compiled forms: 785,057. Existing baseline
suite: 1,808 passed, 31 failed, 18 skipped; `make all` completed its data stages but
failed two pre-existing manual-survey-etymology assertions. Validation must identify
any new failures separately and may not label the full repository gate clean while
those failures remain.

Browser-database refresh, local serving and browser QA are not activated by this
routine ingestion request (checklist section 13). No browser refresh, commit, push,
release, or deployment is part of this batch.

## Representative compiled entries

These persistent IDs refer to the compiled CLDF. The browser database has not been
refreshed, so their app pages are not claimed as browser-QA evidence.

- Angika: `f_xus2igkwkodio` — dehe ‘body’; `selected-angika:p83:i1:c0:v1`, regmi2017angika[p. 83, item 1, Omnagar].
- Majhi: `f_zdv4ona5pevu6` — dziu ‘body’; `selected-majhi:p71:i1:c0:v1`, chalise2014majhi[p. 71, item 1, Manthali].
- Koraga: `f_3iaid3jcbzid2` — akalɨ ‘then’; `selected-koraga:p88:i1:v1`, bhat1971koraga[p. 88, entry 1].
- Orissa: `f_tq7mztqeaujfe` — pɔbɔnɔ ‘air’; `selected-orissa:p192:i1:c0:v1`, census2002orissa[p. 192, item 1, Standard Oriya].
