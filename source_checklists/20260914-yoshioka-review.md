# Yoshioka recovery review — 14 September 2026

The source CSV and importer are corrected. **Full repository integration remains
pending:** only the bounded source parse, transcription and durable-ID checks have
run. The global CLDF and browser database still contain the earlier import.

## Result

| Measure | Result |
| --- | ---: |
| Old installed rows | 3,183 |
| Corrected installed rows | 4,886 |
| Accepted source entries/senses | 3,233 |
| Additional full inflection/alternate rows | 1,653 |
| Historical entry anchors accounted for | 3,212 / 3,212 |
| New physical entry/sense keys | 55 |
| Old fragments/continuations rejoined | 25 |
| Bare headings retained audit-only | 9 |
| Fully resolved cross-reference entries | 214 |
| Partially resolved / unresolved index entries | 2 / 4 |
| Ambiguous forms retained without accepted edges | 6 |
| Source-questioned donor statements flagged | 10 |
| Rows retaining printed grammar in notes | 4,005 (3,949 original + 56 referenced) |
| Rows with plural tags | 481 |
| Rows with noun-class tags | 2,367 |

All 3,123 currently active source-key identities shared with the corrected input
retain their public IDs in the bounded assignment check. Eleven currently active
IDs belonging to rejoined fragments gain aliases to the surviving articles. The
alias table accounts for all 25 rejoined keys, including ones without active IDs.
The durable registry itself has not been regenerated or rewritten by this work.

## Source and extraction

- Canonical source: Noboru Yoshioka, *A Reference Grammar of Eastern Burushaski*,
  TUFS dissertation, 2012. [Institutional record](https://tufs.repo.nii.ac.jp/records/1002),
  [DOI](https://doi.org/10.15026/72148).
- Scope: vocabulary, PDF pp. 505–618 / printed CLXXIX–CCXCII; all 114 pages.
  The 626-page source PDF is pinned by SHA-256 in the extraction manifest.
- Ordinary extraction had missing Gentium Unicode mappings. The embedded font's
  cmap restores those mappings directly. The authoritative snapshot is native
  positioned text; current output does not depend on OCR substitutions.
- Snapshot, parser, audit, hashes and repeatable image-review commands are in
  `data/other/forms/raw_data/yoshioka_2026/` and `yoshioka_cleanup.py`.
- The nine excluded bare headings are bám, γuqú, γaáro, hōš, khín, Y ék, ét,
  šaŋál and zaŋs. Their printed text and physical positions remain in the audit.
  They have no independent definition in this source layout; none is guessed.
- Other thesis sections, grammar examples, text appendix and bibliography are
  outside the installed vocabulary. Introductory notation and bibliography pages
  were used to interpret the vocabulary and its references.
- The thesis is openly accessible; no explicit reuse licence was located. The
  source PDF is not redistributed; the package records extracted lexical facts.

## Grammatical and transcription decisions

Grammatical, plural and noun-class labels are retained. Canonical tags describe
the relevant forms; the complete printed grammatical string remains in
`Source grammar:` notes on the article's emitted rows, including suffix-only
paradigms and argument specifications that cannot be represented safely as
headword tags. This deliberate redundancy preserves the requested source labels.

Examples:

- **aalú / aaloínc**: X-class headword “potato”; its full plural is a separate
  linked row tagged `pl`. The headword is not tagged plural merely because a
  plural appears in its article.
- **alét**: singular/plural pronoun forms retain their H/X/Y class distinctions;
  the dictionary's class labels do not turn the pronoun into a noun.
- **yuúṭis / yuúṭiŋ**: explicitly X-class singular versus Y-class plural.
- **balógan / balógayo**: the first is explicitly number-invariant (`sg pl`);
  the second retains `double-plural`. `DOUBLE PL` is not interpreted as dual.
- **čhu**: the X-class bunch/head sense and Y-class polostick/spear-head sense
  remain separate, including their different plural formations and shared citation.
- **Y.PL.OBJ**, etc.: retained as argument specifications, not applied as the
  number/class of the verb. Dialect stems following a completed verbal paradigm
  do not automatically inherit that paradigm's aspect label.

The sound profile now keeps **c / č / c̣** distinct. Accent, underdot,
nasalization and personal-prefix-slot marks remain in source spelling. Profile
coverage passes for every installed form, with no replacement characters.
Unprinted stem-plus-suffix forms and phonemic interpretations of slot marks are
not invented. Printed English typos remain source readings.

Eight registered, language-qualified Yoshioka dialect tags retain the source's
Eastern Burushaski, Hunza, Nager, Hopar, Ganish, Altit, Hillside and Riverfront
labels. Coordinates reuse existing approximate registry points, explicitly marked
as such. No new fieldwork coordinates or top-level languages are asserted.

## References and graph

Every row cites its printed Yoshioka page and headword. `B.` references are
structured citations to Berger 1998, Teil III; `AA.#` references are item numbers
in the 1967 ILCAA *Linguistic Questionnaire for Asia and Africa, 2*. These are
marked **cited by Yoshioka**, not independently transcribed citations. The ILCAA
bibliographic entry is now registered and formatted; Yoshioka's DOI, coverage,
provenance, editor and current non-OCR extraction are recorded.

The original 163 exact, unique `see` resolutions are unchanged. A follow-up review
of all 57 unresolved index entries checks matching printed stems/subentries in
the referenced root groups. It resolves 56 of 62 emitted forms; six remain
ambiguous, with literal notes, candidate meanings, blank lexical gloss, an
`uncertain` tag and no accepted edge. Multi-form index lines are treated per form,
so one form cannot inherit an unrelated meaning from its neighbour. See the
[complete cross-reference review](20260914-yoshioka-crossreferences.md).

The resolved forms retain both citations and the referenced article's grammatical
string, POS, number, noun-class and applicable dialect tags. The sole accepted
spelling equivalence is index `duqhúlan` versus stem `duqhúlan-`; neither source
spelling is rewritten. Fully resolved index entries now total 214, with two
partially resolved and four unresolved entries.
Source `¶` comparisons, synonyms and donor remarks remain distinguishable from
asserted ancestry. No new borrowing, cognate or derivational edges are inferred.
All existing Yoshioka keys used as Burushaski cognate evidence survive.

## Review and validation

- Fresh sample, seed **20260918**: **0/20 material errors** against rendered source
  images. Earlier samples found 3/20, 3/20, 1/20 and 2/20 issues; all identified
  issue classes were corrected. The sample keys, issues and image hashes are
  recorded in `review-samples.json`.
- Targeted image checks: **93 entries**, covering merged fragments and
  continuations, new indented entries/senses, all nine excluded headings,
  rare symbols, and the first/last vocabulary entries. Five additional grammar
  examples verify `DOUBLE PL`, invariant number and Times-font dialect forms.
- **60 focused tests pass**, including ten cross-reference regression cases.
  The earlier broader run exposed one unrelated existing
  assignment-count failure: `test_the_dictionary_self_reference_assignments_all_point_at_live_sub_entries`
  expects 2,604 self-references, while the shared assignment data contains 2,603.
  That test was explicitly deselected in the final focused pass; its expectation
  and data were not changed to make this work pass.
- Actual `make_cldf.parse_file` on this source: **4,886 rows, zero parse errors**;
  all emitted keys and source spellings survive. The bounded `assign_ids` check
  validates existing IDs and the reviewed fragment aliases.
- Every source dialect and bibliography key is registered. `make_refs.py`
  regenerates 637 formatted references. Source-local variant endpoints and cycles,
  cognate evidence keys, complete transcription coverage and grammar retention
  are covered by the focused tests.
- Cross-reference image review covers every index line in the 57-entry inventory
  and the relevant root/stem contexts, rendered in 28 groups. Only the 62 reviewed
  rows change; all source forms, keys, original grammar and earlier resolutions
  are preserved. Decisions and candidate evidence are pinned to source/row hashes.

## Checklist gates

Applicable addenda: dictionary/glossary and etymological/comparative references.
The historical OCR addendum was considered; native-font recovery supersedes it.
OCR confidence and engine-setting gates are inapplicable to the current output.
Survey matrices, web pagination and upstream CLDF conversion are inapplicable.

| Checklist section | Status/evidence |
| --- | --- |
| 1–3: scope, extraction, identifiers | Source, fixed page inventory, snapshot, per-record audit and ID checks above |
| 4–5: language/dialect and rich schema | Bur + eight registered dialect tags; 15-column rows, NFC, nonempty forms |
| 6–7: linguistic structure and sound profile | Source grammar retained; scoped tags and full profile coverage tested |
| 8–10: references, graph, audit | Formatted bibliography, conservative links, complete key accounting and row hashes |
| 11: focused tests | 60 passing; unrelated corpus-count failure separately recorded |
| 12: full pipeline/full suite | **Deferred.** Full build and suite have not run for these inputs |
| 13: browser database/UI QA | **Deferred.** Browser refresh was not requested |
| 14: documentation/publication | Review and regeneration instructions saved; publication not requested |

The 8 GB local-resource policy reserves heavy full builds/suites for an authorized
remote runner. Existing validation CI runs on pushed branches/PRs; it cannot test
these uncommitted inputs without publishing them. No full local build was started.
The consolidated generated source-audit refresh is deferred with the full build.
These deferred gates mean this is not a claim of completed end-to-end ingestion.

Representative existing public IDs, for the later refreshed app check:
`f_5vporf6ptilc2` (aalú), `f_jt6eok72wx3sy` (alét), and
`f_7c5dkg4jdjskg` (yuúṭis). Their corrected source rows are verified; the current
app database has not been refreshed or visually validated for these changes.
