# KEED 2018 and Muduga 2022 ingestion review

Checklist: SOURCE_INGESTION_CHECKLIST.md, dictionary/glossary, comparative-table and etymological addenda. Browser refresh is not requested in this turn; browser database construction and browser QA are inapplicable. No commit, push or deployment is included.

## Sources and scope

**Učida and Rajapurohit, edited by Takashima (2018), Kannada–English Etymological Dictionary**, second edition / first electronic edition. Canonical PDF: https://publication.aa-ken.jp/KEED_2018.pdf (971 PDF pages; dictionary pp. 1–942). CC BY-NC 4.0. The 2018 edition supplies native spelling, romanization, IPA, expanded definitions and references. Glyph-aware extraction was necessary because the legacy Kannada font has an incomplete Unicode map. The printed alphabet provides the decoding equations; rare consonant clusters and Gandhari private-use glyphs were checked against rendered pages. This is text/font extraction, not OCR. PDF SHA-256 and the reproducible extraction command are in raw_data/keed_2018/acquisition.json and keed_2018.py.

28,797 main anchors reconcile to 28,746 lexical main entries, 50 alphabet headings and one wrapped headword prefix continued in the next anchor. Main entries and subentries yield 31,250 lexical units. Each anchor is accounted for in coverage.jsonl.gz; each lexical unit, its raw evidence, emitted rows and decisions are in audit.jsonl.gz. Native spelling alternates, four abbreviated headword expansions, extra IPA realizations and complete native-script paradigms are separate stable-keyed rows. Examples, illustration captions, alphabet headings and nonlexical frontmatter are excluded. Headless dash constructions, zero forms and conditional/abbreviated paradigms are retained as evidence without inventing a complete attestation. A blank gloss on a pointer-only entry is intentional.

**Arsenault and Abraham (2022), Centralized vowels in Muduga**, JSALL 9(1–2):97–129, DOI 10.1515/jsall-2022-2045; online March 2023. Indexed author-preprint text accessed 2026-09-12 supplies 91 lexical examples from Tables 3–19 and glossed prose, plus 18 phonetic variants: 109 rows. The 1,100-word underlying field corpus is not publicly supplied by this article and is not claimed as ingested. Control-language comparisons, acoustic measurements and Tables 1–2, 20 and A1–A6 are excluded. All 1,065 indexed lines are accounted for in coverage.jsonl; table/section locators use the author preprint rather than guessed journal pages. Direct PDF download was unavailable; this source was audited against indexed preprint text, not PDF images. No redistribution licence was identified; the installed data are extracted lexical facts. The article itself is not installed as a PDF.

## Languages and transcription

Reuse canonical Kannada and Muduga. Source-cited immediate donors reuse Sanskrit, Hindi, Marathi, English, Arabic and Persian. No base languages added. Kannada named varieties map to registered dialects; seven new regional/site labels are Mysore, Central Karnataka, Northern Karnataka, Southern Karnataka, Southern Maratha country, Kumta and Bellary. Existing Havyaka, Halakki, Gowda, Nanjangud, Tiptur, Gulbarga, Barkur and Coorg mappings are reused. Muduga Chindakki is a registered locality dialect. New coordinates are blank: the sources do not supply defensible site points.

KEED Form is converted from source romanization, Original retains that romanization, Native retains Kannada, and Phonemic retains the separate printed IPA. Native-only paradigms/alternates retain script in Original and use an explicit orthographic profile; no IPA is invented. Source long and nasalized vowels, r̥/r̥̄ and r̤ are preserved or explicitly mapped. Meaningful affix hyphens remain. Colloquial stars are not reconstructions; source question marks and nonstandard spelling marks receive typed review reasons. Literary/register symbols and grammatical labels become tags; figurative and pejorative are synchronized with the frontend registry. Verb case-government labels remain in the audit rather than incorrectly tagging the verb as a declined case form.

Muduga preserves dental versus alveolar stops/nasals, tap versus trill, central vowels ɯ/ɤ/æ, and rounded y→ü. Phonetic variants retain their parent phonemic analysis and variant relationship. Every installed input is checked against its explicit profile.

## References and graphs

Direct DEDR D/A and CDIAL T/C references are validated against installed etyma. A denotes the DEDR appendix; M references denote Mayrhofer. Source bibliography abbreviations resolve through the checked reference map to existing or newly catalogued auxiliary works. References retain physical printed page/column or preprint table/section locators. Subentries crossing a column/page cite their physical location while retaining the immutable parent-anchor key.

Only direct unambiguous Kannada DEDR claims become ancestry. Explicit immediate donor forms become borrowed-from rows; foreign donor chains, reverse loans, ambiguous compounds and uncertain comparisons remain unlinked with evidence. Printed compound components resolve only to unique source headwords. Native alternates point to their printed parent, not a guessed synonym. The presence of a pointer hand alone does not create a variant edge.

Muduga has 79 directly DEDR-linked rows, two inflected forms reaching DEDR transitively through their base, and 18 phonetic variants. kɤːkkæ ‘he hears’ derives from kɤːɭɯ ‘listen’. mɯɡa ‘baby boy’ has conflicting possible DEDR references and remains unlinked. The printed dove reference DEDR 2885 is unavailable in the installed etymon register and remains unlinked. The harmonium word retains the reported borrowing claim without inventing an immediate donor chain.

## Audit and validation

Validation counts, final seeded samples, compiled graph checks and test outcomes are appended below. Earlier random-audit rounds found and corrected legacy glyph clusters, split heads, dropped secondary IPA, bibliography/dialect labels, nested etymological brackets, sense grammar, tag-only paradigms, and column/page-break hyphenation. These classes have regression assertions. Complete per-record evidence remains available; a sampled audit is not a claim that every lexical analysis has received individual human review.


## Final installed counts and unresolved cases

| Source | Lexical units | Installed rows | Direct etymon links | Borrowed-from rows | Variants | Rows with derivation parents |
|---|---:|---:|---:|---:|---:|---:|
| KEED | 31,250 | 43,120 | 11,156 | 1,596 | 8,458 | 5,632 |
| Muduga | 91 | 109 | 79 | 0 | 18 | 2 |

These columns describe relationships, not disjoint buckets. KEED includes 41,524 Kannada rows and 1,596 separately attested donor rows: 760 Sanskrit, 231 Persian, 181 Arabic, 254 English, 64 Marathi and 106 Hindi. There are 11,055 main/subentry units with direct DEDR ancestry, 1,596 with a resolved immediate donor and 18,599 with neither assignment. Some of the last group have compound/derivation parents.

5,385 KEED lexical units have at least one typed issue. Issue instances include 3,606 etymological, 2,045 source headword question marks, 267 nonstandard/uncommon spellings, 36 gloss question marks, and 17 units with unresolved auxiliary citations. This includes the dictionary’s own doubts and conservative non-linking decisions, not just extraction problems. Nineteen auxiliary citation occurrences remain unidentified: RV 5.65, Khmd.13.63, PBh 8.78, LSB 1.3 (twice), KRa.19.37, Pn.3.75, Katre 1968.95, PPr.3.139, BIB 47.5, KV 1.8.86, MMV 119.467, TR.5.1.25, HN 1.1.36, KN 82, KN 50, a corrupt SII-like locator, Pt.78 and SII.XX.178-152,1192. Their raw printed strings remain in the audit; no guessed bibliography records or ancestry were installed.

The final KEED seed is 2026091213; 20 entries were compared with rendered source pages, with no remaining material lexical extraction errors. A source gloss question mark was preserved and additionally classified as uncertainty. The Muduga text audit uses seed 20260912 and has 0/20 material errors against the indexed preprint. The checked sample manifests and source-count JSON are under `source_checklists/audits/20260912-*`. `raw_data/keed_2018/sample_audit.py` reproduces the sample and optionally renders full column images from the pinned PDF.

### Checklist gates

- Source/edition/licence/scope: recorded above and in BibTeX/acquisition metadata.
- Reproducible extraction and complete record coverage: passed; pinned font-aware cache for KEED and indexed text/line hashes for Muduga. OCR addendum is inapplicable because no OCR contributed.
- Languages/dialects: canonical registry and blank-coordinate policy verified by focused tests.
- Schema/transcription/tags: full input coverage and original/native/IPA separation checked.
- References/graphs: all installed citation keys, exact row citations and emitted relationship endpoints checked against compiled CLDF.
- ID retention: stable source keys and durable identity registry retained; new imports use namespaced temporary IDs after established files to protect historical numeric aliases.
- Browser database, serving and visual app QA: not requested, therefore inapplicable. Compiled representative IDs below are for inspection after the next requested browser refresh.
- Full build and full-suite results: see the validation results appended below; pre-existing failing checks are reported explicitly, not counted as passes.

## Compiled validation results

All 43,120 KEED and 109 Muduga rows survive as compiled nodes. Across all of them there are **zero missing keys, changed Original/Gloss/Native/Phonemic fields, lost row citations, unregistered citation keys, or incorrect variant/borrowing/derivation endpoints**. `errors.txt` is empty. All 443,574 prior source keys remain present. The final namespacing rebuild changed **zero durable IDs** compared with the complete source-key/ID snapshot from the preceding build.

| Generated file | Before ingestion | Final | Change |
|---|---:|---:|---:|
| `cldf/forms.csv` | 794,270 | 837,499 | +43,229 |
| `cldf/edges.csv` | 374,684 | 403,090 | +28,406 |
| `cldf/form-source-keys.csv` | 443,574 | 486,803 | +43,229 |
| `cldf/form-id-aliases.csv` | 1,146,794 | 1,226,146 | +79,352 |
| `data/form-identities.csv` | 964,047 | 1,007,276 | +43,229 |
| `cldf/concepts.csv` | 3,272 | 3,295 | +23 |
| `cldf/form_concepts.csv` | 581,681 | 597,022 | +15,341 |
| `cldf/alignments.csv` | 2,049,443 | 2,143,644 | +94,201 |
| `cldf/references.csv` | 563 | 631 | +68 |

The form, source-key and identity increases exactly match the 43,229 new source rows. Alias growth also retains aliases from intermediate builds before switching the new files to namespaced temporary IDs; it is not durable-ID churn. The final reference increase is 68 correctly registered records. Earlier intermediate placeholder-like references caused by semicolons within locators were eliminated by the final citation normalization/build.

`PYTHONPATH=. uv run python -m pytest -q tests/test_keed_2018.py tests/test_muduga_2022.py tests/test_sound_profiles.py tests/test_dialects.py`: **27 passed**. `make all` runs every generation stage successfully but exits nonzero on the two pre-existing `test_manual_survey_etymologies.py` assertions (Rajasthani compiled count and duplicate source-owned overlay rows). Both failures are present in the saved 20260911 selected-surveys baseline.

The initial default full-suite invocation encountered a pre-existing duplicate `test_preintegration_contract.py` module name under Bhumij and Noira; the complete suite was rerun with `--import-mode=importlib` to collect both. Final full-suite comparison follows below.

Representative compiled nodes (available in the app after a user-requested database refresh):

- KEED Sanskrit borrowing `keed2018:p13:c2:e14`: `f_qayihtrqsvslo`, with its separately keyed cited Sanskrit donor and CDIAL 8478.
- KEED DEDR-linked causative `keed2018:p718:c1:e5`: `f_6qw2czl65o6ew`, linked to d4723.
- KEED compound-stem entry `keed2018:p243:c1:e9`: `f_vefg4dngodan4`.
- Muduga ‘listen’, `muduga2022:table6:b:1`: `f_x2xfwbksibr72`, linked to d2017. The two ‘he hears’ attestations `f_zsqx3azarss4o` and `f_lkblwfkpio3tu` derive from this base.

## Full-suite outcome and handoff

The full run with `--import-mode=importlib` finished in 572.46 seconds: **32 failed, 1,832 passed, 18 skipped**. Comparison with the saved baseline found all **31 baseline failures unchanged**, no baseline failures resolved, and one new assertion failure in `test_kewa_contributes_prose_only_and_all_blocks_survive_the_full_build`. That test assumed no lexical form could ever cite Mayrhofer. KEED now legitimately cites Mayrhofer as an auxiliary source. The test was narrowed to permit these citations only on compiled KEED-owned records with KEED as their primary citation, retaining its original prose-only import and no-KEWA-edge assertions. The entire KEWA test file then passed: **7 passed in 4.92 seconds**. No source data or build code changed after the completed full-suite run. The remaining failures are the same 31 baseline failures. The full suite was not repeated after this test-only correction; the complete run and targeted rerun are both preserved.

The overall full-build/full-suite gates are therefore **not globally green**: the two survey make-target assertions and the 31 existing suite failures remain outstanding. The two new sources pass all 27 focused checks, complete compiled row/field/citation/relationship validation, and the corrected KEWA compatibility tests. No new ingestion regression remains identified.

Files changed for this work: both dated installed CSVs; the two reproducible importers and their pinned evidence/coverage/audits; `conversion/keed.txt` and `conversion/muduga.txt`; source-profile and append-order routing in `utils.py`/`make_cldf.py`; `cldf/dialects.csv` and `cldf/sources.bib`; canonical figurative/pejorative tags in `tags.py` and the frontend tag registry; source regression tests plus the KEWA secondary-citation test; README and ingestion catalogue/review artifacts; and regenerated CLDF outputs/durable identity state. The checkout contained many prior uncommitted ingestions; those unrelated edits were preserved.

The lexical additions are installed and compiled. Browser refresh, serving, commit and deployment were not part of this turn.
