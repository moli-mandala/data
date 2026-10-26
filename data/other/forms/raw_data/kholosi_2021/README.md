# Kholosi whole-source recovery — source stage installed

The audited 1,135-row source stage replaces the historical 198-row selection.
All 198 historical entry keys remain, and 937 new keys account for the rest of
the source. Five erroneous historical ancestry assignments were independently
reviewed and corrected; the earlier assignments remain in immutable evidence.
Full compilation, compiled identity/graph checks and browser QA are deferred.

## Pinned representations

- Arora and Etebari, *Kholosi Dictionary*, v1, 13 April 2021,
  DOI `10.5281/zenodo.7901554`, CC BY 4.0. The 12-page PDF acquired from the
  author's Git repository exactly matches the DOI's published MD5
  `adcd775e9fede81c3cc2c02122e1c1f4`. Printed dictionary pp.1–10 follow the
  cover and preface p.0. No OCR was used.
- Author repository `aryamanarora/kholosi`, immutable commit
  `18b43f982efad92f35e4054561f2e0fc519f4eb7` (13 April 2021).
  `upstream/acquisition.json` records URLs, sizes, SHA-256 and verified Git blob
  hashes for each scoped acquisition. No audio or full repository was cloned.
  The repository root contains no license file; the DOI's license is not
  automatically attributed to separate fieldwork files.

`whole-source-scope-20260927.json` records the full census and remaining gates.
The published PDF is richer than `lexicon.csv`, while the latter and
`sentences.csv` contain additional paradigms and fieldwork examples. They must
not be represented as identical versions or all cited to invented PDF pages.

## Physical inventory

- PDF: 448 main-head occurrences plus 70 indented subentries. These include
  duplicates of explicit alternate cross-references, not 518 distinct lemmata.
- Structured lexicon: 401 populated records, including 19 explicitly printed
  inflected/comparative forms in 17 paradigm fields and one grammatical-only
  feminine label. Trailing spreadsheet statistics are nonlexical controls.
- Sentence CSV: 116 records, 113 populated responses and three entirely blank
  rows. Four responses have no free translation. Source disagreements between
  interlinear analysis and translation remain evidence, not silent corrections.
- `sentences.txt`: 17 untranslated narrative lines, retained as source text.
  No word glosses are inferred.
- Acoustic `data*.txt` files: formant tracks and segmented environment labels;
  numeric samples are not lexical entries and underscored labels are not
  reconstructed into unprinted lemmata.

## Reproducible review scaffolds

`census_pdf.py` extracts font/geometry evidence and source-order physical units;
`prepare_first_reading.py` retains raw lines and creates candidate lexical fields.
`prepare_structured_reading.py` records every CSV field and explicit paradigm.
All of these are review drafts, not approved importers or canonical installation
entrypoints. They preserve head/subentry hierarchy across pages, homonym numbers,
source authority labels, full etymological prose and original pronunciation.

`legacy-baseline/` preserves the exact historical CSV and YAML. All 198 keys
survive. The reviewed concordance does not merge homonyms by spelling/IPA alone.
Several historical forms reflect an earlier witness
not present literally in the published PDF; they require honest snapshot
provenance rather than a retroactively invented page citation.

## Historical first revision

`whole-proposal-r1/review-manifest.json` preserves the first **1,135-row,
1,110-unit** proposal, which failed graph review. The source owner visually
reviewed all ten dictionary pages and reconciled the complete structured source.
The accounting includes 518 PDF units, 517 physical CSV records (three blank),
29 quoted/embedded occurrences, 15 morphology occurrences (one exact reuse),
six morphology/comparison controls, one complete untranslated text, three
acoustic-file controls, and 21 separately preserved historical witnesses.

The first revision preserved all 198 legacy keys and etymon assignments. Of these, 177 have
reviewed equivalent-witness mappings and 21 remain separate because meaning,
vowels, compound analysis or competing historical ancestry prevents a unique
replacement. The keyed identity review records exact before/after fields.
`source-etymology-claims.jsonl` separately preserves 830 attributed statements;
all 353 numeric CDIAL citations resolve. Three printed-vs-editorial conflicts
remain explicit; no winner or new graph assignment is inferred.

At that historical stage, six focused regressions passed. Actual parser code accepted all 1,135
rows in NFC and NFD, preserving original spelling, phonetic transcription,
notes, citations and ordered keys. The source-local profile has zero unknown
graphemes or policy violations, and variant/derivation keys are closed.
`whole-regeneration-validation.json` records byte-identical regeneration.

The unglossed `sentences.txt` text is one **source-text witness**, not an asserted
single lemma or a guessed set of sentences. Its 17 exact lines remain in the
audit, its translation is blank, and the output explicitly marks the missing
translation. The real parser accepts the full text. Static frontend inspection
shows `FormWord.svelte` renders the full string and the entry headword wraps;
this is not a substitute for the deferred browser check.

## Accepted installation

Revision1 failed for missing source graph links. Revision2 added explicit,
qualified ancestry and component links, but failed for omitted borrowing tags.
Revision3 repaired all 830 attributed claims' donor classification, including
variant/paradigm inheritance and representation-specific component scope.
The fresh independent review passed 20/20 rows plus 12 targeted groups; the
independent complete donor census also passed. Both failed freezes and reports
remain unchanged. See `independent-r3-original-review.json` and
`independent-full-donor-census-review-r3.json`.

The installed output has 317 borrowing-tagged rows, 107 uncertainty-tagged rows,
25 blank-gloss witnesses, 380 ancestry-linked rows, 35 variant links and 90
derivation-parent links. Blank glosses retain unglossed examples/text, unknown
meanings and explicitly identified morphology without invented meanings.
Printed-number errors, competing analyses and unkeyed donor claims remain
separately attributed in the audit. Five historical corrections affect only
the ancestry assignments of moni, xāzen, saodo, tʰal and ke; ke is now unlinked
because its printed kṛte target is unresolved. Other competing historical
analyses remain distinguished from source claims.

`install_reviewed_source.py` is the sole canonical coordinator; it verifies
accepted independent reports, all frozen hashes and exact regeneration before
writing. `make ingest SOURCE=20220913-kholosi` invokes this guarded route.
The final YAML declares mixed etymology provenance, no OCR, literal Form input,
preserved hyphens, disabled automatic alternate splitting and key-based identity.
The source reference, language Khol and its registered coordinates remain stable.

Twenty focused tests passed. The actual canonical parser accepted all 1,135
rows with exact keys, literal/phonetic fields, notes, citations and graph fields;
NFC/NFD checks also passed. Metadata registry validation and repeat installation
through the registered importer passed. All 198 existing identity records are
preserved in `canonical-validation.json`; compiled identity reconciliation for
the expanded source still awaits the deferred full build.

Source-stage installation is recorded in `source-stage-final-handoff.json`.
Full CLDF/database compilation, compiled graph/identity checks, full suite and
browser QA remain explicitly deferred. No release or publication was performed.
