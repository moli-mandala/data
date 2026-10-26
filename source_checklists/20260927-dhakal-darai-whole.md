# Dhakal 2011 Darai whole-thesis source-stage review

The complete 486-page thesis is now installed at the source stage. This is not
full compiled or browser completion. The dictionary/glossary and comparative-table
addenda apply to the grammar's lexical examples, paradigms and witness comparisons;
OCR is inapplicable because native font decoding was used and visually reviewed.

## Scope, counts and evidence

- Canonical source: Dubi Nanda Dhakal, *The Darai Language*, Tribhuvan University,
  2011; university PDF SHA-256
  `6ca12ee82a22a16b0394b08e319afd5a94f075111fb8a994ee10d59da234bebf`.
- All chapters and appendices are accounted for, including full responses,
  grammatical morphemes, verb paradigms, front-matter lexical mentions and five
  interlinear narrative texts. No open redistribution licence is asserted; the
  original thesis is not committed.
- 11,226 indexed units expand to 11,304 audited units: **8,373 emitted entries,
  2,431 citation-preserving reuses and 500 explicit controls**. Repeated occurrence
  evidence remains in the audit, not silently omitted.
- Author-transcription CSV: 8,361 rows. Literal-preservation companion: 12 rows.
  Every earlier 485 key survives as a distinct row; legacy Original, Phonemic and
  graph fields are unchanged.
- Controls identify abstract symbols, explicit other-language comparisons,
  source-unacceptable/absent cells, questionnaire/respondent/reference context and
  misclassified English gloss fragments. Identifiable source morphemes are included.

The authoritative package is `data/other/forms/raw_data/dhakal_darai_2011/`.
`whole-installed-audit.jsonl` retains all raw evidence pointers, occurrence rows,
interpretation decisions, exact reuse targets and citations. Native text evidence,
owner reviews and the earlier failed independent audits remain available.

## Interpretation, metadata and uncertainty

The profile separates Original, author Phonemic and display Form, preserves all
485 old Phonemic values, and has complete symbol and NFC/NFD coverage. Four source
notation groups remain literal because the source does not determine slash scope.
Eleven literal entries preserve unresolved marks/notation; one other entry preserves
Roman spelling without asserting phonemic interpretation. There are 117 entries
with typed source uncertainty. Two ambiguous multiword interlinear groups retain
literal analyses without invented word-level alignment.

All 539 unique source notes were reconciled. Explicit grammar comes from source
labels, headings and reviewed claims. Repairs cover complete derived words versus
markers, verb-root roles, 37 printed root-boundary hyphens, stale alignment flags,
100 residual uppercase/person-number gloss cases, and 23 conditional particles that
must not acquire lexical verb POS. Subject/object agreement roles remain distinct.

Canonical Darai remains `dara1250`, Bihari. Existing Chitwan/Pidrahani records are
unchanged. Tanahun and Nawalparasi district attestations have registered dialects
with blank coordinates; speakers, historical witnesses and recording locations
retain their separate provenance. Only the Dhakal bibliography block was replaced;
578 unrelated bibliography entries were verified unchanged and the formatted
reference passed its focused test.

The local graph contains 12 explicit variant links and 18 derivation links;
all local parents resolve. Two older numerical etymology fields are unchanged.
Co-listing and synonyms do not create inferred variant links. Reuse preserves every
citation and compares complete analyses, Notes, route, attribution and uncertainty.

## Validation

- Independent fresh sample **20260927206: 0/20 material errors** against originals;
  an additional variant-target check passed. The earlier failed samples and fixes
  remain immutable evidence.
- **66 focused tests passed in 5.37 seconds**. One compiled-source test was
  deliberately deselected because the full build is not authorized.
- Actual canonical generic parser preserves all 8,373 keys, Original, Phonemic,
  Notes, tags and citations. Profile-policy/NFC/NFD, metadata, formatted bibliography,
  source graph, exact reuse and legacy-key tests passed.
- Source settings validator passed: 283 files, 270 declared citation keys. Both
  routes declare reproducible importer commands; the citation key is declared once.
- Immutable pre-expansion CSV/profile snapshots prevent reading installed output
  as a legacy input. Postinstallation regeneration reproduced both CSVs, audit and
  profile byte-for-byte. The dependency-only code patch is separately documented.
- Durable `data/form-identities.csv` SHA-256 remained
  `4977c4e205f66233366ee1c00a9c938c9c78e3c087b63e02a94832647525c28a`.

Canonical author CSV SHA-256:
`4c0151607f67b0c181b445899cd1dc7a5d701ea15a8763143f37c4d47787d58a`.
Literal CSV:
`4645f38115177714489d922e4c82544a91598348964029bd059c7061619fe6da`.
Audit:
`cf77517556fed68f5961e03a29b4310ce744d0fa22af065172da59b48017bb4d`.
Profile:
`0cd4b3396c56fc8183f763b278b606d99ac2e37a858613f119ff494fd3d04b31`.

## Deferred gates

Full CLDF/database build, compiled source-key/identity/graph/reference survival,
full test suite and browser/app QA remain explicitly deferred by the user. No DB,
commit, push or release was performed. Representative future app checks should use
`dhakal2011:p368:appendix3b:l70:root` (printed stem boundary),
`dhakal2011:p214:ch7:surface:l53:word:l55:w2` (reviewed interlinear grammar),
and the p141 literal-mark entry (blank Phonemic and typed uncertainty). These are
source keys for a later authorized build, not claims of currently deployed entries.
