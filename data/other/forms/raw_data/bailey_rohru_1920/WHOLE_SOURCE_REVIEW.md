# Rohru whole-source proposal — 27 September 2026

The source stage is frozen for independent review, **not installed**. The canonical
309-row glossary remains unchanged. The applicable glossary, survey and OCR
checklists are active; database builds and browser QA are explicitly deferred.

## Scope and accounting

Original: `tmp/pdfs/bailey1920/bailey1920.pdf`, SHA256
`7a1577a2d2a24eca8b95e9b1ed700ef01f716d1069e4468f36a673611d093b39`.
Printed page +26 is the one-based PDF page. Shared introduction 113–118 and
Rohru chapter 118–130 were checked for scope; all target grammar, translated
sentences and lexical commentary were reread from page images and narrow crops.

* 255 glossary cells: 250 earlier accepted cells retain all 309 installed rows;
  three cross-reference-only cells remain accounted without invented responses;
  storm and village now have three **explicitly uncertain** complete candidates.
* 373 grammar groups on 115–116 and 118–124 emit 381 forms, including identifiable
  bound morphology and only source-supported expansion of abbreviated cells.
* All 22 complete translated responses on 125–126 emit 22 forms. Sentence 22
  includes its printed predicate; it is not discarded as a fragment.
* Eight lexical commentary groups on 126 emit eight forms, including the complete
  independent ability example and its explicitly identified infinitive ending.
* Eleven context/control records preserve shared geography, orthography, tone
  claims and separately attributed comparisons. General Koci/Simla examples are
  not silently assigned specifically to Rohru. Entire-page OCR evidence remains
  in `whole-source-text-inventory.jsonl`, with context pointers in the audit.
* Two repeated numeral/ordinal examples in the page 121 aspiration discussion
  map to exact existing target rows with additional source citations and Notes.

Total: **723 forms, 671 audit units, 17 visibly uncertain rows**. No generic
typography hold remains. Uncertainty concerns the literal identity of printed
marks, not an inferred phonological property. The storm mark especially is a
small upper block/loop, not a securely identified nasal tilde. Its provisional
encoding and exact crop locator are retained in Notes and the review record.

## Reproducibility and transcription

`prepare_whole_source.py` checks immutable first-reading hashes, applies the
keyed `whole-second-reading-20260927.json` corrections, and writes only the local
proposal and audit. Original first readings and the exact 309-row installed
snapshot remain intact. Frozen artifact hashes are recorded in
`whole-proposal-freeze-20260927.json`.

The proposed profile preserves source breves, quantities, underdots, stacked
marks, hyphens and underlining. Sentence-initial capitals lower in display only;
`w` maps to house `v`. No source h/tone interpretation is silently converted to
IPA. U+0331 in provisional underlined digraphs is consistently encoded as U+0332
in emitted source forms; the first-reading string is retained in audit. Compound
and phrase spacing remains unchanged. Installation requires YAML
`preserve_hyphens: true`; Original must remain literal through the real parser.

Source grammatical qualifications survive in Tags or Notes, including object
agreement, the shared conditional/participle heading, imperfect/pluperfect,
instrumental meaning, stative versus active participles, source contractions and
the anomalous printed aspiration comparison. Shared-source genetic and phonetic
claims remain attributed commentary, with no inferred borrowing or etymology
edges.

## Validation and remaining gates

`tests/test_bailey_rohru_whole_recovery.py`: **5 passed in 0.25 s**. Tests cover
deterministic regeneration, every legacy row/key, corrected literal classes,
visible uncertainty, registered tags, all-form proposed-profile coverage,
closed variant references and exact repeated-attestation accounting.

Independent actual-output review, canonical installation, metadata/reference
scope reconciliation and the real canonical parser are pending. The existing
glossary tests must be updated after approved installation without altering
their historical fixtures. Full CLDF/database and browser gates are deferred
under the user's current prohibition; source-stage success must not be reported
as completed publication or application validation.
