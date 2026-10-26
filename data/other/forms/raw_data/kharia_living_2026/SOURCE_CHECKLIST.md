# Kharia Living Dictionary whole-source stage — 2026-09-27

Installed **524 rows** from all 452 entries/senses and all three example sentences. The 521 legacy keys and every legacy Form, Native and Phonemic value are preserved. Source-stage work is complete; full pipeline and browser gates remain deferred, so this is not certification of a completed database ingestion.

The dictionary/glossary and website/API checklist addenda apply. User reuse authorization is standing: “living dictionary is allowed per sarva.” The public SQLite downloaded2026-09-27 has upstream date2026-09-18 and SHA `bc4ab203e8f6edc902cda41171c9feed602fe854ea34aee3869b610199167f93`. It differs physically from the historical2026-08-20 snapshot. All452 IDs, native lexemes, phonetics, sense IDs, bilingual glosses, semantic domains and dialect assignments match the historical raw audit exactly. Both version identities remain documented. Historical first-stage manifests/audits are unchanged.

The complete455-unit audit includes452 lexical records and3sentences. Full schema accounting found no further populated grammatical/etymological/morphological/text collections. Nonlexical exclusions are438audio records,69photographs,2speakers and administrative records; no media fetched or personal speaker metadata exported. Raw source lexical fields are preserved in `whole-source-recovery/lexical-snapshot-20260927.json`.

Three whole sentences retain exact Native/Form, free English translations and blank Phonemic. Upstream automatic tokens split meaningful apostrophes and provide ambiguous candidates, without interlinear glosses; they remain raw evidence rather than invented aligned word rows. Sentence locality is unspecified and does not inherit a linked headword's dialect. Existing Dhelki/Dudh registrations are reused; no new language, dialect or coordinates were invented.

There are152 reviewed metadata changes: explicit numerals/ordinals, person/count, full-clause meanings, an explicit noun label and expression boundaries. Qualifiers are scoped, not blindly extracted from English: bare you has no number; her has no case/possessive inference; my kinship terms do not receive noun-number tags from their possessor; English plural nouns do not imply target plural morphology. Existing source sense/homograph distinctions and68variant links remain intact. The culture head's leading hyphen is retained with typed uncertainty. Total rows carrying uncertainty: 138.

The sound profile adds12literal Devanagari characters only. Six old native-only heads plus3sentences have no invented phonetic transcription. Native, Original, source IPA, Unicode NFC/NFD, tags and every key pass actual canonical parser checks. No donor or etymology edges were inferred. All citation IDs resolve to the updated bibliography; all dialect tags remain registered and local variant targets close.

Independent audit passed20/20units,25rows, zero material errors against the verified SQLite; all3sentences also received targeted review. Frozen manifest approval gates the idempotent installer. The historical importer now delegates installation to this coordinator, preventing headword-only regeneration from dropping recovered sentences or metadata. Repeated regeneration preserved exact CSV/YAML/profile bytes.

Focused canonical validation: `12 passed, 1 deselected in 0.97s`. The single compiled-data test was deliberately not run because no compiled rebuild is authorized; it remains a strict524-row future gate. Evidence and canonical hashes: `whole-source-recovery/installed-validation-20260927.json`. Full CLDF/full suite/compiled graph and identity/reference refresh/database/browser QA remain explicitly deferred. No commit, push or deployment occurred.

Representative source keys are the three `living-kharia2026:sentence:<UUID>` entries listed in the validation report. New app entries are not yet built or browser-verified.

## Gate disposition

Passed at source stage: exactsource/version/permission, complete extraction/audit, immutable keys, language/dialect, rich fields/grammar, profile, scoped references, conservative local graph, focused regressions and guarded installation. Deferred compiled/full-suite/browser gates are listed above. OCR addendum is inapplicable: native digital SQLite; no phonological analysis inferred from audio.
