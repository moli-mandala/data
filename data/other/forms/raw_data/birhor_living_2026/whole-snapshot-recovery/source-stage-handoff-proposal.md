# Birhor Living Dictionary — whole pinned snapshot proposal

This is a proposal, not an installed source or a completed database ingestion. The snapshot was acquired on 2026-09-22 and last modified upstream on 2026-08-08. It is not the separate 2021 printed dictionary. User authorization on 2026-09-27 explicitly states “living dictionary is allowed per sarva”; the authorization file records the reuse decision.

The proposal has 3,480 rows from all 2,746 structured head records and explicitly glossed annotation attestations. Every one of the 3,100 earlier preparation keys survives. There are 4,090 candidate analyses before 610 exact subsidiary-attestation reuses; all reused attestations retain their source-unit keys and citations in the 3,795-unit audit. No form-only or cross-analysis merging is used.

All 98 note fields (79,453 characters), 967 parenthetical spans and 82 outside-parenthesis decisions are accounted for. All 10 morphology fields and the single interlinear field remain available. The 2,522 populated semantic-domain lists and all auxiliary gloss-language fields remain in raw audit; unsupported semantic or morphological analyses are not inferred. Pure media credits and explanatory English/foreign controls remain documentary evidence.

Three previously held heads have explicit recovery decisions: the blank-English entry uses its supplied botanical identification; the composite bulbil/tuber head produces two distinct senses without forced Native/Phonemic pairing; bare babat buna is separately attested, while alkusi has a qualified probable Hindi-comparison interpretation. One malformed numbered gloss is split at its explicit second-sense marker, preserving the old first key. Source-explicit second-person plural, nonfinite verb, borrowing, and interlinear subject information is accounted for.

Annotation English renderings are contextual source renderings, not independently inferred dictionary definitions. Note-only forms have no invented Phonemic or Native layer. General source descriptions of medicinal use remain attributed reports, not independently verified therapeutic claims. Explicit alternatives alone receive variant links; borrowed origin without an exact donor form does not create a graph edge. The source's botanical and ethnobotanical references remain scoped secondary citations; subsidiary note forms do not automatically inherit the botanical head's references.

The profile preserves source contrasts, glottal stops and nasal marks; w normalizes to house v while Original/Phonemic retain source spelling. The note notation uses the same IPA-like j/affricate conventions as the head transcription. NFC and NFD input produce identical normalized output. Seven focused tests passed, including the actual proposed YAML/parser route, all keys and transcription layers, source-role citation resolution, and exact-reuse accounting.

Pending: independent whole-output approval, canonical source-stage installation and registration, full CLDF/database build, compiled reference and graph checks, and browser QA. No database build is authorized here. The frozen manifest, owner review files and independent targeted reviews distinguish their respective scopes; a targeted decision review is not whole-output approval.

## Proposed installation paths

- `data/other/forms/20260927-birhor-living.csv` from `proposal.csv`.
- `data/other/forms/20260927-birhor-living.yaml` from the identically named source-local proposal.
- `conversion/birhor-living.txt` from `proposal-profile.txt`.
- Three scoped BibTeX entries from `proposal-sources.bib`: primary `living-birhor2026`; secondary `tropicos` and `lakra-etal2022birhor-ethnobotany`. No existing canonical entries with these IDs were found in the current bibliography.
- Keep complete source-local raw snapshot, historical preparation, review inputs, full audit, profile decisions and frozen-manifest evidence. No media assets are required.
