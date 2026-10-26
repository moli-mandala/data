# Complete review scaffold — not an accepted export

`prepare_review_candidate.py` pins the immutable first reading and preserves all97 source units in `review-audit.jsonl`. Its105 candidate rows retain105 physical witnesses:85 Aranadan and20 explicitly labelled Malayalam/Kannada comparisons. The comparison rows exist for review; final lexical scope is pending root approval. Twelve documentary/context positions remain audit-only. Two slash-separated alternatives are already individual witnesses; the exporter must not split them again.

Each candidate keeps the original `:wN` witness key, whole expressions, literal form, source lexical gloss and source-only Notes. All graph columns, Native and Phonemic remain blank. Citation source is `nair2013aranadan`; the Usha work is background authority, not a directly collated attestation. Candidate grammar tags and typed uncertainty come from the first reading and are not represented as independently accepted.

Future root overlays belong outside this directory in `root-review/`. Interface:

```json
{"approved":true,"first_reading_sha256":"22fe3f77dd063609585a70cbe032392e143c955dc237df471a4c16dedf316136","evidence_sha256":{"review/report.json":"..."},"decisions":[{"witness_key":"...:w1","field":"form","old":"...","new":"...","decision":"accepted"}]}
```

Allowed fields are form, gloss, language, tags, note and typed_uncertainty. Every old value and pinned evidence file is checked. A held decision leaves the first reading unchanged and remains explicit in status.

The same approved overlay may include `unit_exclusions` (source_unit_key, exact expected_witness_keys, reason), `witness_exclusions` (witness_key, expected_form, reason), `added_witnesses` (source_unit_key, shared_physical_witness_key, expected_physical_form, complete witness object, reason, evidence), and `variant_links` (child_key, parent_key, expected_child_form, expected_parent_form, reason, evidence). A shared-language attribution must retain the exact printed anchor form and citation locator and does not increase the immutable105 physical-witness count. It has its own stable key under the existing unit. Excluded witnesses remain wholly present in the audit. Variant endpoints must be distinct, same-language, emitted and closed; no borrowing or derivation edges are created. Root controls the settings/profile/provenance proposal; no source YAML or canonical file is written here.

Run via local tmux with the data virtualenv. The deterministic scaffold writes only integration outputs. It has no install option. Complete original review, root approvals, metadata/profile/actual-parser validation and fresh output acceptance remain mandatory before a separate guarded installation route.
