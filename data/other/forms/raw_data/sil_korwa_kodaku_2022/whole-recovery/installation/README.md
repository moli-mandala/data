# Guarded whole-report installation draft

`install_source.py` is a Behera-specific validator and adapter around the
existing Aranadan installer. It reuses that engine's staged journal, atomic
replacement, interrupted-write recovery, and complete-receipt no-op behavior.
The engine's exact source hash must itself be approved. No generic pipeline
code is changed.

No acceptance manifest is supplied here. Root must create
`root-acceptance.json` only after final source review. Default invocation is
read-only and fails closed when approval is absent. The real installer has
not been executed. Tests use isolated, synthetic DATA roots only.

## Required root manifest

- `installation_approved: true`, `result: "pass"`,
  `review_scope: "whole-report"`, and `engine_sha256`.
- `files`: whole-recovery-relative approved input paths and exact SHA256s,
  including `installation/install_source.py` and every final asset.
- `assets`: exactly eight roles: `legacy_csv`, `addition_csv`, `legacy_yaml`,
  `addition_yaml`, `languages`, `dialects`, `bibliography`, `audit`. Each has
  `input` and `output_sha256`; these must agree with `files`.
- `canonical_baselines`: the eight exact DATA-relative target paths from the
  installer's `TARGETS` map, with before hashes or null for absent files.
- `durable_identity_sha256` for `data/form-identities.csv`.
- `legacy_rows`, `addition_rows`, total `rows`, and `audit_units`.
- `legacy_changed_keys` and `legacy_added_keys`, exactly matching the reviewed
  legacy repair. Every previous legacy key must survive.
- `registry_added_ids` with `languages` and `dialects` lists. Every existing
  registry record must remain identical.
- `bibliography_changed_keys` (only the real Behera source is allowed) and
  `bibliography_added_keys`. All unrelated entries must remain identical.
- `audit_schema`: `key_field` (`audit_key` or `source_unit_key`) and
  `entry_keys_field` (`entry_keys` or `proposed_entry_keys`). The accepted
  derivative must have top-level `accepted_for_installation: true` on each
  audit record and exact installed key closure; raw historical flags stay
  nested in preserved source evidence.
- Optional `original_inputs` pins read-only source files by workspace-relative
  path. These originals are never staged or modified.

Inputs for shared registries and bibliography must be fully merged, reviewed
files. The installer does not generate their scholarly content. Canonical
drift after root approval stops the operation rather than re-merging edits.

The source YAML must preserve legacy transcription in file defaults, use
literal conversion-disabled defaults for additions, and declare the real
source's `identity.dedupe_by_entry_key: true`, `split_alternates: false`, and
`notes.audit_only: true`. Both YAMLs must declare this guarded installer.

## Recovery and rollback

The reused engine writes `whole-recovery/installation-transaction.json` before
canonical writes and `whole-recovery/installation-receipt.json` only after all
outputs match. Interrupted writes can resume only from exact approved staged
bytes, with targets still equal to either their baseline or reviewed output.
The adapter additionally checks journal/receipt output hashes against root's
approved assets, so changing both a stage file and its journal cannot bypass
approval. A successful repeated invocation is a byte-preserving no-op.

Before an installation starts, baseline bytes are retained under
`installation/baselines/` with `baseline-backups.json`. `--rollback` alone
validates a rollback plan without writes; `--rollback --install` applies it.
Rollback refuses any intervening target edit, restores exact baseline bytes,
and removes only newly introduced targets still matching approved outputs.
It retains the journal, receipt, and a rollback receipt for review. Subsequent
reinstallation requires root adjudication rather than silently reusing state.

`test_installer.py` covers dry-run immutability, missing approval, input,
canonical, identity and engine drift, partial recovery, forged stage/journal
output rejection, repeated no-op, unrelated metadata protection, rollback,
path/role guards, actual synthetic install/recovery/rollback cycles, and the
source-key identity requirement. No real canonical target is touched.

Full CLDF/DB builds, compiled identity effects, browser QA, and publication
remain deferred. This draft and isolated tests are not installation approval.
