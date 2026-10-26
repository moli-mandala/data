# Kholosi source-stage installation plan

The accepted revision3 is installed under the parent/root's explicit source-stage
approval. Revisions1 and2 failed independent review and remain immutable.
`install_reviewed_source.py` provides a read-only default preflight and an
approval-guarded, idempotent install mode. The final mixed-provenance and importer
routing metadata addenda are independently recorded; earlier metadata proposals
are historical evidence. See `source-stage-final-handoff.json` for final hashes.

1. Verify every frozen input, output and generator hash from the accepted
   review manifest, and verify that the independent report pins the same output.
   Reproduce the candidate byte-identically in a temporary source-local tree.
2. Compare canonical CSV/YAML to `legacy-baseline/` before any first install.
   Preserve all 198 keys in their existing order and the exact reviewed ancestry
   assignments, including five independently supported historical corrections.
   Repeated installs must verify exact approved installed bytes.
3. Install the approved complete CSV into `20220913-kholosi.csv`; install the
   approved profile under a source-specific name. Route transcription from
   `form`, with `preserve_hyphens: true`, `split_alternates: false` and
   `dedupe_by_entry_key: true`. Keep original phonetic fields separate.
4. Retain reference key `kholosi`, DOI `10.5281/zenodo.7901554`, language `Khol`
   and its existing coordinates/classification. Add exact complete-source and
   pinned-repository provenance to the existing bibliography entry. No new
   dialect or informant identity may be inferred from A./N./M. labels. OCR is
   false: extraction used selectable PDF text and original CSVs.
5. Preserve the author's printed bibliography and etymological statements in
   the audit. Record the independently verified Anonby publication metadata as
   a correction to the printed reference, not as a retroactively exact page
   citation for every A.-labelled word. Source etymon claims remain distinct
   from historical Jambu assignments and their separately reviewed corrections.
6. Run the actual canonical parser, profile/NFC/NFD, source references, language,
   closed graph-key and focused regression checks. Record final installed hashes,
   counts, uncertainty/exclusion accounting, and all source-key deltas. Update
   only this source's checklist and ledger line.

Full CLDF/database compilation, compiled durable-ID/graph reconciliation, full
suite, browser QA, commits and publication are outside the current source-stage
authorization. Do not describe those gates as passed.
