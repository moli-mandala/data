> Historical plan fulfilled. Exact source-stage installation and validation are recorded in `source-stage-installation-20260926.json`.

# Whole-source installation plan

The 835-row proposal is pending independent acceptance after pass 4 continuity reconciliation. Canonical source files remain unchanged. This plan does not authorize installation before that review passes.

1. Verify the exact proposal CSV, audit, profile, and YAML hashes in `whole-source-freeze-20260926.json` against the independent acceptance report. Preserve the existing 621-row CSV, audit, importer, manifest, and README as historical source-stage artifacts.
2. Archive the existing importer as `historical-lexical-importer.py` before replacing its entry point. `prepare_whole.py` must continue to derive legacy keys from that historical importer, not from the expanded canonical CSV. Install only the reviewed CSV, audit, and profile bytes.
3. Register `preserve_hyphens: true`, retain append order 98 and the existing identity policy, and update the importer command and whole-source scope note. Do not change language identity or infer a new dialect. Preserve all 621 prior keys and leave `data/form-identities.csv` unchanged.
4. Reconcile the Hahn bibliography scope, source manifest, README, checklist entry, and completeness ledger. Describe the 835 emitted rows separately from the 670 editorial audit records, including 24 page context records and seven prior-scope cross-checks. Report the three citation-preserving exact reuses, all recovered expressions and morphology, explicit uncertainty, and removed unsupported variant edges.
5. Update historical count assertions while preserving existing fine-glyph regression cases. Run the focused Hahn tests, actual registered YAML parser, source profile policy and symbol coverage, all citation parsing, in-memory reference formatting, dialect routing, and source-local key/graph checks. Confirm Original, Native, Phonemic, and hyphen retention against the installed raw rows.
6. Confirm canonical CSV/audit/profile hashes equal the reviewed proposal and that durable form identities are unchanged. Save the installation handoff with test evidence and exact counts.

Full database generation, the full suite, compiled global identity/graph/reference checks, database refresh, and browser QA remain deferred under the user's explicit no-build instruction. Report source-stage completion only; do not claim application entries have been refreshed.

The exact read-only postinstallation command is `data/.venv/bin/python data/data/other/forms/raw_data/hahn_asur_1900/verify_whole_installation.py` from the workspace root. It must report 835 parsed rows, 621 retained keys, exact accepted hashes and unchanged durable identities. See `preinstallation-readiness-20260926.json`.
