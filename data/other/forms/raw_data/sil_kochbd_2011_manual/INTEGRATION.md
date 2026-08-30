# Shared integration status — ESR 2011-023 Koch Bangladesh

Shared source-specific steps 1–6 below are complete. The frozen manual ledger
and source-local artifacts remain authoritative. Steps 7–8 and global checklist
regeneration are still deferred.

1. **Done:** verify `post_freeze_manifest.json` and rerun focused tests.
2. **Done:** replace the old Koch source forms with `staged_forms.csv` byte-for-byte. Do
   not stage controls, blanks, not-used rows, ambiguous rows, or visible bases.
3. **Done:** update shared dialect/site records from `site_metadata.tsv`. Exact
   coordinates are absent from the report and must remain blank; remove any
   invented or centroid coordinates. Preserve all report spelling variants in
   notes rather than choosing an unsupported form.
4. **Done:** replace the shared bibliography entry from `reference_metadata.json`,
   retaining the manual-only provenance, exclusions, primary archive URL, page
   extent, and source PDF SHA-256.
5. **Done:** route the exact base-profile snapshot for source key
   `kim-ahmad-kim-sangma2011kochbd`. Preserve manual IPA byte-for-byte in
   Phonemic/raw fields and apply the profile only to display Form.
6. **Done:** record the installed CSV, audit, profile, metadata, and reference
   state in the shared integration manifest and source checklist. Regenerate the
   global checklist/audit only after concurrent Garo work finishes.
7. **Deferred:** run the consolidated CLDF build. Confirm 1,017 unique Koch source keys and
   reconcile opaque generated form IDs before accepting any graph changes.
8. **Deferred:** run graph validation and the full test suite, then rebuild the browser
   database. Inspect representative target rows across `b`, `c`, `q`, and `r`,
   the mixed item-241/site-`r` case, target/control exclusions, source page
   citations, language pages, and absence of source-derived etymology edges.

Do not infer any of the 226 unresolved modifiers during integration. Their exact
coordinates, visible bases, and notes remain in `staging_audit.tsv` and
`post_freeze_manifest.json`.
