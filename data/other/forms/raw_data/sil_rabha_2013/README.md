# SIL ESR 2013-016 Rabha source-local package

This is a complete source-local manual transcription audit for the Rongdani and Maituri wordlists in
Alexander Kondakov's 2013 SIL survey. It is not wired into Jambu's shared registries, sound profiles,
references, or build.

The source contains 194 published prompt rows across two lists (388 cells), even though the report
describes the original elicitation instrument as 210 items. Source-order IDs `S001`--`S194` avoid
inventing mappings for the sixteen omitted prompts.

Coverage is complete at `S001`--`S194`: all 388 cells, comprising 387 attested cells and one explicit source
blank (`S148_MTR` prints `no data`), with no ambiguous, illegible, or unresolved cell. The seven-column
manual chunks are under `manual_chunks/`; every coordinate includes
physical and printed page, source-order item, list, and table side.

All published prompt rows and the report's sixteen explicitly omitted problematic elicitation items
are accounted without inventing standard-list mappings. The source-local audit and focused validation
are complete. This duplicate audit does not stage new rows because the repository already contains the
legacy complete Unicode installation; any decisions to reconcile its diplomatic differences, plus the
consolidated build, graph checks, and browser QA, remain deferred to the parent integration phase.

After independent entry, a repository inventory found that this report already has a legacy complete
Unicode extraction and shared integration. `RECONCILIATION.md` records the comparison; none of those
pre-existing files were used to transcribe any chunk or changed by this audit. Accordingly, shared
integration gates apply to any future changes prompted by this manual audit, not to the completeness
of the source-local visual review or the status of the legacy install.
