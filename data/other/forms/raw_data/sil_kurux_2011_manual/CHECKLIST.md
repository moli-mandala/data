# Checklist - ESR 2011-040 Kurux Bangladesh manual recovery

Active addendum: survey wordlists/comparative tables. The legacy-font text layer is locator-only;
the installed lexical evidence is a complete manual rendered-page transcription, not OCR output.

- [x] Recover the exact publisher PDF and verify its canonical SHA-256.
- [x] Pin the Wayback capture, physical/printed topology, bytes, and page count.
- [x] Render all physical wordlist pages 39-57 at 300 dpi.
- [x] Visually confirm the first and last wordlist pages and item boundaries 1-307.
- [x] Hand-transcribe all 38 printed response lines for items 1-10 from page 39.
- [x] Hand-transcribe all 34 printed response lines for items 11-20 from pages 39-40.
- [x] Hand-transcribe all 33 printed response lines for items 21-30 from page 40.
- [x] Hand-transcribe all 31 printed response/disposition lines for items 31-40 from pages 40-41.
- [x] Hand-transcribe all 37 printed response lines for items 41-50 from pages 41-42.
- [x] Hand-transcribe all 33 printed response lines for items 51-60 from page 42.
- [x] Hand-transcribe all 35 printed response lines for items 61-70 from pages 42-43.
- [x] Hand-transcribe all 32 printed response/disposition lines for items 71-80 from page 43.
- [x] Hand-transcribe all 29 printed response lines for items 81-90 from pages 43-44.
- [x] Hand-transcribe all 34 printed response lines for items 91-100 from pages 44-45.
- [x] Hand-transcribe all 30 printed response/disposition lines for items 101-110 from page 45.
- [x] Hand-transcribe all 42 printed response lines for items 111-120 from pages 45-46.
- [x] Hand-transcribe all 33 printed response/disposition lines for items 121-130 from page 46.
- [x] Hand-transcribe all 36 printed response lines for items 131-140 from page 47.
- [x] Hand-transcribe all 33 printed response lines for items 141-150 from pages 47-48.
- [x] Hand-transcribe all 25 printed response/disposition lines for items 151-160 from page 48.
- [x] Hand-transcribe all 33 printed response/disposition lines for items 161-170 from pages 48-49.
- [x] Hand-transcribe all 28 printed response/disposition lines for items 171-180 from page 49.
- [x] Hand-transcribe all 28 printed response lines for items 181-190 from pages 49-50.
- [x] Hand-transcribe all 30 printed response/disposition lines for items 191-200 from page 50.
- [x] Hand-transcribe all 31 printed response lines for items 201-210 from page 51.
- [x] Hand-transcribe all 33 printed response lines for items 211-220 from pages 51-52.
- [x] Hand-transcribe all 26 printed response lines for items 221-230 from page 52.
- [x] Hand-transcribe all 19 printed response/disposition lines for items 231-240 from pages 52-53.
- [x] Hand-transcribe all 32 printed response/disposition lines for items 241-250 from page 53.
- [x] Hand-transcribe all 40 printed response lines for items 251-260 from pages 53-54.
- [x] Hand-transcribe all 40 printed response lines for items 261-270 from pages 54-55.
- [x] Hand-transcribe all 36 printed response/disposition lines for items 271-280 from page 55.
- [x] Hand-transcribe all 30 printed response/disposition lines for items 281-290 from pages 55-56.
- [x] Hand-transcribe all 30 printed response/disposition lines for items 291-300 from page 56.
- [x] Hand-transcribe all 17 printed response/disposition lines for items 301-307 from pages 56-57.
- [x] Use targeted 1200-dpi visual crops for small IPA marks without OCR.
- [x] Freeze all thirty-one line ledgers before any reconciliation.
- [x] Expand bracket codes mechanically to 1,842 conceptual cells / 1,869 retained rows.
- [x] Preserve all printed variants, including items 114/site A, 118/site D, 119/site B, 120/site 0, 131/site A, 147/site E, 149/site C, 150/site E, 165/site C, 202/sites D and E, 245/sites A and D, 274/site D, and 283-284/site D, separately.
- [x] Record exact blanks and confirm no ambiguity or illegibility in items 1-307.
- [x] Keep code 0 Bangla/control rows audit-only through the manual freeze and confirm its identity from report Table 2 before staging.
- [x] Verify source hashes, ledger hashes, Unicode hygiene, counts, and focused tests.
- [x] Preserve globally unused items 31, 74, 107, 124, 152, 163, 171, 194, 240, 247, 301, and 306 as six explicit `not_used` cells each.
- [x] Record line-level visual confidence for items 41-307.
- [x] Hand-transcribe items 301-307 from rendered pages only.
- [x] Reconcile every frozen manual row against the legacy audit without using legacy forms to supply or verify a reading.
- [x] Resolve target/control site identities from report Table 2 and prepare source-local language/dialect rows; leave exact coordinates blank because the report does not print them.
- [x] Complete source-local metadata, sound profile, reference, exclusion, staging, and audit gates.
- [x] Install the frozen 1,365-row target file with immutable source Entry_Keys.
- [x] Install exact shared reference metadata and the five target-site plus audit-only control identities.
- [x] Replace invented survey-site coordinates with explicit blank coordinates and source locality notes.
- [x] Install and route the complete source-local sound profile; preserve manual source IPA in `Phonemic`.
- [x] Update the source-specific checklist/audit hooks and focused shared-integration tests.
- [ ] Run the consolidated CLDF build, opaque `f_*` reconciliation, graph validation, and full test suite.
- [ ] Rebuild and inspect the browser database if requested.

Shared source-specific integration is complete. The consolidated build, opaque identity
reconciliation, browser refresh/QA, commit, and shipping remain explicitly deferred.
