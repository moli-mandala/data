# Manual audit: items 6-10

The repository ingestion checklist, survey/comparative-table addendum, and
strict rendered-page manual-review policy controlled this block. Every one of
the 80 conceptual cells was independently read by eye from physical pp.32-33 /
printed pp.27-28 at 400 dpi, with 1200-dpi crops used for cell-level rechecking.
PDF text, OCR, and the legacy CSV did not supply or verify any transcription.

## Accounting

- Items: 6 `ear`, 7 `nose`, 8 `mouth`, 9 `teeth`, 10 `tongue`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 80; source blanks: 0; ambiguous: 0; illegible: 0.
- Expanded occurrences: 80 (75 target candidates; 5 controls).
- Source group labels were retained as evidence only.
- Item 8 has twelve literal `(4)` cross-reference qualifiers. They remain in
  `Source_Qualifier` and were not expanded as forms or treated as analysis.

All transcriptions are NFC. The deterministic ledger SHA-256 is
`267b7579d876db7811dc3ef0ac0f1102097a2693319bfb8b629c41ff33e19a39`.

## Unresolved site identities

No lexical reading is ambiguous or illegible. Ten coordinates remain
medium-confidence site assignments: for each item, the source prints two `RNS`
responses without distinguishing the Sisaikhara and Sisana metadata rows. The
first and second occurrences are provisionally mapped to metadata row order.
Every coordinate is enumerated in `../unresolved_readings.tsv`.

## Post-entry reconciliation

Only after this hand-keyed block was written did the guarded importer compare
reviewed target occurrences with `20230530-tharu2.csv`. Cumulatively through
item 10, all 154 manual target occurrences match 154 legacy occurrences
exactly, with zero manual-only and zero legacy-only occurrences. The scan
remains authoritative.

Staging remains refused at 160/3,360 reviewed cells. Item 11 `breast`, physical
p.33 / printed p.28, is next.
