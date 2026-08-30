# Manual audit: items 61-65

The repository ingestion checklist, survey/comparative-table addendum, and
strict rendered-page manual-review policy controlled this block. Every one of
the 80 conceptual cells was independently read by eye from physical pp.42-43 /
printed pp.37-38 at 400 dpi and rechecked in 900-dpi crops. PDF text, OCR, and
the legacy CSV did not supply, complete, normalize, infer, correct, or verify
any transcription.

## Accounting

- Items: 61 `tree`, 62 `leaf`, 63 `root`, 64 `thorn`, 65 `flower`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 80; source blanks: 0; ambiguous: 0; illegible: 0.
- Expanded occurrences: 87 (81 target candidates; 6 controls).
- Item 61 crosses the left/right column boundary; its RNK cell itself spans the
  boundary. Item 64 crosses physical pp.42-43 / printed pp.37-38.

Every form, repeated response, group label, and cell coordinate was visually
rechecked against the rendered source. Tight crops preserve item 61/SkP
`ɾukʰːa` and CCC `ɡatʃʰ`, item 62's contrast between retroflex
`pʌʈːa` in the first RNS row and plain `pʌtːa` in the second, and item
64/KkP's second response `ɡaŋʈʰi`. All strings are NFC. The deterministic
ledger SHA-256 is
`a33a10b5907c5793e4951a1253723a190a05a6611073aca8126d0d13471e5631`.

## Unresolved site identities

No lexical reading is ambiguous or illegible. Ten coordinates remain
medium-confidence site assignments because both locality lists print as `RNS`.
The item-61 pair falls on opposite columns, and item 62 visibly distinguishes
retroflex `ʈ` in the first RNS response from plain `t` in the second. Every
physical/printed page, item, site key, column, visible response description,
and candidate locality is enumerated in `../unresolved_readings.tsv`.

## Post-entry reconciliation

Only after the independent ledger was frozen did the guarded importer compare
reviewed target occurrences with `20230530-tharu2.csv`. The block has 81 manual
and 81 legacy target occurrences: 63 agree exactly, leaving 18 paired multiset
differences. Rendered-source rechecks retain dotted `i` in three leaf forms;
`a`-nasal versus legacy schwa-nasal and one `ŋ`/`n` difference in root; the
absence of legacy's inserted `ɳ` in six thorn forms; and source tap `ɾ` versus
legacy retroflex flap `ɽ` in four tree forms. The legacy data were never
accepted as transcription evidence.

Cumulatively through item 65, 963/1,044 legacy target occurrences agree
exactly; the multiset retains 101 manual-only and 81 legacy-only occurrences.
Staging remains refused at 1,040/3,360 reviewed cells. Item 66 `fruit`, physical
p.43 / printed p.38, left column, is next.
