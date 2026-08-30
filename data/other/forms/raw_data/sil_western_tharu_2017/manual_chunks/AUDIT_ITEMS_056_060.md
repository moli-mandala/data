# Manual audit: items 56-60

The repository ingestion checklist, survey/comparative-table addendum, and
strict rendered-page manual-review policy controlled this block. Every one of
the 80 conceptual cells was independently read by eye from physical pp.41-42 /
printed pp.36-37 at 400 dpi and rechecked in 900-dpi crops. PDF text, OCR, and
the legacy CSV did not supply, complete, normalize, infer, correct, or verify
any transcription.

## Accounting

- Items: 56 `smoke`, 57 `ash`, 58 `mud`, 59 `dust`, 60 `gold`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 78; source blanks: 2; ambiguous: 0; illegible: 0.
- Expanded occurrences: 78 (73 target candidates; 5 controls).
- Item 58 crosses physical pp.41-42 / printed pp.36-37. Its two RNS rows
  occur on opposite sides of the break.

The two blanks are item 58/CCC across the complete item block on physical
pp.41-42 / printed pp.36-37, right/left columns, and item 59/CCC on physical
p.42 / printed p.37, left column. Both site codes are absent from their
complete printed item blocks. No form is inferred from another list or copied
from prior data.

Every form, response-group label, blank, and cell coordinate was visually
rechecked against the rendered source. A tight post-entry recheck confirms
item 59/RNS occurrence 1 as `dʰudʰʌ̃ɾ` and item 58/TkN as the
source-visible dotted-`i` form `miʈːi`. All strings are NFC. The deterministic
ledger SHA-256 is
`57cf4981df378a0993a5d94e03e3c9cef906ec9ba24cfa0ba4ac7f4b310b44c9`.

## Unresolved site identities

No lexical reading is ambiguous or illegible. Ten coordinates remain
medium-confidence site assignments because both locality lists print as `RNS`.
Item 58's first occurrence is on physical p.41 / printed p.36 and the second is
on physical p.42 / printed p.37. Every physical/printed page, item, site key,
column, visible response description, and candidate locality is enumerated in
`../unresolved_readings.tsv`.

## Post-entry reconciliation

Only after the independent ledger was frozen did the guarded importer compare
reviewed target occurrences with `20230530-tharu2.csv`. The block has 73 manual
and 73 legacy target occurrences: 72 agree exactly. The sole paired difference
is source `miʈːi` versus legacy `mɪʈːi` at item 58/TkN; the rendered source
clearly prints dotted `i`. The legacy data were never accepted as transcription
evidence.

Cumulatively through item 60, 900/963 legacy target occurrences agree exactly;
the multiset retains 83 manual-only and 63 legacy-only occurrences. Staging
remains refused at 960/3,360 reviewed cells. Item 61 `tree`, physical p.42 /
printed p.37, left column, is next.
