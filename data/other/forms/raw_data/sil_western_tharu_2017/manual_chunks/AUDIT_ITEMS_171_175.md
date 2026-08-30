# Manual audit: items 171-175

The repository ingestion checklist, survey/comparative-table addendum, PDF
inspection policy, and strict rendered-page manual-review policy controlled
this block. Every one of the 80 conceptual cells was independently read by eye
from physical pp.61-62 / printed pp.56-57 at 400 dpi and rechecked in tight
900/2400-dpi rendered-page crops. PDF text, OCR, and the legacy CSV did not
supply, complete, normalize, infer, correct, or verify any transcription.

## Accounting

- Items: 171 `this`, 172 `that`, 173 `these`, 174 `those`, 175 `same`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 79; source blanks: 1; ambiguous: 0; illegible: 0.
- Expanded occurrences: 88 (80 target candidates; 8 controls).
- Item 172 crosses physical pp.61-62 / printed pp.56-57.
- Item-172/RNS metadata row 2 has no independently assignable response: the
  only printed RNS responses occur singly in groups 3 and 6, so both remain
  provisionally under metadata row 1 with locality identity unresolved.

Every form, group label, qualifier, page/column coordinate, and cell boundary
was visually rechecked against the rendered source. Tight crops preserve
item 171's dotted `i`; item-172/DDK `ʊ` and TkN `(171)` qualifier; item-173
nasal `jẽ`, BNM `ɪtna`, repeated-form qualifiers and expanded alternatives;
item-174/DDK `ʊ` against five `u` forms; and all 19 item-175 occurrences,
including source Latin `g` in DGC `ekːægʰʌs`, plain `t` in KkP `eketaɾ`,
CCC `ɾitto` with `(alike)` retained only as a qualifier, and the Hindi
group labels `1 / 3 / b`.

The independent ledger was frozen before legacy comparison at SHA-256
`00d4fa9122b05cfdfc2bbb91a6bc71717309ee7a920f3f519a76295ee8dcc1b2`.
The post-comparison source-image audit required no correction, so the final
deterministic ledger has the same SHA-256.

## Unresolved site identities

No lexical reading is ambiguous or illegible. Ten coordinates remain
medium-confidence site assignments because both locality lists print as `RNS`.
The source does not identify which occurrence belongs to Sisaikhara or Sisana;
responses are provisionally mapped in within-group print order to metadata-row
order, with the unmatched item-172 responses retained under the first metadata
row. Every coordinate and visible response description is enumerated in
`../unresolved_readings.tsv`.

## Post-entry reconciliation

Only after the independent ledger was frozen did the guarded importer compare
reviewed target occurrences with `20230530-tharu2.csv`. Seventy-seven of 80
manual target occurrences agree exactly with 77 of 81 legacy occurrences. The
three manual-only and four legacy-only differences preserve source Latin `g`,
plain `t`, the source-visible absence of an initial vowel in CCC `ɾitto`, and
`(alike)` as a qualifier rather than a lexical form. The legacy data were never
accepted as transcription evidence.

Cumulatively through item 175, 2,435/2,783 legacy target occurrences agree
exactly; the multiset retains 363 manual-only and 348 legacy-only occurrences.
Staging remains refused at 2,800/3,360 reviewed cells. Item 176 `different`,
physical p.62 / printed p.57, is next.
