# Manual audit: items 136-140

The repository ingestion checklist, survey/comparative-table addendum, PDF
inspection policy, and strict rendered-page manual-review policy controlled
this block. Every one of the 80 conceptual cells was independently read by eye
from physical p.56 / printed p.51 at 400 dpi and rechecked in tight
900/1200/1600-dpi crops, with targeted 2400-dpi crops for small glyphs. PDF
text, OCR, and the legacy CSV did not supply, complete, normalize, infer,
correct, or verify any transcription.

## Accounting

- Items: 136 `hot`, 137 `cold`, 138 `right`, 139 `left`, 140 `near`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 79; source blanks: 1; ambiguous: 0; illegible: 0.
- Expanded occurrences: 88 (83 target candidates; 5 controls).
- Item 138 crosses the left/right column boundary on physical p.56.
- Item 139/RNS_Sisana is the sole blank: only one RNS response is printed in
  each of its two response groups, both provisionally assigned to the first
  metadata row under the documented unmatched-occurrence rule.

Every form, repeated response, group label, qualifier, page/column coordinate,
and cell boundary was visually rechecked against the rendered source. Item 136
retains BNM's two responses and records `(weather)` as a qualifier rather than
a lexical form. Items 137-140 retain all printed variants. Tight crops preserve
item-136 RkM/BNM `ʈ`, item 137's initial `ʈʰ` series but ordinary initial `t`
in CCC `tʌɳɖʰa`, item-138 dotted `i`, item-139/CCC `lʌdʌɖi`, item-140
`dʒʰɔn-`, SkP `ʈʰɔɽe`, and RKB `ɖʰiŋgai`.

The independent ledger was frozen before legacy comparison at SHA-256
`ffdef8c63913346f780d20970ad8e7dd513e47220b48db18b533ef27b447e0a4`.
During the post-comparison source-image audit, targeted 2400-dpi crops corrected
the retroflex/alveolar distinctions above, item-137/RKB to `dʒudo`,
item-139/CCC to `lʌdʌɖi`, and item 140's open vowels. These corrections came
solely from the rendered source. The final deterministic ledger SHA-256 is
`7ee0c911a0e6f2ac9b2f66686aacb181f12826bdbbfb6963ffc2e79179443ff8`.

## Unresolved site identities

No lexical reading is ambiguous or illegible. Ten coordinates remain
medium-confidence site assignments because both locality lists print as `RNS`.
The source does not identify which occurrence belongs to Sisaikhara or Sisana;
responses are provisionally mapped in print order to metadata-row order. Item
139 has one unmatched occurrence per response group, both provisionally mapped
to Sisaikhara, leaving Sisana blank. Every physical/printed page, item, site
key, column, visible response description, and candidate locality is enumerated
in `../unresolved_readings.tsv`.

## Post-entry reconciliation

Only after the independent ledger was frozen did the guarded importer compare
reviewed target occurrences with `20230530-tharu2.csv`. The block has 83 manual
and 84 legacy target occurrences: 52 agree exactly, leaving thirty-one
manual-only and thirty-two legacy-only multiset occurrences. Source-retained
differences principally preserve ordinary `g`, dotted `i`, printed vowel
quality, and the printed alveolar/retroflex contrasts. The excess legacy row is
item-136/BNM `(weather)`, misparsed as a lexical form in the legacy CSV. Every
difference was rechecked visually after the comparison. The legacy data were
never accepted as transcription evidence.

Cumulatively through item 140, 1,967/2,272 legacy target occurrences agree
exactly; the multiset retains 322 manual-only and 305 legacy-only occurrences.
Staging remains refused at 2,240/3,360 reviewed cells. Item 141 `far`, physical
p.56 / printed p.51, is next.
