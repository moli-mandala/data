# Manual audit: items 51-55

The repository ingestion checklist, survey/comparative-table addendum, and
strict rendered-page manual-review policy controlled this block. Every one of
the 80 conceptual cells was independently read by eye from physical pp.40-41 /
printed pp.35-36 at 400 dpi and rechecked in 900/1200-dpi crops. PDF text, OCR,
and the legacy CSV did not supply, complete, normalize, infer, correct, or
verify any transcription.

## Accounting

- Items: 51 `wind`, 52 `stone`, 53 `path`, 54 `sand`, 55 `fire`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 78; source blanks: 2; ambiguous: 0; illegible: 0.
- Expanded occurrences: 83 (76 target candidates; 7 controls).
- Item 52 crosses physical pp.40-41 / printed pp.35-36; its DDK cell itself
  spans the page break and retains both `pʌtʰʌɾa` and `dũŋɡa`.
- Item 55 crosses the left/right column boundary.

The two blanks are item 51/BNT on physical p.40 / printed p.35, right column,
and item 53/BNT on physical p.41 / printed p.36, left column. Both site codes
are absent from their complete printed item blocks. No form is inferred from
another list or copied from prior data.

Every form, response-group label, blank, and cell coordinate was visually
rechecked against the rendered source. In particular, a post-entry 1200-dpi
check confirms the item-52 source's plain `t` (not legacy `ʈ`) in the affected
stone forms. All strings are NFC. The deterministic ledger SHA-256 is
`4204211a14935162e8610c4d3a4c0ac03290d61023f644b9a38580e5f2da4d1e`.

## Unresolved site identities

No lexical reading is ambiguous or illegible. Ten coordinates remain
medium-confidence site assignments because both locality lists print as `RNS`.
Item 53 prints three RNS responses: the lone group-1 `ɾasta` and first group-3
`ɾʌtːa` are provisionally assigned to metadata row 1, while the second
group-3 `ɾaha` is assigned to metadata row 2. Every physical/printed page,
item, site key, column, visible response description, and candidate locality is
enumerated in `../unresolved_readings.tsv`.

## Post-entry reconciliation

Only after the independent ledger was frozen did the guarded importer compare
reviewed target occurrences with `20230530-tharu2.csv`. The block has 76 manual
and 77 legacy target occurrences: 71 agree exactly, leaving five manual-only
and six legacy-only multiset occurrences. Five paired differences are the
source-visible plain `t` versus legacy `ʈ` in item 52; the remaining legacy-only
occurrence results from the legacy collapse of the two RNS locality cells in
item 53. The rendered source controls all decisions, and the legacy data were
never accepted as transcription evidence.

Cumulatively through item 55, 828/890 legacy target occurrences agree exactly;
the multiset retains 82 manual-only and 62 legacy-only occurrences. Staging
remains refused at 880/3,360 reviewed cells. Item 56 `smoke`, physical p.41 /
printed p.36, right column, is next.
