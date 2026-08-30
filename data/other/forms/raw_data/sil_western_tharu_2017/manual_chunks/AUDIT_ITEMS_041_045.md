# Manual audit: items 41-45

The repository ingestion checklist, survey/comparative-table addendum, and
strict rendered-page manual-review policy controlled this block. Every one of
the 80 conceptual cells was independently read by eye from physical pp.38-39 /
printed pp.33-34 at 400 dpi and rechecked in 900/1200-dpi crops. PDF text, OCR,
and the legacy CSV did not supply, complete, normalize, infer, correct, or
verify any transcription.

## Accounting

- Items: 41 `sun`, 42 `moon`, 43 `sky`, 44 `star`, 45 `rain`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 80; source blanks: 0; ambiguous: 0; illegible: 0.
- Expanded occurrences: 90 (82 target candidates; 8 controls).
- Item 41 crosses physical pp.38-39 / printed pp.33-34; its BNT cell itself
  spans that page break.
- Item 44/SkP literal `tʌɾʌi + ja` is preserved as one printed response, not
  split or normalized. Item 45/DkR `(46)` is retained as a source qualifier.

Every form, response-group label, qualifier, and cell coordinate was visually
rechecked against the rendered source. All strings are NFC. The deterministic
ledger SHA-256 is
`3efa2ce93afb6b3d752b39782ab0500bf251e15c575573d30dbd9e660edf692b`.

## Unresolved site identities

No lexical reading is ambiguous or illegible. Ten coordinates remain
medium-confidence site assignments because both locality lists print as `RNS`.
Item 42 prints distinct group-3 forms `dʒoni` and `dʒõni`; item 44 prints
distinct group-1 forms `taɾe` and `taɾa`. Both pairs are provisionally mapped
by source occurrence order to metadata-row order. Every physical/printed page,
item, site key, column, visible response description, and candidate locality is
enumerated in `../unresolved_readings.tsv`.

## Post-entry reconciliation

Only after the independent ledger was frozen did the guarded importer compare
reviewed target occurrences with `20230530-tharu2.csv`. A fresh rendered-source
recheck retained all visible distinctions. The block has 82 manual and 82
legacy target occurrences: 59 agree exactly and 23 differ as paired multiset
occurrences. The source-led differences include dotted `i` where legacy has
`ɪ`, source nasalization, literal `+` segmentation, and rhotic/vowel details.
The legacy data were never accepted as transcription evidence.

Cumulatively through item 45, 707/735 legacy target occurrences agree exactly;
the multiset retains 49 manual-only and 28 legacy-only occurrences. Staging
remains refused at 720/3,360 reviewed cells. Item 46 `water`, physical p.39 /
printed p.34, is next.
