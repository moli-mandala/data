# Manual audit: items 36-40

The repository ingestion checklist, survey/comparative-table addendum, and
strict rendered-page manual-review policy controlled this block. Every one of
the 80 conceptual cells was independently read by eye from physical p.38 /
printed p.33 at 400 dpi and rechecked in 1200/1600-dpi crops. PDF text, OCR, and
the legacy CSV did not supply, complete, normalize, or verify any transcription.

## Accounting

- Items: 36 `rope`, 37 `thread`, 38 `needle`, 39 `cloth`, 40 `ring`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 80; source blanks: 0; ambiguous: 0; illegible: 0.
- Expanded occurrences: 90 (83 target candidates; 7 controls).
- Item 38 crosses the left/right column boundary.
- Literal `(thick)`, `(men's)`, `(women's)`, and `(both)` qualifiers are
  preserved as evidence rather than lexical form text.

Every form, response-group label, qualifier, and cell coordinate was visually
rechecked against the rendered source. All strings are NFC. The deterministic
ledger SHA-256 is
`b6962e56e1f6d88635ab2195b85abfd7338e7722ff66567a06aa6463e6891f8e`.

## Unresolved site identities

No lexical reading is ambiguous or illegible. Ten coordinates remain
medium-confidence site assignments because both locality lists print as `RNS`.
Item 38's first RNS response is in the left column and its second response is
in the right column. Every affected physical/printed page, item, site key,
column, and candidate locality is enumerated in `../unresolved_readings.tsv`.

## Post-entry reconciliation

Only after the independent ledger was frozen did the guarded importer compare
reviewed target occurrences with `20230530-tharu2.csv`. A fresh rendered-source
recheck resolved typographic distinctions without copying legacy strings.
Through item 40, 648 of 653 legacy occurrences agree exactly. Five item-40
legacy spellings differ from the rendered source: BNT and TkN preserve
`mʊndʌɾija`, DKS preserves `mʊndiɾi`, RkM preserves `mũdʌɾija`, and SkP
preserves `ãŋɡutʰi`. The cumulative multiset therefore has 26 manual-only and
5 legacy-only occurrences; twelve of the manual-only occurrences are genuinely
additional source-visible alternatives.

Staging remains refused at 640/3,360 reviewed cells. Item 41 `sun` begins on
physical p.38 / printed p.33 and continues on physical p.39 / printed p.34.
