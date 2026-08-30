# Manual audit: items 46-50

The repository ingestion checklist, survey/comparative-table addendum, and
strict rendered-page manual-review policy controlled this block. Every one of
the 80 conceptual cells was independently read by eye from physical pp.39-40 /
printed pp.34-35 at 400 dpi and rechecked in 900/1200-dpi crops. PDF text, OCR,
and the legacy CSV did not supply, complete, normalize, infer, correct, or
verify any transcription.

## Accounting

- Items: 46 `water`, 47 `river`, 48 `cloud`, 49 `lightning`, 50 `rainbow`.
- Reviewed: 80/80 cells (75 target; 5 Standard Hindi control).
- Attested: 79; source blanks: 1; ambiguous: 0; illegible: 0.
- Expanded occurrences: 84 (78 target candidates; 6 controls).
- Item 46 crosses physical pp.39-40 / printed pp.34-35; its Hindi cell itself
  spans the break. Item 49 crosses the left/right column boundary.
- Item 47/RNS occurrence 2 has an extra literal `1` after the group label.
- Six item-48 `(43)` cross-references are preserved as qualifiers.

The single blank is item 50/CCC, physical p.40 / printed p.35, right column.
The complete printed item has no CCC row and instead prints DGC responses under
groups 1 and 3. Both DGC responses are retained in the DGC cell; no form is
invented, copied, or reassigned to CCC.

Every form, response-group label, qualifier, blank, and cell coordinate was
visually rechecked against the rendered source. All strings are NFC. The
deterministic ledger SHA-256 is
`c21c55219b4ece0de5fe7e8c5d1accdf50f69462763d0aa57d65db6e4d7c541a`.

## Unresolved site identities

No lexical reading is ambiguous or illegible. Ten coordinates remain
medium-confidence site assignments because both locality lists print as `RNS`.
The item-47 second occurrence includes the extra printed `1`; item 49's two RNS
responses fall on opposite sides of the column break; and item 50 prints
distinct `dʰʌnuʃ` and `dʰʌnuʃman` responses. Every physical/printed page,
item, site key, column, visible response description, and candidate locality is
enumerated in `../unresolved_readings.tsv`.

## Post-entry reconciliation

Only after the independent ledger was frozen did the guarded importer compare
reviewed target occurrences with `20230530-tharu2.csv`. A fresh rendered-source
recheck retained all visible distinctions. The block has 78 manual and 78
legacy target occurrences: 50 agree exactly and 28 differ as paired multiset
occurrences. The source-led differences include dotted `i` where legacy has
`ɪ`, source `u` where legacy has `ʊ`, visible aspiration, and the absence of a
legacy length mark on item 47/SkP. The legacy data were never accepted as
transcription evidence.

Cumulatively through item 50, 757/813 legacy target occurrences agree exactly;
the multiset retains 77 manual-only and 56 legacy-only occurrences. Staging
remains refused at 800/3,360 reviewed cells. Item 51 `wind`, physical p.40 /
printed p.35, is next.
