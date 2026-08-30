# Applied shared integration — JLSR 2022-005

All 2,310 cells and the complete Dumripada replacement relation have been manually reviewed. Shared forms, bibliography, dialect rows, profile routing, and replacement handling are applied; generated builds and browser QA remain deferred to the requested consolidated rebuild.

## Bibliography

```bibtex
@techreport{mathew2022bonda-further,
  author = {Mathew, Chacko},
  title = {The Bonda: Further Sociolinguistic Survey},
  year = {2022},
  institution = {SIL International},
  series = {Journal of Language Survey Reports},
  number = {2022-005},
  url = {https://www.sil.org/resources/archives/92609},
  note = {Survey completed in 2002; Appendix A manually transcribed from rendered source pages}
}
```

## Dialect policy

Create new `re` dialect rows for the two genuinely new 2002 collection sites:

```csv
sil-bonda-further-2002-podeiguda-u-bonda,dialect:re:sil-bonda-further-2002-podeiguda-u-bonda:Podeiguda%20U.%20Bonda,re,POD,Podeiguda U. Bonda,bond1245,,,Munda,"Podeiguda, Upper Bonda; 2002 further SIL survey",C
sil-bonda-further-2002-bondapada-u-bonda,dialect:re:sil-bonda-further-2002-bondapada-u-bonda:Bondapada%20U.%20Bonda,re,BON,Bondapada U. Bonda,bond1245,,,Munda,"Bondapada, Upper Bonda; 2002 further SIL survey",C
```

Reuse existing dialect `sil-bonda-didayi-1997-dumripada-u-bonda` for the same Dumripada site. Section 2.1 explicitly says the checked current list replaces the old Dumripada list. Integration must reconcile it row by row rather than installing two conflicting current values or discarding source citations.

The eight JLSR 2022-004 comparison lists remain audit-only. Reuse their prior site codes solely in the comparison audit; do not install or overwrite them from this report.

## Profile and build

- Route source key `mathew2022bonda-further` through a finalized Bonda survey profile derived after complete review.
- Preserve source colons and final apostrophes in `Original`/`Phonemic`. Colon-to-length conversion is provisional; final apostrophe is distinct from printed `ʔ` and must not silently become a glottal stop.
- Similarity groups remain descriptive notes, not cognacy claims.
- Final target rows are installed at `data/other/forms/20260829-sil-bonda-further.csv`; bibliography, dialects, and profile routing are integrated and focused tests pass.
- The checked 2002 Dumripada list is the published current list. The 1997 rows are excluded from current build input but preserved unchanged in their raw source package and exhaustively related in `dumripada_replacement_reconciliation.tsv`.
- Consolidated build, full pytest, graph validation, and browser QA remain deferred.
