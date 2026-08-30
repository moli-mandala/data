# Shared integration decisions

The manual ledger reached 3,360/3,360 cells and the guarded importer permits
staging. Source-specific shared integration is complete; the consolidated CLDF
and browser build remain deferred.

The existing bibliography key `webster` now records exact Appendix B coverage,
the canonical SHA-256, source-local audit paths, editor attribution, and
`ocr = {No}`. No accepted transcription is OCR/PDF-text-derived.

Canonical parent languages `Buksa`, `Rana`, `Kathoriya`, `Sunha`, `Dang`, and
`Chitwan` are reused. Site-registry decisions are:

- split legacy combined `Tharu-RNS` into `Tharu-RNS-Sisaikhara` and
  `Tharu-RNS-Sisana`; every conceptual assignment retains the duplicate-code
  uncertainty and exact source coordinate;
- add `Tharu-CCC` beneath `Chitwan` and migrate the old bare-Chitwan source rows;
- preserve metadata `RKM` / response `RkM` as an alias while reusing `Tharu-RkM`;
- retain the existing `TkN` parent route (`Rana`) provisionally while preserving
  the source label `Thakur Tharu` and classification caveat in every TkN row.

The historical `tharu2`/`chattisgarhi` route is replaced by the explicit
`sil-western-tharu` preservation profile. Its exhaustive fixture covers every
installed source form without silently normalizing visibly distinct symbols.
Printed group numbers remain Notes and never become cognacy or etymology;
`Parameter_ID` remains blank.

`20230530-tharu2.csv` is replaced, not appended: 3,560 target forms install in
the 15-column rich schema with unique immutable source-local keys. The 98 target
source blanks and all 210 Standard Hindi controls are audit-only. A frozen copy
of the retired 3,570-row legacy file remains beside the source package solely to
reproduce the post-freeze 2,794 exact / 766 manual-only / 754 legacy-only
reconciliation.

Shared BibTeX, dialect/profile routing, installed forms, checklist registry, and
focused tests are integrated. Consolidated CLDF/full-suite validation, graph
review, durable opaque-ID reconciliation, browser QA, and commit remain deferred.
