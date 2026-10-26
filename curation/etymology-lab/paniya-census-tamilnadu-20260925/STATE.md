# Paniya Census Tamil Nadu residual pass — 2026-09-25

Research is complete and pending user review. No accepted sidecar, source row, rebuilt database, commit, or remote has been changed.

The compiled `census2023tamilnadu` Paniya column has 499 records. An independent exact-ID audit proves 38 older pending records, 1 older held record, 60 new proposed records, and 400 new research holds, with zero unexamined records and no duplicate owners. The pass covers precisely the 460 residual IDs in `survey-inventory.json` and `ownership-baseline.json`.

All 460 residual forms were queried through Jambu's existing homepage fuzzy-search implementation in 14 bounded exact-form chunks; the retained requests, results, and logs account for 672 returned rows. The 60 proposals use 52 fully cached DEDR primary articles, retained in `primary-evidence.json`. Eleven new pending Paniya review manifests, batches 015–025, contain every exact source ID, citation, and assignment row.

There are conspicuous form–gloss mismatches in this source column. `source-gloss-conflict-indices.json` identifies 147 exact residual records, including **ellu** ‘blood’ where the form means bone, **kaṇṇu** ‘chest’ where it means eye, **irəcci** ‘fever’ where it means meat, and the identical **kuḷikiñcai** in separate ‘bathe’ and ‘buy’ records. These rows remain held without changing the source. Other compounds, weakly compared stems, and probable loans without an attested immediate donor are also held.

The canonical read-only `make check-pass` passed for `paniya-census-tamilnadu-001`: 60 affected records, 60 rows, 52 parent nodes and zero changes on second application. `verify-checkpoint.py` independently compared canonical rows with the pending manifests and full-source ownership; `checkpoint-001-validation.json` reports manifest and coverage agreement. Review entry point: `REVIEW.md`.
