# Urali Census Tamil Nadu residual pass — 2026-09-25

Research is complete and pending user review. No accepted sidecar, source row, rebuilt database, commit, or remote has been changed.

The compiled `census2023tamilnadu` Urali column has 507 records. An independent exact-ID audit proves 58 older pending records, 8 older held records, 119 new proposed records, and 322 new research holds, with zero unexamined records and no duplicate current owners. This pass covers precisely the 441 residual IDs in `survey-inventory.json` and `ownership-baseline.json`; superseded older manifests are excluded from ownership.

All 441 residual forms were queried through Jambu's existing homepage fuzzy-search implementation in 13 bounded exact-form chunks; the retained requests, results, and logs account for 741 returned rows. The 119 proposals use 98 full cached DEDR primary articles, retained in `primary-evidence.json`. Twenty-four new pending Urali review manifests, batches 043–066, contain every exact source ID, citation, and assignment row.

The Census column also has 34 specific form–gloss or transcription conflicts listed in `source-gloss-conflict-indices.json`. For example, `manḍe` appears under ‘hair’ rather than ‘head’, `toṭa` is elicited both as ‘garden’ and ‘bullet’, `adə` is called ‘it/this’ despite its distal stem, and `day` occurs under both ‘cut’ and ‘split wood’. These rows remain held. Other multiword responses, weakly compared stems, and probable loans without attested immediate donors are held without changing the source.

The canonical read-only `make check-pass` passed for `urali-census-tamilnadu-001`: 119 affected records, 119 rows, 98 parent nodes, and zero changes on second application. `verify-checkpoint.py` independently compared canonical rows with the pending manifests and full-source ownership; `checkpoint-001-validation.json` reports manifest and coverage agreement. Review entry point: `REVIEW.md`.
