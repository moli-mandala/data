# UTN198 identity reconciliation plan

The review proposal is stable and remains uninstalled. This investigation reads existing compiled tables and runs pure helper functions on tiny in-memory fixtures; it does not build CLDF or modify identity state.

## Verified baseline

- `silkoya1985:utn:i198:v1` owns durable ID `f_clrv55fanq2my`, currently Original `vator`. The immutable source-key matcher preserves that ID when the original is corrected to `va:val` and its generated position changes.
- `silkoya1985:utn:i198:v2` has **no row** in either the identity registry or compiled source-key sidecar. Its `va` attestation is folded into `f_7mlyijnudplki`, whose owning key is `silkoya1985:jag:i198:v1`; that active node carries both old JAG/UTN citations and dialect tags.
- Therefore, there is no standalone old-v2 persistent ID to redirect. Redirecting `f_7mlyijnudplki` to the Utnoor entry would incorrectly hijack the surviving Jaganathapuram identity.
- Positional alias `135-1214` is not trustworthy evidence of the old v2 identity: it currently points to active Tshangla `f_vtiprk6vhpmco`. That unrelated ID must not be changed in this source recovery.

Exact selected registry/form/source-key/alias rows and scan counts are retained in `evidence-and-fixture-result.json`.

## Existing implementation route

`source_key_aliases.apply_source_key_aliases` reads CSVs in `data/other/form_aliases/` with `Retired_Source_Key,Target_Source_Key,Reason`. `assign_form_ids.py` calls it after generating the next registry and active ID set. The helper maps *previous registry IDs owned by the retired key* to the target's active ID; it also retargets old URL aliases. It rejects attempts to redirect active IDs. It does not create a separate public source-key lookup alias table.

After independent source acceptance, the guarded source installer may copy `proposed-source-key-alias.csv` to `data/other/form_aliases/20260927-koya.csv`. No shared helper change is needed. Keep the reconciliation in the source audit and installation manifest. For this verified baseline the helper correctly adds **zero** persistent ID aliases because no registry ID owns v2. The CSV still documents the retired source-key relation and safely handles a genuine keyed tombstone should one exist in an independently verified prior registry.

Do not insert a fabricated v2 identity, edit `form-identities.csv` by hand, or directly patch generated `form-id-aliases.csv`. Do not alias the surviving JAG ID or the unrelated positional-alias target. Source-key coverage validation must explicitly accept this audited retired-key mapping rather than assert that v2 remains an active compiled key.

## Reproducible focused test

Run `review_durable_alias.py` from the workspace with the data virtual environment, through local tmux. The script scans the existing tables with bounded memory and tests the actual `assign_ids` and `apply_source_key_aliases` functions:

1. Corrected/reordered UTN v1 retains `f_clrv55fanq2my`.
2. JAG retains `f_7mlyijnudplki`.
3. The absent v2 key produces no fabricated persistent alias.
4. A synthetic genuinely retired keyed ID redirects, including its old URL alias.
5. An attempted active-ID redirect raises an error.

All five checks passed. Following an eventual user-authorized build, separately verify those two active durable IDs and their corrected source-owned records, retired v2 source-key accounting, and unchanged unrelated Tshangla identity. That post-build gate is deferred; this test is not a full build or browser result.
