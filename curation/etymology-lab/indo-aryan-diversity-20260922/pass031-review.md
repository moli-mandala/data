# Zargari: pending local derivations and variant resolved

8 records / 8 rows: seven derived links and one variant. These were researched in passes006/030; the shared saver now validates compiled graph plus accepted sidecars in scratch, without rebuilding.

| # | Zargari | Proposed etymology | Evidence |
|---|---|---|---|
| 1 | **nāśay / nāśayipey / nāśāvd** ‘make run’ | Derived from same-source **nāś**, f_73nygaykqt5ka | Primary p141,3.15.1 explicitly derives the causative with -ay and perfect -āvd; infinitive -ipey is defined in3.17.3.1. Base ancestry7027 is already saved. |
| 2 | **čhindiv / čhindil** ‘be cut’ | Derived from same-source perfect stem **čhind**, f_ldvlw2qaq5ix6 | Primary p141,3.16.1 explicitly forms the mediopassive from the perfect stem with -iv/-il. Base ancestry5046 is already saved, with its alternative history noted. |
| 3 | **xaliv / xalil** ‘be eaten’ | Derived from same-source perfect stem **xāl**, f_l5cg5m5rzv2ks | Primary p141,3.16.1 explicitly gives this mediopassive pair. Base ancestry3865-2 is already saved; the local derivation is not assigned directly to ancient khādyate. |
| 4 | **čhiv** ‘language; tongue’ | Variant of same-source **čhib**, f_dmpttw4whm66w | Primary p128,2.3.19 explicitly prints čhiv < čhib as a word-final b/v alternation. Base ancestry5228 is already saved. |

Validation design: the shared helper validates all existing overlay rows with the pass, applies existing rows only to a temporary graph, then applies the proposed rows twice. Every accepted ancestor of this pass is checked for cycles, missing nodes, redirects and unresolved leaves. Unrelated edges are compared against the effective scratch baseline, while hashes retain the on-disk compiled-input invariant. Current edits preserve the helper’s existing compound/retraction work.

Focused tests: helper-effective-ancestry-tests.log (53 passed, one corpus-dependent test deselected), including previously-saved parents, rejection-only/rank2 parents, same-pass chains, composed cycles and unchanged on-disk graph. Full build deliberately not run.

Saved:8 records /8 accepted rows (7 derived,1 variant). Shared check-pass/save-pass/source-placement and independent exact-row audit passed; zero-change second application. Compiled forms/edges/identity hashes unchanged during save. No rebuild.

Test limitation: test_the_dictionary_self_reference_assignments_all_point_at_live_sub_entries expects2604 dictionary self-reference rows but current corpus has2603. This unrelated hard-coded inventory failure is retained in helper-effective-ancestry-initial-tests.log; no corpus counts were edited to make it pass. The initial missing-ancestor test also needed its expected error pattern to allow the existing structural validator’s parent-not-a-node wording; the rejection itself worked.
