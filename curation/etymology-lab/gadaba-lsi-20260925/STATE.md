# LSI Gadabā lect-identity audit

**All 222 staged records are excluded from this Dravidian etymology pass.** There are zero pending DEDR proposals, zero accepted assignments, and zero unexamined staged records. No `check-pass` was run because there are no scholarly assignments to validate. No source mapping, sidecar, importer, build, or browser database was changed.

The LSI source identifies `GADABA` as Glottocode `bodo1267`, Bodo Gadaba/Gutob (Munda), while the compiled import places its forms under `Language_ID=Gadaba`, whose language row is `mudh1235`, Mudhili Gadaba (Central Dravidian). The homonymous community/language names caused a source-to-parent mismatch:

| Evidence | Exact location | Finding |
|---|---|---|
| Raw import audit | `data/data/other/forms/raw_data/20260813-grierson-lsi-audit.csv:41` | `GADABA,Gadabā,bodo1267,222,Yes,Gadaba,Gadaba,unique existing-language name` |
| Survey dialect register | `data/cldf/dialects.csv:765` | `lsi_gadaba` retains `bodo1267` but assigns `Language_ID=Gadaba` and `Clade=C. Dravidian` |
| Language register | `data/cldf/languages.csv:80` | `Gadaba` is `mudh1235`, `C. Dravidian` |
| Compiled source records | `data/cldf/forms.csv`, filtered by the exact `dialect:Gadaba:LSI-GADABA:` tag | Exactly 222 distinct IDs, all `unlinked`; the IDs agree exactly with `survey-inventory.json` |
| External classification cross-check | [Glottolog Bodo Gadaba](https://glottolog.org/resource/languoid/id/bodo1267), [Glottolog Mudhili Gadaba](https://glottolog.org/resource/languoid/id/mudh1235), [ASJP Bodo Gadaba](https://asjp.clld.org/languages/BODO_GADABA) | Bodo Gadaba/Gutob (`bodo1267`) is separate from Mudhili Gadaba (`mudh1235`); ASJP classifies the former under Austroasiatic/Mundaic |

The existing homepage `fetchLemmaList` was queried read-only for each of the 222 exact survey forms in seven bounded chunks; 414 rows returned with no capped result. Those search results are retained as `discovery-*-results.json`, but an exact surface match to an imported LSI record or a DEDR comparandum cannot justify an **inherited Dravidian reflex** in a Munda source. Possible individual contact loans require separate donor and source evidence after the language identity is corrected.

`lect-identity-exclusion.json` lists every excluded persistent ID. These exclusions resolve the research queue for the present Dravidian task without approving an etymology or adjudicating a future Munda/contact pass. The import-language mismatch is a separate data-cleanup issue requiring source-mapping review and, eventually, the normal source/build workflow; that work is outside this no-rebuild task.
