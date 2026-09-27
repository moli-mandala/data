# Hislop 1866 Kuri/Muasi: full lexical source stage

Stephen Hislop, *Papers Relating to the Aboriginal Tribes of the Central Provinces*, edited by R. Temple, Nagpore, 1866. The original is public domain; Cornell's scan states no known US copyright restrictions. The scan is pinned by SHA256 in `manifest.json` and read from a local supplied PDF. No remote execution or database generation is involved.

The full source census covers the Kuri/Muasi column on all 43 printed Vocabulary pages (scan 53–95), the 20-row, five-collector comparative table (scan 109), and five comparative prose units in Essay p.27 (scan 47). `full_inventory.tsv` records all 359 main-table prompt cells, `comparison_inventory.tsv` all 100 collector cells, and `prose_inventory.tsv` all five prose units. This yields 464 audit units and 326 proposed rows. There are 154 printed absences, two explicitly held damaged Elliott readings, and 13 exact same-Hislop head-and-sense reuses with both source locators preserved. Printed blanks are absence marks, never ditto. The preface's approximate “some 362 words” is not the observed physical count of 359 prompt rows.

All main target cells were visually read against the scan, including blanks and page boundaries. Neighboring Gondi, Gayeti, Rutluk, Naikude, Kolami, Madi/Maria, Madia, Keikadi, Bhatrain, Parja, Telugu and Tamil columns are controls. The Gondi-only supplement and connected Gondi songs are outside this target-language package. Appendix VI's ethnography, personal/deity names, and regional explanatory words without Korku-language attribution are accounted as prose controls rather than falsely assigned to Korku.

The essay explicitly attributes six forms jointly to Kúrs and Kóls. They carry `uncertain` and an explicit language-attribution reason; their comparative prose survives without inferred ancestry edges. The comparison retains distinct collector witnesses. Only identical Hislop spelling and sense reuse the main record; comparison spelling differences (including printed `Minnco` and `Gomci`) remain separate. Elliott's buffalo and star cells are held because damaged type prevents a secure complete headword; provisional readings and alternatives remain in the audit. Main `Ing` followed by a grave-like mark is preserved with transcription uncertainty.

Original printed capitalization, acute accents and macron are retained in the raw form. The sound profile lowercases case and applies the project’s house w→v spelling, retaining other symbols and boundary marks: the source supplies no adequate orthographic key to justify the former speculative `bh→bʰ`, `ch→c` or circumflex-to-length conversions. No phonemic layer is invented. Explicit verb and adjective qualifiers are structured, as are unambiguous numeral and pronoun meanings; English plural prompts are not treated as evidence of target inflection. `Halka` alone receives the printed adjective label; only `Thora` has the wild-plantain qualification. Co-listed lexical alternatives are not automatically variant edges.

All forms use canonical Korku `ko`. The historical Kuri/Muasi source label is retained. Elliott's Kalibheet, Bradley's Gawil hills, and Voysey's Hoshungabad–Berar hill region have registered source-site dialect tags with blank coordinates; individual collector names remain provenance. Pearson's Korku/Muasi identification is reported by Temple. Exact elicitation points and speakers are not invented.

From the data repository:

```sh
.venv/bin/python data/other/forms/raw_data/hislop_kuri_muasi_1866/import_source.py
.venv/bin/python data/other/forms/raw_data/hislop_kuri_muasi_1866/audit_sample.py --seed 2026092637
.venv/bin/python -m pytest -q tests/test_hislop_kuri_muasi_1866.py
```

The importer creates `staged.csv` and `staged-audit.jsonl`. `--install` copies reviewed output to the canonical CSV and `audit.jsonl`. Legacy 18 keys are preserved despite corrected accents and expanded source scope. Earlier pilot inventories/reviews remain historical evidence, superseded by the full inventory. Independent review and installation status are recorded in the manifest. Full CLDF/build, full-suite and browser gates remain deferred under the user's explicit no-build instruction; source-stage closure does not claim compiled or app verification.

Final source-stage validation (2026-09-26): all326 rows installed; independent seed2026092637 passed0/20 material errors;26 focused source/profile/dialect checks passed in23.04s; canonical parser326/326 clean; metadata validation passed. The two damaged Elliott cells remain specific documented holds. No compiled or browser checks claimed.
