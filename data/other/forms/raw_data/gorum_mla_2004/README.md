# Complete Gorum Munda Lexical Archive snapshot

The source is Patricia J. Donegan and David Stampe's 2004 *Munda Lexical Archive* Gorum file, preserved as the exact 2020-10-22 Wayback ASCII snapshot (`source-wayback.txt`; archived Last-Modified 2014-09-05). Its SHA-256 is pinned in the importer and manifest. The source-specific same-conditions reuse notice in `LICENSE` applies to this snapshot and its derivatives; no generic Creative Commons licence is inferred. No OCR is used.

The full physical-line census contains 6,564 nonblank body lines. Two explicit continuation lines are joined to their preceding lexical records, leaving **6,562 audited units**. There are 5,824 hash-numbered chunks and 5,743 distinct hash-numbered IDs; the numeral *mulgi* ‘nine’ also prints ID 22960 without its hash. All body units, including unnumbered lexical entries, headings, empty numeric records and control separators, are accounted. The 559 legacy entry keys survive. Numbered keys depend on source identity; unnumbered keys use exact pinned body lines. Separate occurrence keys protect conflicting duplicate IDs, and stable sense/head/witness children preserve expanded lexical distinctions.

The installed source CSV has **7,493 rows** from 5,694 ingested units. Seventy-five repeated lexical units reuse their first occurrence while retaining raw provenance and distinct commentary; 46 units have empty or placeholder definitions and remain held; 747 units are unglossed grouping headings, editorial context, separators or empty records. `duplicate-reconciliation.json` documents all 81 repeated numeric IDs. Six sense scopes across five entries explicitly name a witness absent from their headword declarations; these unresolved alignments remain visible in the audit rather than being attributed to a guessed form.

The parser separates headword declarations, bounded senses, witness restrictions, grammar, bracket transcriptions and source analysis. Headings such as `<aj>::` cannot donate a gloss to their following child. Genuinely glossed unnumbered *buluG*, *suG*, *tam* and other entries are retained, as is an embedded lexical remark under entry 6680. `(Z)`, `(A)` and joint witnesses remain provenance, not invented dialects. The rare `(Bh.)` and `(S)` target-head witnesses are retained with typed unresolved provenance; their real-world referents are not inferred. All forms use existing `go` (Gorum, `pare1266`); no coordinates or new varieties are invented.

Uppercase letters and source punctuation—including `D`, `G`, `J`, `R`, `T`, `?`, `~`, `=`, brackets and boundaries—are preserved by the complete identity profile. They are not silently converted to guessed IPA. Unique bracket transcriptions with defensible full-form alignment enter `Phonemic` exactly as printed. Partial component transcriptions and ambiguous alternate alignment remain clearly labelled in Notes and raw audit. Native script remains blank. Source grammar is structured where supported, including verb phrases, person-number, case, affixes and explicit donor claims. The source's contradictory person-number descriptions in entries 19830 and 24240 remain typed uncertainties; no preferred analysis is imposed.

Explicit same-witness alternate lists can create variant edges, but slash/backslash analyses and source-questioned lists do not acquire guessed relationships. Morphological analysis, comparisons and qualified donor claims remain in Etymology. Named non-Munda donor statements and explicit Loan labels receive `loanword`; no unattested donor node or cognacy edge is created. All distinct source IDs remain distinct even where spelling and gloss repeat.

`transcription` repairs are limited to recoverable delimiters: missing closing gloss quotes at 3650/34620/34622, the closing headword marker at 24670, and three restarted quoted synonym groups at 2890/2892/4140. Exact raw source text is always retained. The latter groups have separately printed Z/A scopes and cannot leak synonyms between witnesses. The full source audit records ambiguity rather than narrowing the inventory to easy spellings.

Independent full-scope audit reports retain failed samples and corrected error classes. Fresh final pass 4 has **0/20 material errors** (seed 2026092614); its hash matches the canonical installed CSV. All **26 source/profile/dialect tests passed**, including scoped parser survival of all 7,493 rows without conversion errors. Settings validation passed for 275 files and 262 citation keys. Exact results are retained in the manifest. The historical `sample-20260925.jsonl` describes only the earlier 559-row pilot and is retained as history, not evidence for current completeness.

From the data repository:

```sh
.venv/bin/python data/other/forms/raw_data/gorum_mla_2004/import_source.py
.venv/bin/python data/other/forms/raw_data/gorum_mla_2004/audit_sample.py --seed 2026092615
.venv/bin/python data/other/forms/raw_data/gorum_mla_2004/import_source.py --install
.venv/bin/python -m pytest -q tests/test_gorum_mla_2004.py tests/test_sound_profiles.py tests/test_dialects.py
.venv/bin/python source_meta.py
```

The default importer writes only source-local staging; `--install` writes the canonical source CSV and audit. Full CLDF build, compiled identity/graph/reference checks and full-suite gates remain deferred under the user's instruction. Browser database construction and representative application QA require an explicit refresh request. This source stage does not publish a release.
