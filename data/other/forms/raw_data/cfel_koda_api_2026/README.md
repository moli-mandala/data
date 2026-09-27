# Complete print-bounded publisher source stage

All2448publisher IDs matched to the2450-entry printed scope are represented in3223installed rows; all47publisher pilot keys survive. Independent full-scope audit:20/20pass, seed20260926112. Full build and database/browser gates remain deferred.

Current reproducible importer: `prepare_full.py --install`; it verifies the independent audit hashes before installation. Current evidence: `full-proposal-audit.jsonl`. The historical canonical filenames are retained for stable identity. See [complete recovery notes](../cfel_koda_2022/FULL_RECOVERY.md) for scope, transcription exceptions and validation.

## Retained pilot history

The files and account below describe the original bounded pilot; their counts do not describe the current canonical installation.

# CFEL Koda online dictionary: Adornments and Costumes

The source-ingestion checklist is active with the dictionary/glossary, OCR-heavy
print-comparator, and publisher website/API addenda. The canonical input here is
the Centre for Endangered Languages, Visva-Bharati online dictionary snapshot
queried on 25 September 2026. `queries.json` pins the complete 60 English labels
in the first domain of the publisher's Mahali inventory; these are search probes,
not Koda attestations. `snapshot.py` makes bounded, resumable POST requests to
the publisher API and retains response hashes and retrieval times. The source
package contains the 60-query `proposals.jsonl` factual snapshot and a 62-record
audit. Local raw API response cache is used for regeneration and includes
publisher prose; it is not part of the installed source CSV.

The publisher's 2022 *Koda–Bangla–Hindi–English* printed dictionary (ISBN
978-81-957226-1-7, 366 XPS pages) was rendered for independent comparison.
Its lexical text is vector outlines and cannot be extracted reliably as text.
The selected domain runs from printed pp. 7–17 (XPS pp. 10–20) with 5, then
nine pages of 6, then 1 printed entry: 60 in all. A fresh seeded
20-record visual check found English labels, native spellings and grammatical
categories corresponding to the online results; IPA is retained as an online
reading because print and API glyphs or spellings can differ. The print edition
is not silently substituted for the online snapshot.

The online query responses contain 60 Koda records in the selected domain and
two unrelated General-domain hits (Ring as a verb and Pant as a verb). The latter
are excluded. Of the 60 in-domain records, 47 have unambiguous bounded IPA fields
and are installed. Thirteen remain in `audit.jsonl` with typed transcription
issues: question marks where the publisher also uses a glottal-stop sign,
Armenian/Greek lookalike glyphs, an unpositioned aspiration marker, a turned-æ
glyph, an internal slash that may denote alternatives, or an IPA phrase shorter
than the native Lingerie headword. Thirteen secondary native spellings are also
retained in the audit without inventing unattested IPA. No phonological
meaning is inferred from those source-encoding anomalies. The Unicode Bengali
headwords and lexical glosses are source facts; descriptions and images are not
republished. The publisher site states all rights reserved.

`import_source.py --install` installs only the dated 15-column source CSV. It
preserves API record IDs as entry keys, concept IDs and response hashes in the
audit, and separates native spelling, original IPA, English gloss and grammar.
The installed text is NFC-normalized; exact API Unicode remains in the audit.
`conversion/cfel-koda-api.txt` routes the unambiguous IPA to house notation.
Dental marks on `t̪/d̪` follow the house `t/d` display convention; the source
marks remain in `Original` and the audit rather than being silently lost.
No etymological or borrowing relations are asserted. Koda is an existing
canonical Jambu language, and this source specifies no separate dialect.

Focused tests, source registry validation and source-row parse/conversion are
run locally. The complete CLDF build, full suite, compiled identity/graph/
reference checks and browser QA remain deferred under the user's explicit
no-database-build instruction. No remote execution was performed.
