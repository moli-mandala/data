# Complete paired-print source stage

All 2450 entries in both printed editions are represented in3230 installed rows; all17print pilot keys survive. Independent full-scope audit:20/20pass, seed20260926112. Full build and database/browser gates remain deferred.

Current reproducible importer: `prepare_full.py --install`; it verifies the independent audit hashes before installation. Current evidence: `full-proposal-audit.jsonl`. The historical canonical filenames are retained for stable identity. See [FULL_RECOVERY.md](FULL_RECOVERY.md) for scope, transcription exceptions and validation.

## Retained pilot history

The files and account below describe the original bounded pilot; their counts do not describe the current canonical installation.

# CFEL Koda 2022: bounded print pilot

This package records a **17-entry pilot**, not the complete 2,450-word *English--Hindi--Bangla--Koda Dictionary* (Pradhan and Tripathi, eds., first edition 2022, ISBN 978-81-957226-0-0). The [publisher PDF](https://cfelvb.in/uploads/pdf/English-Hindi-Bangla-Koda.pdf) has 371 PDF pages. Its SHA-256 is `0d48856a0173c671318b0f7eed80d45d73f92ef183aaa545573afc01bac2180e`. PDF page numbers are three greater than printed page numbers in the inventoried section. The PDF is a local research cache, not a repository asset.

`reviewed-inventory.jsonl` accounts for **all 53 entries** on PDF pages 10, 25, 76, 120, 180, 260, and 345. Seventeen were selected after visual page-image checking of English head, Koda spelling and item position. Thirty-six were excluded: 33 outside the bounded pilot, plus *Cow* (publisher IPA omits a printed final visarga), *Two* (publisher IPA includes `?`), and *Die* (API Koda form differs from print). The 20-row visual sample comprises all 17 selected entries and these three explicit deferrals, with no material errors in the recorded print comparison. The page-image checks are manual evidence; the importer cannot reperform them without the copyrighted PDF.

`api-evidence.jsonl` preserves only the queried head, stable publisher Koda record ID, Koda Unicode, IPA, domain, update time, retrieval time and SHA-256 of each complete HTTP response. Full API responses are discarded in memory. The selected Unicode was checked against print for every row. IPA comes from the paired publisher API, **not from the printed entry**. `import_source.py --install` validates the exact pairing and writes the 17-row CSV without building the database. It rejects IPA containing `?`. The source profile maps the unambiguous publisher IPA to house transcription; `Native` preserves the visually checked printed Bengali-script Koda form. No source etymology, borrowing or genealogical claim is supplied. No dialect site is inferred from the publisher's broad region.

The previously compiled 620 Koda entries include 611 survey entries from two Bangladesh sites. This CFEL edition draws on West Bengal fieldwork and is an independent witness. Five of the 17 English labels (*Cat*, *Face*, *Finger*, *Month*, *Morning*) intersect the compiled Koda gloss set; none is a same-source duplicated entry. Existing online CFEL adornments rows are a separate dated API source; this pilot cites the 2022 print edition with page/item locators and a distinct source key.

The PDF states © Visva-Bharati, and no open redistribution licence has been identified. Following the project's established extracted-lexical-facts policy for other copyrighted dictionaries, only attributed lexical facts are in the installed CSV. The PDF, page images, prose definitions and complete API payloads are excluded. **Review the public-release decision before redistributing this package or derived database.** The current work is local source-stage installation only. Full database build, reference/graph generation and browser QA remain deferred under the user's explicit instruction not to build the DB.
