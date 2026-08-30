# SIL Northern Dhule Bhils 2013 source package

This package represents Stephen Watters's 2013 SIL survey, Appendix C. The
mandatory source-ingestion checklist is active with the survey-wordlist and
OCR-heavy addenda. The canonical 133-page PDF is pinned in
`source_manifest.json`; its image-only wordlist was hand-keyed from rendered
pages. OCR/PDF text never supplied, seeded, or verified a lexical reading.

## Authoritative result

Appendix C contains 210 prompts × 13 lists = 2,730 cells. All 2,730 were
visually reviewed: 2,703 attested, 24 confirmed printed blanks, 3 ambiguous,
and 0 illegible. The twelve target lists account for 2,520 cells (2,497
attested, 21 blank, 2 ambiguous). The Toranmal comparison control accounts for
210 cells (206 attested, 3 blank, 1 ambiguous) and is excluded deliberately.

`manual_review.tsv` is the immutable coordinate topology; six OCR-blind,
hand-keyed chunks overlay it by unique `Item+Site_Code`. The importer rejects
OCR fields, duplicate/unknown/overlapping keys, coordinate drift, non-NFC
text, absent declaration/method stamps, and invalid status/form combinations.
`staged_audit.tsv` accounts for every cell. `staged_forms.csv` contains the
2,497 resolved target attestations only; source similarity numbers are retained
in the audit and stripped from staged form text. Shared source-specific
integration copies that file byte-for-byte to
`data/other/forms/20260829-sil-northern-dhule-bhils.csv`.

The pre-integration audit freezes the evidence before it opens any later-source
file. The manual six-chunk bundle SHA-256 is
`046ff03ef2af36c51f1b25538f081aabb7c28d3ccd1776d3ec545fad6463e8c1`;
the staged forms SHA-256 is
`5641b9d7ecfb44e6e644efba35e65223260291b7a8724b1fd25fac2fc94d3ed4`;
and the exhaustive staged audit SHA-256 is
`4bc5aa3bf41e79622494fea7426b6c77532ab4600c263a32179aa6c248b9c302`.
`render_hashes.tsv` records all 234 currently retained render/crop artifacts
(127,377,314 bytes; deterministic tree SHA-256
`816261371d0ada57996b2b1135267024629fcb3a7827b07bac4e53bc68f8ec43`).

`cross_source_reconciliation.tsv` is a post-freeze, 1,470-cell crosswalk. It
accounts for the 630 Astamba/Mundalwad/Toranmal cells republished in ESR
2015-012 and the 840 Mandvi/Amalwadi/Segwi/Shahana cells republished in ESR
2018-011. Later-source forms are comparison evidence only and never supply or
verify a Dhule reading. Among the 2018 cells, 261 contain one literally equal
form, 567 differ in publication representation, eight are blank in both
reports, and four are attested in Dhule but excluded as item 70 in Bareli.
The 2013 reading remains primary; later citations may merge only when complete
compiled lexical identity agrees, and differing readings must remain distinct.

## Reproduce

From `data/`:

```sh
python3 data/other/forms/raw_data/sil_northern_dhule_bhils_2013/manual_chunks/hand_keyed_items_176_210.py
python3 data/other/forms/raw_data/sil_northern_dhule_bhils_2013/import_northern_dhule_bhils.py --verify-pdf --write-unresolved --stage
python3 data/other/forms/raw_data/sil_northern_dhule_bhils_2013/preintegration_audit.py
python3 -m pytest -q tests/test_sil_northern_dhule_bhils_2013.py
```

Rendered 400/900-dpi pages are disposable files under `tmp/`, not ingestion
authority. `shared_integration_manifest.json` records the installed hash,
reference, language/dialect and sound-profile routes, exact exclusions, and
remaining consolidated build/browser/identity gates. `INTEGRATION.md` records
the evidence and the applied source-specific shared changes.
