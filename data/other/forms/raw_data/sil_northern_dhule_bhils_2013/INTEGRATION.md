# Shared source-specific integration record

All 2,730 cells are manually resolved in the source-local audit. The exact
2,497 target attestations and the source-specific bibliography, language,
dialect and profile routes described below have now been installed. The one
consolidated CLDF/full rebuild, opaque compiled-identity reconciliation, global
audit regeneration, browser refresh/QA and commit remain deferred.

The independent evidence is frozen before reconciliation. See
`preintegration_manifest.json`: PDF SHA-256
`edeeeda98cb76624df1a0d70c765cc816ea463d75bc79ec20883c62e6fc1c482`,
manual-cell bundle SHA-256
`046ff03ef2af36c51f1b25538f081aabb7c28d3ccd1776d3ec545fad6463e8c1`,
staged target SHA-256
`5641b9d7ecfb44e6e644efba35e65223260291b7a8724b1fd25fac2fc94d3ed4`,
and 234-artifact render tree SHA-256
`816261371d0ada57996b2b1135267024629fcb3a7827b07bac4e53bc68f8ec43`.

## Installed bibliography

```bibtex
@techreport{watters2013northerndhule,
  author = {Watters, Stephen},
  title = {A Sociolinguistic Profile of the Bhils of Northern Dhule District},
  institution = {SIL International},
  type = {SIL Electronic Survey Report},
  number = {2013-004},
  year = {2013},
  pages = {1--125},
  url = {https://www.sil.org/resources/archives/52641}
}
```

The form citation suffix should be
`[Appendix C, printed p. N, item I, list CODE]`.

The older bibliography-only record under `bhildhule` had no form citations and
was retired during shared integration rather than leaving two records for the
same report. Installed rows use the canonical immutable key
`watters2013northerndhule`.

## Installed language rows

Add only the missing parents; RathwiBareli and PauriBareli already exist.

```csv
Vasavi,Vasavi,vasa1239,,,Bhil,"Northern Maharashtra and Gujarat; parent for four Watters 2013 Vasave lists",C
Noiri,Noiri,noir1238,,,Bhil,"Northern Maharashtra; parent for the Astamba Noiri and Mundalwad Bhilori lists",C
```

## Installed dialect rows

```csv
sil-dhule-2013-vasavi-kelpada,dialect:Vasavi:sil-dhule-2013-vasavi-kelpada:Kelpada,Vasavi,sil-dhule-2013-vasavi-kelpada,Kelpada,vasa1239,,,Bhil,"Kelpada Vasave survey list, northern Dhule district, Maharashtra",C
sil-dhule-2013-vasavi-dhanoura,dialect:Vasavi:sil-dhule-2013-vasavi-dhanoura:Dhanoura,Vasavi,sil-dhule-2013-vasavi-dhanoura,Dhanoura,vasa1239,,,Bhil,"Dhanoura Vasave survey list, northern Dhule district, Maharashtra",C
sil-dhule-2013-vasavi-digiamba,dialect:Vasavi:sil-dhule-2013-vasavi-digiamba:Digiamba,Vasavi,sil-dhule-2013-vasavi-digiamba,Digiamba,vasa1239,,,Bhil,"Digiamba Vasave survey list, northern Dhule district, Maharashtra",C
sil-dhule-2013-vasavi-amoda,dialect:Vasavi:sil-dhule-2013-vasavi-amoda:Amoda,Vasavi,sil-dhule-2013-vasavi-amoda,Amoda,vasa1239,,,Bhil,"Amoda Vasave survey list, northern Dhule district, Maharashtra",C
sil-dhule-2013-noiri-mundalwad,dialect:Noiri:sil-dhule-2013-noiri-mundalwad:Mundalwad,Noiri,sil-dhule-2013-noiri-mundalwad,Mundalwad,noir1238,,,Bhil,"Mundalwad Bhilori survey list, northern Dhule district, Maharashtra",C
sil-dhule-2013-noiri-astamba,dialect:Noiri:sil-dhule-2013-noiri-astamba:Astamba,Noiri,sil-dhule-2013-noiri-astamba,Astamba,noir1238,,,Bhil,"Astamba Noiri survey list, northern Dhule district, Maharashtra",C
sil-dhule-2013-pauri-bhusha,dialect:PauriBareli:sil-dhule-2013-pauri-bhusha:Bhusha,PauriBareli,sil-dhule-2013-pauri-bhusha,Bhusha,paur1238,,,Bhil,"Bhusha Pauri survey list, northern Dhule district, Maharashtra",C
sil-dhule-2013-rathwi-kangai,dialect:RathwiBareli:sil-dhule-2013-rathwi-kangai:Kangai,RathwiBareli,sil-dhule-2013-rathwi-kangai,Kangai,rath1242,,,Bhil,"Kangai Rathwi Pauri survey list, northern Dhule district, Maharashtra",C
```

Reuse these existing dialect IDs instead of creating duplicates:

- MAN → `sil-bareli-2018-bareli-pauri-mandvi`
- AML → `sil-bareli-2018-rathwi-pauri-amalwadi`
- SEG → `sil-bareli-2018-rathwi-pauri-segwi`
- SHA → `sil-bareli-2018-bareli-pauri-shahana`

The 2013 report is the earlier publication. Reconciliation should attach the
2013 citation/provenance to a single form identity or deliberately supersede
the later-source route; it must not install a second identical response.
Toranmal is comparison-only here and should not add a language/dialect row.

The exact post-freeze crosswalk is `cross_source_reconciliation.tsv` (1,470
rows; SHA-256
`25e11401d4d89c8698105d1138def1709f298473d00d61c36f0b1e891906f104`).
It proves:

- ESR 2015-012 republishes Astamba, Mundalwad, and Toranmal: 630/630 cells are
  accounted for (627 representation differences, three literal blank matches),
  and the Noira package excludes every republication in favour of ESR 2013-004.
- ESR 2018-011 explicitly labels Mandvi, Amalwadi, Segwi, and Shahana as coming
  from Watters's Dhule report: all 840 conceptual cells are accounted for.
  There are 261 literal single-form matches, 567 publication-representation
  differences, eight cells blank in both, and four item-70 cells attested in
  Dhule but excluded/disqualified in Bareli.
- Those four lists contribute 832 of the 2,497 staged Dhule attestations; the
  remaining 1,665 target attestations are not Bareli republications.

Integration must preserve both publications' diplomatic evidence. Merge
citations only where the complete compiled lexical identity agrees; otherwise
retain distinct source attestations. Never replace a frozen 2013 reading with
a later transcription merely because the reports reuse the same elicitation.

## Profile and routing

- Installed the audited source-local `conversion_profile.tsv` as
  `conversion/sil-northern-dhule-bhils.txt` and route only this source key.
- Routed only `watters2013northerndhule` through that profile.
- Strip numeric similarity labels only after retaining them in the audit;
  preserve alternatives, qualifiers, dental/retroflex distinctions,
  nasalization, aspiration, and source uncertainty.
- Exclude Toranmal control cells and every ambiguous/illegible cell from forms.

## Current validation commands and counts

```sh
python3 data/other/forms/raw_data/sil_northern_dhule_bhils_2013/import_northern_dhule_bhils.py --verify-pdf --write-unresolved --stage
python3 data/other/forms/raw_data/sil_northern_dhule_bhils_2013/preintegration_audit.py
python3 -m pytest -q tests/test_sil_northern_dhule_bhils_2013.py
```

Expected importer counts: 2,703 attested, 24 blank, 3 ambiguous, 0 illegible,
0 unreviewed. Source-local staging emits 2,497 target rows and an exhaustive
2,730-row audit. Shared source-specific registry/profile/reference tests pass.
Remaining work is the deliberately deferred consolidated `make all`, compiled
identity and survival checks, generated diff and `errors.txt` review, global
audit regeneration, full test suite, and (only if requested) browser refresh/QA.
