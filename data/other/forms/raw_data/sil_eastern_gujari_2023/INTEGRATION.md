# Shared integration proposal

This source-local task does not edit shared registries or generated CLDF. Apply
the following after the parallel source packages have landed.

## Bibliography

Add to `cldf/sources.bib`:

```bibtex
@techreport{hugoniot-polster-ahmad-rajan2023easterngujari,
  author = {Hugoniot, Ken and Polster, Dietmar and Ahmad, Bashir and Rajan, Kennedy},
  title = {A Sociolinguistic Profile of Eastern Gujari},
  year = {2023},
  number = {2023-002},
  institution = {SIL International},
  address = {Dallas},
  series = {Journal of Language Survey Reports},
  url = {https://www.sil.org/resources/archives/95899},
  note = {Fieldwork conducted 1996; report dated 1997; Appendix B wordlists visually verified cell by cell; PDF SHA-256 41352b2db97dbd059a1bc229a8ed370fed700c1726f3886a580cba586137475e; six Pakistan Gujari lists are audit-only reprints of SSNP volume 3}
}
```

## Language and dialects

Reuse the existing base language row:

```csv
Goj,Gujari,guja1253,34.043002,73.225443,Rajasthanic,,C
```

Add the eight rows below to `cldf/dialects.csv`. The source supplies districts
but no defensible collection coordinates, so latitude/longitude are blank and
quality is C. Historical state labels are retained as source evidence.

```csv
sil-eastern-gujari-1996-udhampur,dialect:Goj:sil-eastern-gujari-1996-udhampur:Udhampur,Goj,UDH,Udhampur,guja1253,,,Rajasthanic,"Udhampur district, Jammu and Kashmir; 1996 survey list Udhampur",C
sil-eastern-gujari-1996-jammu,dialect:Goj:sil-eastern-gujari-1996-jammu:Jammu,Goj,JAM,Jammu,guja1253,,,Rajasthanic,"Jammu district, Jammu and Kashmir; 1996 survey list Jammu",C
sil-eastern-gujari-1996-chamba,dialect:Goj:sil-eastern-gujari-1996-chamba:Chamba,Goj,CHA,Chamba,guja1253,,,Rajasthanic,"Chamba district, Himachal Pradesh; 1996 survey list Chamba",C
sil-eastern-gujari-1996-rampur,dialect:Goj:sil-eastern-gujari-1996-rampur:Rampur,Goj,RAM,Rampur,guja1253,,,Rajasthanic,"Shimla district, Himachal Pradesh; 1996 survey list Rampur",C
sil-eastern-gujari-1996-nalagarh,dialect:Goj:sil-eastern-gujari-1996-nalagarh:Nalagarh,Goj,NAL,Nalagarh,guja1253,,,Rajasthanic,"Solan district, Himachal Pradesh; 1996 survey list Nalagarh",C
sil-eastern-gujari-1996-dehra-dun,dialect:Goj:sil-eastern-gujari-1996-dehra-dun:Dehra%20Dun,Goj,DEH,Dehra Dun,guja1253,,,Rajasthanic,"Dehra Dun district, Uttar Pradesh (source-era label); 1996 survey list Dehra Dun",C
sil-eastern-gujari-1996-kotdwara,dialect:Goj:sil-eastern-gujari-1996-kotdwara:Kotdwara,Goj,KOT,Kotdwara,guja1253,,,Rajasthanic,"Uttar Pradesh (district not supplied); 1996 survey list Kotdwara",C
sil-eastern-gujari-1996-haldwani,dialect:Goj:sil-eastern-gujari-1996-haldwani:Haldwani,Goj,HAL,Haldwani,guja1253,,,Rajasthanic,"Naini Tal district, Uttar Pradesh (source-era label); 1996 survey list Haldwani",C
```

Do not add dialect rows for Urdu or for the six Pakistan reprints. Preserve
the existing primary-source SSNP dialects `SSNP-gojri-CHT`, `-SSW`, `-GLT`,
`-KGH`, `-NAK`, and `-CAK`.

## Profile and routing

- Add `sil-eastern-gujari` to the profile-name allowlist in `make_cldf.py`.
- Route source key `hugoniot-polster-ahmad-rajan2023easterngujari` to
  `conversion/sil-eastern-gujari.txt` in `make_cldf.py` and the parallel route
  in `utils.py`.
- Preserve exact source IPA as `Original`; conversion maps `dʒ`/`tʃ` to Jambu
  `j`/`c`, `ɽ` to `ṛ`, and standalone `j` to `y`.
- Register the audit/manifest in the shared source-audit registry.
- Do not create graph or cognacy claims from lexical-similarity numbers.

## Commands and expected results

```sh
python data/other/forms/raw_data/sil_eastern_gujari_2023/import_eastern_gujari.py --install
UV_CACHE_DIR=/tmp/uv-cache uv run --with pytest --with segments --with pypdf \
  pytest -q tests/test_sil_eastern_gujari_2023.py
make all
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

Expected source-local results: 210 prompts, 15 lists, 3,150 visually reviewed
cells, 3,117 attested cells, 33 blanks (25 target/eight non-target), 1,655
attested target cells, 1,754 printed target alternatives, one exact duplicate
alternative audit-only, 1,753 installed forms, 1,254 attested SSNP reprint
cells excluded, 208 attested Urdu cells excluded, and 3,150 audit rows. There
are zero ambiguous, clipped, illegible, or unresolved cells. Full build, full
suite, generated CLDF, and browser QA remain deferred.

