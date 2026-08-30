# Shared integration proposal

Do not treat the source package as globally integrated until these shared edits
and the consolidated build/QA pass are complete.

## Bibliography

Add to `cldf/sources.bib`:

```bibtex
@techreport{stahl2021korku,
  author = {Stahl, James},
  title = {A Sociolinguistic Survey of the Korku [kfq] Language Area},
  year = {2021},
  number = {2021-040},
  institution = {SIL International},
  address = {Dallas},
  series = {Journal of Language Survey Reports},
  url = {https://www.sil.org/resources/archives/90546},
  note = {Survey fieldwork conducted in 1985; PDF SHA-256 d17426da3788d66c95f05824483941e7d5468e154c66d43c6354262fda00190d; Appendix F image-only cells manually transcribed and reviewed; OCR retained only as a comparison scaffold}
}
```

Glottolog cross-check: `wsoc:Stahl:Korku`, reference 645875, type
`wordlist;socling`.

## Languages and dialects

No language row is needed: target language `ko` (Korku, `kork1243`) and
control language `Ni` (Nihali, `niha1238`) already exist. Add these exact
`cldf/dialects.csv` rows; coordinates stay blank because the report does not
pin the historical elicitation point precisely enough for this source package.

```csv
sil-korku-1985-chikli-ruma,dialect:ko:sil-korku-1985-chikli-ruma:Chikli%20Ruma,ko,CHI,Chikli Ruma,,,,Munda,"Chikli Ruma, 1985 SIL Korku survey lect",C
sil-korku-1985-khanapur-ruma,dialect:ko:sil-korku-1985-khanapur-ruma:Khanapur%20Ruma,ko,KHA,Khanapur Ruma,,,,Munda,"Khanapur Ruma, 1985 SIL Korku survey lect",C
sil-korku-1985-bagdara-ruma,dialect:ko:sil-korku-1985-bagdara-ruma:Bagdara%20Ruma,ko,BAG,Bagdara Ruma,,,,Munda,"Bagdara Ruma, 1985 SIL Korku survey lect",C
sil-korku-1985-warsari-ruma,dialect:ko:sil-korku-1985-warsari-ruma:Warsari%20Ruma,ko,WAR,Warsari Ruma,,,,Munda,"Warsari (called Marsari once in the report), 1985 SIL Korku survey lect",C
sil-korku-1985-moragao-bouriya,dialect:ko:sil-korku-1985-moragao-bouriya:Moragao%20Bouriya,ko,MOR,Moragao Bouriya,,,,Munda,"Moragao Bouriya, 1985 SIL Korku survey lect",C
sil-korku-1985-lahi-bouriya,dialect:ko:sil-korku-1985-lahi-bouriya:Lahi%20Bouriya,ko,LAH,Lahi Bouriya,,,,Munda,"Lahi Bouriya, 1985 SIL Korku survey lect",C
sil-korku-1985-amdhana-mawasi,dialect:ko:sil-korku-1985-amdhana-mawasi:Amdhana%20Mawasi,ko,AMD,Amdhana Mawasi,,,,Munda,"Amdhana Mawasi, 1985 SIL Korku survey lect",C
sil-korku-1985-khamalpur-bondoy,dialect:ko:sil-korku-1985-khamalpur-bondoy:Khamalpur%20Bondoy,ko,KHM,Khamalpur Bondoy,,,,Munda,"Khamalpur Bondoy, 1985 SIL Korku survey lect",C
```

## Profile and routing

- Add `"sil-korku"` to `PRESERVE_SOURCE_PROFILE_INPUT` in `make_cldf.py`.
- Route citation key `stahl2021korku` to `row_ipa = "sil-korku"` and
  `row_convert = True` beside the other SIL survey routes.
- Use `conversion/sil-korku.txt`; the focused test proves coverage of all
  1,521 installed forms.
- Register the source checklist/audit/manifest and update the discovery census
  from “five Korku locality lists plus Nihali” to “eight Korku locality lists
  plus one Nihali comparison list (210 cells each; installed source package).”
- No language row, cognacy route, comparison edge, or etymology change is needed.

## Commands and expected counts

```sh
python data/other/forms/raw_data/sil_korku_2021/import_korku.py --install
UV_CACHE_DIR=/tmp/uv-cache uv run --with pytest --with segments pytest -q tests/test_sil_korku_2021.py
make all
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

Source-local expected results: 1,890 audit cells; 1,890 manual visual reviews;
1,463 installed target cells; 1,521 installed rows after slash expansion; 216
confirmed blank target cells; one unresolved/illegible target cell; 210 excluded
Nihali comparison cells.
