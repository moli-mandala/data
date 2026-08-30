# Shared integration proposal

Shared bibliography, dialect registry, profile routing, audit-registry, census,
and focused metadata integration were applied by the coordinating task on
2026-08-28. The consolidated CLDF build and browser QA remain deferred until
all parallel source packages have landed. The sections below retain the source-
local proposal and expected counts as an audit trail.

## Bibliography

Add to `cldf/sources.bib`:

```bibtex
@techreport{blair2021kullu,
  author = {Blair, Frank},
  title = {A Sociolinguistic Profile of Kullu District, Himachal Pradesh},
  year = {2021},
  number = {2021-009},
  institution = {SIL International},
  address = {Dallas},
  series = {Journal of Language Survey Reports},
  url = {https://www.sil.org/resources/archives/88003},
  note = {Fieldwork conducted in 1985--1986; PDF SHA-256 720a97198254160bfa88a9557b33955b2814878e346901ff399cacc53d5c4fdd}
}
```

## Base language

No new `cldf/languages.csv` row is required. All 16 sites attach to the existing
row `kul,Kullui,kull1236,31.83,77.38,W. Pahari,,`. This follows the report's
explicit conclusion that Kullui, Inner Seraji, and Outer Seraji constitute one
Kullui Pahari language area. Ani is retained as the report's Outer Seraji survey
lect; no modern reclassification is inferred.

## Dialects

Add these rows to `cldf/dialects.csv`. Precise coordinates and quality are left
blank because Appendix C gives localities but not historical collection-point
coordinates. The report itself prints `Banjar Tehsil (?)` for Maraur and that
qualification is preserved.

```csv
sil-kullu-1985-churla,dialect:kul:sil-kullu-1985-churla:Churla,kul,sil-kullu-1985-churla,Churla,,,,W. Pahari,"Churla/Lag Valley, Kullu Tehsil; Kullui survey lect; 10 April 1985",
sil-kullu-1985-loren,dialect:kul:sil-kullu-1985-loren:Loren,kul,sil-kullu-1985-loren,Loren,,,,W. Pahari,"Loren/S. Kullu Valley, Kullu Tehsil; Kullui survey lect; 10-11 April 1985",
sil-kullu-1985-shalwar,dialect:kul:sil-kullu-1985-shalwar:Shalwar,kul,sil-kullu-1985-shalwar,Shalwar,,,,W. Pahari,"Shalwar Village, Banjar Tehsil; Inner Seraji survey lect; 19 April 1985",
sil-kullu-1985-chinninal,dialect:kul:sil-kullu-1985-chinninal:Chinninal,kul,sil-kullu-1985-chinninal,Chinninal,,,,W. Pahari,"Chinninal Village, Banjar Tehsil; Inner Seraji survey lect; 18 April 1985",
sil-kullu-1985-shangarh,dialect:kul:sil-kullu-1985-shangarh:Shangarh,kul,sil-kullu-1985-shangarh,Shangarh,,,,W. Pahari,"Shangarh, Banjar Tehsil; Inner Seraji survey lect; 20 April 1985",
sil-kullu-1985-manali,dialect:kul:sil-kullu-1985-manali:Manali,kul,sil-kullu-1985-manali,Manali,,,,W. Pahari,"Manali, Kullu Tehsil; Kullui survey lect; 10 April 1985",
sil-kullu-1985-raila,dialect:kul:sil-kullu-1985-raila:Raila,kul,sil-kullu-1985-raila,Raila,,,,W. Pahari,"Raila Village, Kullu Tehsil; Inner Seraji survey lect; 25 April 1985",
sil-kullu-1985-maraur,dialect:kul:sil-kullu-1985-maraur:Maraur,kul,sil-kullu-1985-maraur,Maraur,,,,W. Pahari,"Maraur Village, Banjar Tehsil (?); Inner Seraji survey lect; 3 May 1985",
sil-kullu-1985-sidua,dialect:kul:sil-kullu-1985-sidua:Sidua,kul,sil-kullu-1985-sidua,Sidua,,,,W. Pahari,"Sidua, Banjar Tehsil; Inner Seraji survey lect; 3 May 1985",
sil-kullu-1985-jibhi,dialect:kul:sil-kullu-1985-jibhi:Jibhi,kul,sil-kullu-1985-jibhi,Jibhi,,,,W. Pahari,"Jibhi, Banjar Tehsil; Inner Seraji survey lect; 9 May 1985",
sil-kullu-1985-bathad,dialect:kul:sil-kullu-1985-bathad:Bathad,kul,sil-kullu-1985-bathad,Bathad,,,,W. Pahari,"Bathad, Banjar Tehsil; Inner Seraji survey lect; 10 May 1985",
sil-kullu-1985-garsah,dialect:kul:sil-kullu-1985-garsah:Garsah,kul,sil-kullu-1985-garsah,Garsah,,,,W. Pahari,"Garsah, Kullu Tehsil; Kullui survey lect; 22 May 1985",
sil-kullu-1985-kullu,dialect:kul:sil-kullu-1985-kullu:Kullu,kul,sil-kullu-1985-kullu,Kullu,,,,W. Pahari,"Kullu HQ, Kullu Tehsil; Kullui survey lect; 6 June 1985",
sil-kullu-1985-bhutti,dialect:kul:sil-kullu-1985-bhutti:Bhutti,kul,sil-kullu-1985-bhutti,Bhutti,,,,W. Pahari,"Bhutti Village (Lag Valley), Kullu Tehsil; Kullui survey lect; 1 June 1985",
sil-kullu-1985-manikaran,dialect:kul:sil-kullu-1985-manikaran:Manikaran,kul,sil-kullu-1985-manikaran,Manikaran,,,,W. Pahari,"Manikaran, Kullu Tehsil; Kullui survey lect; 21 May 1985",
sil-kullu-1985-ani,dialect:kul:sil-kullu-1985-ani:Ani,kul,sil-kullu-1985-ani,Ani,,,,W. Pahari,"Ani, Ani Tehsil; Outer Seraji survey lect; 17 June 1985",
```

## Sound profile and routing

- Route citation key `blair2021kullu` to `conversion/sil-kullu.txt` in the
  source-profile selection logic in `utils.py`/`make_cldf.py`.
- The source-local profile covers every grapheme in the manually normalized IPA
  forms. It is intentionally identity-like: the handwriting has already been
  reviewed and normalized rather than transliterated mechanically.
- No etymology, historical cognacy, borrowing, or derivation routing is needed.

## Commands and expected results

```sh
python data/other/forms/raw_data/sil_kullu_2021/import_kullu.py --install
UV_CACHE_DIR=/tmp/uv-cache uv run --with pytest --with segments pytest -q tests/test_sil_kullu_2021.py
make all
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

Expected source-local results: 3,168 manually reviewed cells; 2,753 installed
cells; 415 confirmed blanks; 2,963 installed rows after slash expansion; one
excluded layout/header record. The shared `make all`, full pytest, and browser
QA are deferred to the coordinating task as requested.
