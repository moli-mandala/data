# Shared integration proposal

This source-local task does not edit shared registries or generated CLDF. Apply
the exact proposal below only after all parallel source packages have landed.

## Bibliography

Add to `cldf/sources.bib`:

```bibtex
@techreport{abraham-daimary2021amrikarbi,
  author = {Abraham, Binny and Daimary, Pronay},
  title = {A Sociolinguistic Study of Amri Karbi [ajz] in Northeast India},
  year = {2021},
  number = {2021-050},
  institution = {SIL International},
  address = {Dallas},
  series = {Journal of Language Survey Reports},
  url = {https://www.sil.org/resources/archives/91601},
  note = {Appendix B.3 wordlists manually verified against the rendered canonical PDF; PDF SHA-256 cd121ad102e96b43bf68a1cc5b44f1559c764bc4ae8d71988c6b292a1896ccb1; no OCR-derived form is installed}
}
```

## Languages

Add to `cldf/languages.csv`:

```csv
amri_karbi,Amri Karbi,amri1238,26.02,91.48,Other,"Assam and Meghalaya, India",C
karbi,Karbi,karb1241,25.73,93.05,Other,"Assam and Arunachal Pradesh, India",C
```

The report's low Karbi/Amri lexical-similarity results and its explicit Amri
Karbi classification support separate base-language rows rather than one being
treated as a dialect of the other.

## Dialects

The report provides localities but not defensible collection coordinates, so
latitude and longitude are blank and quality is C. Add to `cldf/dialects.csv`:

```csv
sil-amri-karbi-2021-holanki,dialect:karbi:sil-amri-karbi-2021-holanki:Holanki,karbi,A,Holanki,karb1241,,,Other,"Holanki, Papumpare district, Arunachal Pradesh; source code A",C
sil-amri-karbi-2021-hajarongpi,dialect:karbi:sil-amri-karbi-2021-hajarongpi:Hajarongpi,karbi,H,Hajarongpi,karb1241,,,Other,"Hajarongpi, East Karbi Anglong district, Assam; source code H",C
sil-amri-karbi-2021-amguri-kamrup,dialect:amri_karbi:sil-amri-karbi-2021-amguri-kamrup:Amguri%20%28Kamrup%29,amri_karbi,K,Amguri (Kamrup),amri1238,,,Other,"Amguri, Kamrup district, Assam; source code K",C
sil-amri-karbi-2021-paboi-misamari,dialect:karbi:sil-amri-karbi-2021-paboi-misamari:Paboi%20Misamari,karbi,M,Paboi Misamari,karb1241,,,Other,"Paboi Misamari, Sonitpur district, Assam; source code M",C
sil-amri-karbi-2021-maina-kharong,dialect:amri_karbi:sil-amri-karbi-2021-maina-kharong:Maina%20Kharong,amri_karbi,P,Maina Kharong,amri1238,,,Other,"Maina Kharong, Kamrup district, Assam; source code P",C
sil-amri-karbi-2021-plasha,dialect:amri_karbi:sil-amri-karbi-2021-plasha:Plasha%20%28Rongjari%29,amri_karbi,S,Plasha (Rongjari),amri1238,,,Other,"Rongjari Plasha, Ri-Bhoi district, Meghalaya; source code S",C
sil-amri-karbi-2021-amguri-wka,dialect:karbi:sil-amri-karbi-2021-amguri-wka:Amguri%20%28West%20Karbi%20Anglong%29,karbi,a,Amguri (West Karbi Anglong),karb1241,,,Other,"Amguri, West Karbi Anglong district, Assam; source code a",C
sil-amri-karbi-2021-sermansingner,dialect:karbi:sil-amri-karbi-2021-sermansingner:Sermansingner,karbi,b,Sermansingner,karb1241,,,Other,"Sermansingner, East Karbi Anglong district, Assam; source code b",C
sil-amri-karbi-2021-langhemphi,dialect:karbi:sil-amri-karbi-2021-langhemphi:Langhemphi,karbi,c,Langhemphi,karb1241,,,Other,"Langhemphi, West Karbi Anglong district, Assam; source code c",C
sil-amri-karbi-2021-umrinti,dialect:karbi:sil-amri-karbi-2021-umrinti:Umrinti,karbi,d,Umrinti,karb1241,,,Other,"Umrinti, West Karbi Anglong district, Assam; source code d",C
sil-amri-karbi-2021-bankri,dialect:karbi:sil-amri-karbi-2021-bankri:Bankri%20%28Bhankri%29,karbi,h,Bankri (Bhankri),karb1241,,,Other,"Bhankri, West Karbi Anglong district, Assam; source code h",C
sil-amri-karbi-2021-rongtheang,dialect:karbi:sil-amri-karbi-2021-rongtheang:Rongtheang,karbi,k,Rongtheang,karb1241,,,Other,"Rongtheang, East Karbi Anglong district, Assam; source code k",C
sil-amri-karbi-2021-sunajoli,dialect:karbi:sil-amri-karbi-2021-sunajoli:Sunajoli,karbi,l,Sunajoli,karb1241,,,Other,"Sunajoli, Lakhimpur district, Assam; source code l",C
sil-amri-karbi-2021-mikirgaon,dialect:karbi:sil-amri-karbi-2021-mikirgaon:Mikirgaon,karbi,m,Mikirgaon,karb1241,,,Other,"Mikirgaon, Nagaon district, Assam; source code m",C
sil-amri-karbi-2021-sardoka-ingti,dialect:karbi:sil-amri-karbi-2021-sardoka-ingti:Sardoka%20Ingti,karbi,s,Sardoka Ingti,karb1241,,,Other,"Sardoka Ingti, East Karbi Anglong district, Assam; source code s",C
```

## Profile and routing

- Add `sil-amri-karbi` to the profile-name allowlist in `make_cldf.py`.
- Route citation key `abraham-daimary2021amrikarbi` to
  `conversion/sil-amri-karbi.txt` in `make_cldf.py` and any parallel route in
  `utils.py`.
- Preserve exact source IPA as `Original`; use the converted value only for
  Jambu display form.
- Register the audit/manifest in any shared source-audit registry.
- Do not create cognacy, etymology, borrowing, derivation, or variant edges
  from the report's similarity-group numbers.

## Commands and expected results

```sh
python data/other/forms/raw_data/sil_amri_karbi_2021/import_amri.py --install
UV_CACHE_DIR=/tmp/uv-cache uv run --with pytest --with segments pytest -q tests/test_sil_amri_karbi_2021.py
make all
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

Expected source-local results: 307 prompts, 17 printed lists, 5,219 manually
reviewed conceptual cells, 5,960 reviewed response occurrences, six confirmed
blank cells, 5,329 target response occurrences, 631 excluded control response
occurrences, 237 exact repeated target occurrences retained audit-only, 5,092
installed forms (993 Amri Karbi and 4,099 Karbi), and 5,966 audit rows. Four
reported but unpublished Amri lists are source-absent. There are zero unresolved
transcriptions and one faithfully retained source-marked uncertainty in an
excluded Assamese control. Full build, full suite, and browser QA are deferred.

