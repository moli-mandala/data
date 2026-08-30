# Shared integration proposal

Shared bibliography, Kodaku base-language registration, eighteen dialect rows,
profile routing, audit-registry, census, and focused metadata integration were
applied by the coordinating task on 2026-08-28. The consolidated CLDF build and
browser QA remain deferred until all parallel source packages have landed. The
sections below retain the source-local proposal and expected counts as an audit
trail.

## Bibliography

Add to `cldf/sources.bib`:

```bibtex
@techreport{behera2022korwakodaku,
  author = {Behera, Gangadhar},
  title = {A Sociolinguistic Profile of Korwa and Kodaku Tribes in Chhattisgarh and Jharkhand, India},
  year = {2022},
  number = {2022-014},
  institution = {SIL International},
  address = {Dallas},
  series = {Journal of Language Survey Reports},
  url = {https://www.sil.org/resources/publications/entry/94564},
  note = {Survey fieldwork conducted in 2004--2005; PDF SHA-256 a8efbe88405e27024a7a6ec786cd6fde3e382f0eaf0d0081197d3880ed97eb0c; all installed forms visually reviewed; OCR retained as comparison only}
}
```

## Base language

Korwa target rows attach to the existing `kw` base row. Add this exact row to
`cldf/languages.csv` for the report's explicitly distinct Kodaku language (ISO
639-3 `ksz`):

```csv
Kodaku,Kodaku,koda1256,,,Munda,"Chhattisgarh, Jharkhand, and Uttar Pradesh; ISO 639-3 ksz",C
```

## Dialects

Add to `cldf/dialects.csv`:

```csv
sil-korwa-2004-chilma,dialect:kw:sil-korwa-2004-chilma:Chilma%20Korwa,kw,sil-korwa-2004-chilma,Chilma Korwa,,,,Munda,"Chilma, Rajpur, Surguja, Chhattisgarh; source code C",C
sil-korwa-2004-dhaneshpur,dialect:kw:sil-korwa-2004-dhaneshpur:Dhaneshpur%20Korwa,kw,sil-korwa-2004-dhaneshpur,Dhaneshpur Korwa,,,,Munda,"Dhaneshpur, Kusmi, Surguja, Chhattisgarh; source code D",C
sil-korwa-2005-gaseband,dialect:kw:sil-korwa-2005-gaseband:Gaseband%20Korwa,kw,sil-korwa-2005-gaseband,Gaseband Korwa,,,,Munda,"Gaseband, Bagicha, Jashpur, Chhattisgarh; source code G",C
sil-korwa-2004-harrapat,dialect:kw:sil-korwa-2004-harrapat:Harrapat%20Korwa,kw,sil-korwa-2004-harrapat,Harrapat Korwa,,,,Munda,"Harrapat, Manora, Jashpur, Chhattisgarh; source code H",C
sil-korwa-2004-bladerpat,dialect:kw:sil-korwa-2004-bladerpat:Bladerpat%20Korwa,kw,sil-korwa-2004-bladerpat,Bladerpat Korwa,,,,Munda,"Bladerpat, Sanna, Jashpur, Chhattisgarh; source code K",C
sil-korwa-2004-kirkima,dialect:kw:sil-korwa-2004-kirkima:Kirkima%20Korwa,kw,sil-korwa-2004-kirkima,Kirkima Korwa,,,,Munda,"Kirkima, Lundra, Surguja, Chhattisgarh; source code L",C
sil-korwa-2004-musakhoel,dialect:kw:sil-korwa-2004-musakhoel:Musakhoel%20Korwa,kw,sil-korwa-2004-musakhoel,Musakhoel Korwa,,,,Munda,"Musakhoel, Govindpur, Surguja, Chhattisgarh; source code M",C
sil-korwa-2004-rakkaya,dialect:kw:sil-korwa-2004-rakkaya:Rakkaya%20Korwa,kw,sil-korwa-2004-rakkaya,Rakkaya Korwa,,,,Munda,"Rakkaya, Sankargarh, Surguja, Chhattisgarh; source code R",C
sil-korwa-2005-sardih,dialect:kw:sil-korwa-2005-sardih:Sardih%20Korwa,kw,sil-korwa-2005-sardih,Sardih Korwa,,,,Munda,"Sardih, Korba, Chhattisgarh; source code Z",C
sil-kodaku-2004-sagardinwa,dialect:Kodaku:sil-kodaku-2004-sagardinwa:Sagardinwa%20Kodaku,Kodaku,sil-kodaku-2004-sagardinwa,Sagardinwa Kodaku,,,,Munda,"Sagardinwa, Chainpur, Palamau, Jharkhand; source code S",C
sil-kodaku-2005-jamuniatanr,dialect:Kodaku:sil-kodaku-2005-jamuniatanr:Jamuniatanr%20Kodaku,Kodaku,sil-kodaku-2005-jamuniatanr,Jamuniatanr Kodaku,,,,Munda,"Jamuniatanr, Ranka, Jharkhand; source code b",C
sil-kodaku-2005-chainpur,dialect:Kodaku:sil-kodaku-2005-chainpur:Chainpur%20Kodaku,Kodaku,sil-kodaku-2005-chainpur,Chainpur Kodaku,,,,Munda,"Chainpur, Bhavni, Uttar Pradesh; source code c",C
sil-kodaku-2005-dhengura,dialect:Kodaku:sil-kodaku-2005-dhengura:Dhengura%20Kodaku,Kodaku,sil-kodaku-2005-dhengura,Dhengura Kodaku,,,,Munda,"Dhengura, Ranka, Garhwa, Jharkhand; source code d",C
sil-kodaku-2005-jhaleria,dialect:Kodaku:sil-kodaku-2005-jhaleria:Jhaleria%20Kodaku,Kodaku,sil-kodaku-2005-jhaleria,Jhaleria Kodaku,,,,Munda,"Jhaleria, Balrampur, Chhattisgarh; source code j",C
sil-kodaku-2005-chilma,dialect:Kodaku:sil-kodaku-2005-chilma:Chilma%20Kodaku,Kodaku,sil-kodaku-2005-chilma,Chilma Kodaku,,,,Munda,"Chilma, Balrampur, Chhattisgarh; source code m",C
sil-kodaku-2005-kodakupara,dialect:Kodaku:sil-kodaku-2005-kodakupara:Kodakupara%20Kodaku,Kodaku,sil-kodaku-2005-kodakupara,Kodakupara Kodaku,,,,Munda,"Kodakupara, Pratapur, Chhattisgarh; source code p",C
sil-kodaku-2005-tharki,dialect:Kodaku:sil-kodaku-2005-tharki:Tharki%20Kodaku,Kodaku,sil-kodaku-2005-tharki,Tharki Kodaku,,,,Munda,"Tharki, Rajpur, Surguja, Chhattisgarh; source code t",C
sil-kodaku-2005-baikanthpur,dialect:Kodaku:sil-kodaku-2005-baikanthpur:Baikanthpur%20Kodaku,Kodaku,sil-kodaku-2005-baikanthpur,Baikanthpur Kodaku,,,,Munda,"Baikanthpur, Wadrafnagar, Chhattisgarh; source code w",C
```

## Profile and routing

- Add `sil-korwa-kodaku` to the profile-name collection in `make_cldf.py`.
- Route source key `behera2022korwakodaku` to
  `conversion/sil-korwa-kodaku.txt` and set `row_convert = True`, immediately
  alongside the other SIL survey routes.
- No `utils.py` graph or etymology routing is needed. Similarity groups remain
  Notes only and all `Parameter_ID`, `Cognateset`, and `Etymology` fields are blank.

## Commands and expected counts

```sh
python data/other/forms/raw_data/sil_korwa_kodaku_2022/import_korwa_kodaku.py --verify-pdf --install
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_sil_korwa_kodaku_2022.py
make all
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

Expected source-local results: 2,900 reviewed printed response lines; 5,250
audited conceptual cells; 3,730 attested target cells; 50 blank/unlisted target
cells; 4,458 installed target rows; 1,453 excluded attested control cells; 17
blank/unlisted control cells; two excluded unidentified source-code assignments.
