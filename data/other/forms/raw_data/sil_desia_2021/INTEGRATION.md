# Shared integration proposal

This source-local package deliberately does not edit shared registries or build
outputs. The coordinating task should apply the following exact changes once
parallel survey packages have landed.

## Bibliography

Add to `cldf/sources.bib`:

```bibtex
@techreport{behera2021desia,
  author = {Behera, Gangadhar},
  title = {A Sociolinguistic Survey among Desia-Speaking People Groups in South Orissa, India},
  year = {2021},
  number = {2021-056},
  institution = {SIL International},
  address = {Dallas},
  series = {Journal of Language Survey Reports},
  url = {https://www.sil.org/resources/publications/entry/91960},
  included = {Appendix B.5, printed pages 71--118: all 210 prompts across nineteen Desia wordlists},
  provenance = {data/other/forms/20260828-sil-desia.csv; data/other/forms/raw_data/20260828-sil-desia-audit.csv},
  jambu_editor = {Aryaman Arora and OpenAI Codex},
  note = {Survey created in 2007; source record states no distinct reuse licence; PDF not redistributed; PDF SHA-256 04de0004c1375955c1adbeb8941b187aa4fc88f484ee00e9bc69655813e6690b; all 4,696 printed response lines visually checked; no ambiguous, clipped, illegible, OCR-derived, or unreviewed form installed}
}
```

## Base language

No new language row is required. Attach all forms and dialects to the existing
`AdivasiOriya` row (`adiv1239`, Kotia-Adivasi Oriya-Desiya). The report's ISO
639-3 code `dso` is retired/spurious in current Glottolog and should not create
a second base language.

## Dialects

Add to `cldf/dialects.csv`. The report supplies no coordinates, so latitude and
longitude remain blank. Location notes retain conflicting B.4/Table 1 metadata.

```csv
sil-desia-2007-potenda-rona,dialect:AdivasiOriya:sil-desia-2007-potenda-rona:Potenda%20Rona%20Desia,AdivasiOriya,Potenda,Potenda Rona Desia,,,,Eastern,"Potenda; Lamtaput; Koraput; Rona community",C
sil-desia-2007-ghumar-rona,dialect:AdivasiOriya:sil-desia-2007-ghumar-rona:Ghumar%20Rona%20Desia,AdivasiOriya,Ghumar,Ghumar Rona Desia,,,,Eastern,"Ghumar; Kotpad; Koraput; Rona community",C
sil-desia-2007-sabhapatiguda-gaud,dialect:AdivasiOriya:sil-desia-2007-sabhapatiguda-gaud:Sabhapatiguda%20Gaud%20Desia,AdivasiOriya,Sabhapatiguda,Sabhapatiguda Gaud Desia,,,,Eastern,"Sabhapatiguda; Malkangiri; Malkangiri; Gaud community",C
sil-desia-2007-kantigad-gaud,dialect:AdivasiOriya:sil-desia-2007-kantigad-gaud:Kantigad%20Gaud%20Desia,AdivasiOriya,Kantigad,Kantigad Gaud Desia,,,,Eastern,"Kantigad; Lamtaput in Table 1 but Boriguma in B.4; Koraput; Gaud community",C
sil-desia-2007-kakalpoda-bod-mali,dialect:AdivasiOriya:sil-desia-2007-kakalpoda-bod-mali:Kakalpoda%20Bod%20Mali%20Desia,AdivasiOriya,Kakalpoda,Kakalpoda Bod Mali Desia,,,,Eastern,"Kakalpoda; Lamtaput in Table 1 but Boriguma in B.4; Koraput; Bod Mali community",C
sil-desia-2007-konda-maliguda-bod-mali,dialect:AdivasiOriya:sil-desia-2007-konda-maliguda-bod-mali:Konda%20Maliguda%20Bod%20Mali%20Desia,AdivasiOriya,Konda Maliguda,Konda Maliguda Bod Mali Desia,,,,Eastern,"Konda Maliguda; Laxmipur; Koraput; Bod Mali community",C
sil-desia-2007-patta-maliguda-san-mali,dialect:AdivasiOriya:sil-desia-2007-patta-maliguda-san-mali:Patta%20Maliguda%20San%20Mali%20Desia,AdivasiOriya,Patta Maliguda,Patta Maliguda San Mali Desia,,,,Eastern,"Patta Maliguda; Laxmipur; Koraput; San Mali community",C
sil-desia-2007-gumalput-gadaba,dialect:AdivasiOriya:sil-desia-2007-gumalput-gadaba:Gumalput%20Gadaba%20Desia,AdivasiOriya,Gumalput,Gumalput Gadaba Desia,,,,Eastern,"Gumalput; Lamtaput in Table 1 but Boriguma in B.4; Koraput; Gadaba community",C
sil-desia-2007-gagnapur-poroja,dialect:AdivasiOriya:sil-desia-2007-gagnapur-poroja:Gagnapur%20Poroja%20Desia,AdivasiOriya,Gagnapur,Gagnapur Poroja Desia,,,,Eastern,"Gagnapur; Jeypore; Koraput; Poroja community",C
sil-desia-2007-dame-side-dom,dialect:AdivasiOriya:sil-desia-2007-dame-side-dom:Dame%20side%20Dom%20Desia,AdivasiOriya,Dame side,Dame side Dom Desia,,,,Eastern,"B.5 says Dame side; B.4 says Dom Sahi; Table 1 says Bodgaon; Potangi/Patangi; Koraput; Dom community",C
sil-desia-2007-burja-dom,dialect:AdivasiOriya:sil-desia-2007-burja-dom:Burja%20Dom%20Desia,AdivasiOriya,Burja,Burja Dom Desia,,,,Eastern,"Burja; Laxmipur; Koraput; Dom community",C
sil-desia-2007-chhatrabor-harijan,dialect:AdivasiOriya:sil-desia-2007-chhatrabor-harijan:Chhatrabor%20Harijan%20Desia,AdivasiOriya,Chhatrabor,Chhatrabor Harijan Desia,,,,Eastern,"Chhatrabor; Papdahandi; Nabrangpur; Harijan community",C
sil-desia-2007-bodgaon-dhulia,dialect:AdivasiOriya:sil-desia-2007-bodgaon-dhulia:Bodgaon%20Dhulia%20Desia,AdivasiOriya,Bodgaon,Bodgaon Dhulia Desia,,,,Eastern,"Bodgaon; Potangi/Patangi; Koraput; Dhulia community",C
sil-desia-2007-gemelput-mania,dialect:AdivasiOriya:sil-desia-2007-gemelput-mania:Gemelput%20Mania%20Desia,AdivasiOriya,Gemelput,Gemelput Mania Desia,,,,Eastern,"Gemelput; Padua/Nandapur in Table 1 and Nandput in B.4; Koraput; Mania community",C
sil-desia-2007-sindhiguda-bonda,dialect:AdivasiOriya:sil-desia-2007-sindhiguda-bonda:Sindhiguda%20Bonda%20Desia,AdivasiOriya,Sindhiguda,Sindhiguda Bonda Desia,,,,Eastern,"B.5/Table 1 say Sindhiguda but B.4 says Rasbeda; Khairput/Khaiput; Malkangiri; Bonda community",C
sil-desia-2007-souraguda-soura,dialect:AdivasiOriya:sil-desia-2007-souraguda-soura:Souraguda%20Soura%20Desia,AdivasiOriya,Souraguda,Souraguda Soura Desia,,,,Eastern,"Souraguda; Jeypore; Koraput; Soura community",C
sil-desia-2007-aunli-bhotra,dialect:AdivasiOriya:sil-desia-2007-aunli-bhotra:Aunli%20Bhotra%20Desia,AdivasiOriya,Aunli,Aunli Bhotra Desia,,,,Eastern,"Aunli; Boriguma; Koraput; Bhotra community",C
sil-desia-2007-sourakundi-bhotra,dialect:AdivasiOriya:sil-desia-2007-sourakundi-bhotra:Sourakundi%20Bhotra%20Desia,AdivasiOriya,Sourakundi,Sourakundi Bhotra Desia,,,,Eastern,"Sourakundi; Kotpad; Koraput; Bhotra community",C
sil-desia-2007-jujhari-kamar,dialect:AdivasiOriya:sil-desia-2007-jujhari-kamar:Jujhari%20Kamar%20Desia,AdivasiOriya,Jujhari,Jujhari Kamar Desia,,,,Eastern,"Jujhari; Boriguma; Koraput; Kamar community",C
```

## Profile and routing

- Add `sil-desia` to the profile-name collection in `make_cldf.py`.
- Route source key `behera2021desia` to `conversion/sil-desia.txt` and set
  `row_convert = True`, alongside the other SIL survey routes.
- No `utils.py` graph or etymology routing is needed. Similarity groups remain
  Notes only and all `Parameter_ID`, `Cognateset`, and `Etymology` fields remain blank.

## Commands and expected counts

```sh
python3 data/other/forms/raw_data/sil_desia_2021/import_desia.py --verify-pdf --install
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_sil_desia_2021.py
make all
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

Expected source-local result: 4,696 reviewed printed response lines; 3,990
audited conceptual target cells; 4,658 attested source response lines; 38
explicit blank cells; 4,655 installed forms; zero controls; zero unresolved,
ambiguous, clipped, or illegible readings.
