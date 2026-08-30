# Shared integration proposal

Shared bibliography, dialect registry, profile routing, audit-registry, census,
and focused metadata integration were applied by the coordinating task on
2026-08-28. The consolidated CLDF build and browser QA remain deferred until
all parallel source packages have landed. The sections below retain the source-
local proposal and expected counts as an audit trail.

## Bibliography

Add to `cldf/sources.bib`:

```bibtex
@techreport{devagnanavaram-et-al2021koya,
  author = {Devagnanavaram, D. and Chitrarasu, K. and Kew, Jonathan and Marshall, David and Prabhakar, D. and Rensch, Cal and Stahl, James},
  title = {A Sociolinguistic Survey of Koya Dialects},
  year = {2021},
  number = {2021-029},
  institution = {SIL International},
  address = {Dallas},
  series = {Journal of Language Survey Reports},
  url = {https://www.sil.org/resources/archives/88873},
  note = {Survey fieldwork conducted in 1985--1986; PDF SHA-256 a6541e0d2397849ce7c36961b3849f3b2c1f1c267036cfa1a3f6025796e14e7d}
}
```

## Base language

No new `cldf/languages.csv` row is proposed. All seven target lects attach to
the existing canonical `Gondi` row. This follows the report's own treatment of
Koya as the southernmost Gondi variety and current Jambu practice (`koya` is
already registered under `Gondi`).

## Dialects

Add these exact rows to `cldf/dialects.csv`. Coordinates and quality are left
blank because the report names survey lects/localities but does not supply
precise historical wordlist coordinates.

```csv
sil-koya-1985-jaganathapuram,dialect:Gondi:sil-koya-1985-jaganathapuram:Jaganathapuram%20Koya,Gondi,sil-koya-1985-jaganathapuram,Jaganathapuram Koya,,,,S. Dravidian II,"Jaganathapuram, eastern Koya survey lect; 1985 SIL survey", 
sil-koya-1985-chintoor,dialect:Gondi:sil-koya-1985-chintoor:Chintoor%20Koya,Gondi,sil-koya-1985-chintoor,Chintoor Koya,,,,S. Dravidian II,"Chintoor, eastern Koya survey lect; 1985 SIL survey", 
sil-koya-1985-podia,dialect:Gondi:sil-koya-1985-podia:Podia%20Koya,Gondi,sil-koya-1985-podia,Podia Koya,,,,S. Dravidian II,"Podia, eastern Gotte Koya survey lect; 1985 SIL survey", 
sil-koya-1985-utnoor,dialect:Gondi:sil-koya-1985-utnoor:Utnoor%20Gondi,Gondi,sil-koya-1985-utnoor,Utnoor Gondi,,,,S. Dravidian II,"Utnoor, western Gondi/Koya survey lect; 1985 SIL survey", 
sil-koya-1985-bhamani-gondi,dialect:Gondi:sil-koya-1985-bhamani-gondi:Bhamani%20Gondi,Gondi,sil-koya-1985-bhamani-gondi,Bhamani Gondi,,,,S. Dravidian II,"Bhamani, western Gondi/Koya survey lect; 1985 SIL survey", 
sil-koya-1985-bhamani-madia,dialect:Gondi:sil-koya-1985-bhamani-madia:Bhamani%20Madia,Gondi,sil-koya-1985-bhamani-madia,Bhamani Madia,,,,S. Dravidian II,"Bhamani, western Madia/Gotte survey lect; 1985 SIL survey", 
sil-koya-1985-malakanagiri,dialect:Gondi:sil-koya-1985-malakanagiri:Malakanagiri%20Koya,Gondi,sil-koya-1985-malakanagiri,Malakanagiri Koya,,,,S. Dravidian II,"Malakanagiri, eastern Koya survey lect; 1985 SIL survey", 
```

## Profile and routing

- Add `conversion/sil-koya.txt` to the source-profile router for citation key
  `devagnanavaram-et-al2021koya`.
- The profile is a source-local extension of the reviewed Gondi survey profile;
  it covers every installed form, including ASCII `g` in the older eastern
  transcription, `gʰ`, `gː`, parentheses, half length, dentality, and glides.
- No changes to etymology/edge routing are required; all rows have blank
  `Parameter_ID`, `Cognateset`, and `Etymology`.

## Commands and expected counts

```sh
python data/other/forms/raw_data/sil_koya_2021/import_koya.py --install
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_sil_koya_2021.py
make all
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

Expected source-local results before shared build: 1,890 audit slots; 1,840
printed cells manually reviewed; 1,401 installed target cells; 1,438 installed
rows after slash expansion; 69 missing target slots; 420 excluded controls.
