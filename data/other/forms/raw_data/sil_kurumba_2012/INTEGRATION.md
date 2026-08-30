# Shared source-specific integration

The 10,450-cell manual review is complete. Shared source-specific installation
is now applied; the consolidated CLDF/full build, opaque identity reconciliation,
graph/full-suite validation, browser refresh/QA, and commit remain deferred.

## Bibliography

Add to `cldf/sources.bib`:

```bibtex
@techreport{blairetal2012kurumba,
  author = {Blair, Frank and Chitrarasu, K. and Prabhu, R. and Rajah, B. B. and Rajaiah, J. and Rensch, Cal and Rensch, Carolyn},
  title = {A Sociolinguistic Profile of Kurumba Dialects},
  year = {2012},
  number = {2012-015},
  institution = {SIL International},
  address = {Dallas, Texas},
  series = {SIL Electronic Survey Reports},
  pages = {ii+431},
  url = {https://www.sil.org/resources/archives/50805},
  note = {Fieldwork conducted in 1984--1985; report written in 1986--1987; official PDF SHA-256 250dc3d83661227caa66bf16e390e51c2dcb7186fa435252541ed13bbfcd9137},
  included = {Appendix C, printed pages 212--431: nineteen 550-item phonetic wordlists, 10,450 conceptual cells, including fifteen target lists and four explicitly marked comparison controls},
  provenance = {Official SIL PDF acquired 2026-08-28. Source-local renderer, corrupt OCR comparison scaffold, immutable manual cell ledger, importer, audit, and manifest are under data/other/forms/raw_data/sil_kurumba_2012},
  ocr = {Embedded Paper Capture OCR is locator-only. Every installed form must be manually transcribed and visually verified from the rendered scan; no OCR-only reading is accepted},
  etymology_provenance = {none},
  jambu_editor = {OpenAI Codex}
}
```

## Base languages and report classification

No new `cldf/languages.csv` row is proposed. The nineteen stable wordlist
varieties attach to existing canonical rows: thirteen SNSK/NSK/JNSK lists and
the Standard Kannada/Vakkaliga comparanda to `Kannada`; Kotagiri Alu to
`AluKurumba`; Maddur Betta to `BettaKurumba`; and the remaining controls to
`Tamil` and `Badaga`. This preserves the report's explicit analysis: SNSK is
Southern Nonstandard Kannada, ANSK is Alu Kurumba Nonstandard Kannada, JNSK is
treated as colloquial Kannada, NST is Betta Kurumba Nonstandard Tamil, and the
Chitradurga list is CNSK/Nonstandard Kannada.

## Dialects

The following nineteen exact rows are installed in `cldf/dialects.csv`. Source-local
identifiers are retained in `Source_Language_ID`. Coordinates are blank because
the report does not provide exact historical collection points; locality text
and quality `C` preserve that limitation.

```csv
sil-kurumba-1985-tamil-madras,dialect:Tamil:sil-kurumba-1985-tamil-madras:Standard%20Tamil%2C%20Madras,Tamil,tamil_madras,"Standard Tamil, Madras",,,,S. Dravidian I,"Madras variety; elicited 25 January 1985; ESR 2012-015 control list",C
sil-kurumba-1985-kannada-bangalore,dialect:Kannada:sil-kurumba-1985-kannada-bangalore:Standard%20Kannada%2C%20Bangalore,Kannada,kannada_bangalore,"Standard Kannada, Bangalore",,,,S. Dravidian I,"Bangalore variety; elicited 22 February 1985; ESR 2012-015 control list",C
sil-kurumba-1984-belavarthy,dialect:Kannada:sil-kurumba-1984-belavarthy:SNSK%2C%20Belavarthy%20Kurumba,Kannada,belavarthy,"SNSK, Belavarthy Kurumba",,,,S. Dravidian I,"Belavarthy; Krishnagiri taluk; Dharmapuri district; Tamil Nadu",C
sil-kurumba-1976-pudukkottai,dialect:Kannada:sil-kurumba-1976-pudukkottai:SNSK%2C%20Pudukkottai%20Kurumba,Kannada,pudukkottai,"SNSK, Pudukkottai Kurumba",,,,S. Dravidian I,"Pudukkottai district; Tamil Nadu; list dated 1976-1977",C
sil-kurumba-1985-kotagiri-alu,dialect:AluKurumba:sil-kurumba-1985-kotagiri-alu:ANSK%2C%20Kotagiri%20Alu%20Kurumba,AluKurumba,kotagiri_alu,"ANSK, Kotagiri Alu Kurumba",,,,S. Dravidian I,"Banigudisole village; Kotagiri taluk; Nilgiris district; Tamil Nadu",C
sil-kurumba-1985-badaga-arvenu,dialect:Badaga:sil-kurumba-1985-badaga-arvenu:Badaga%2C%20Arvenu%2FKotagiri,Badaga,badaga_arvenu,"Badaga, Arvenu/Kotagiri",,,,S. Dravidian I,"Arvenu; Kotagiri taluk; Nilgiris district; Tamil Nadu; control list",C
sil-kurumba-1985-kolar,dialect:Kannada:sil-kurumba-1985-kolar:SNSK%2C%20Kolar%20Kurubas,Kannada,kolar_kuruba,"SNSK, Kolar Kurubas",,,,S. Dravidian I,"Basavanatha village; Kolar district; Karnataka",C
sil-kurumba-1985-chitradurga,dialect:Kannada:sil-kurumba-1985-chitradurga:NSK%2C%20Chitradurga%20Kurubas,Kannada,chitradurga_kuruba,"NSK, Chitradurga Kurubas",,,,S. Dravidian I,"Malappanahatti village; Chitradurga district; Karnataka",C
sil-kurumba-1984-buringi,dialect:Kannada:sil-kurumba-1984-buringi:SNSK%2C%20Buringi%20Kurumba,Kannada,buringi,"SNSK, Buringi Kurumba",,,,S. Dravidian I,"Buringi village; Tiruppattur taluk; North Arcot district; Tamil Nadu",C
sil-kurumba-1984-madapalli,dialect:Kannada:sil-kurumba-1984-madapalli:SNSK%2C%20Madapalli%20Kurumba,Kannada,madapalli,"SNSK, Madapalli Kurumba",,,,S. Dravidian I,"Madapalli village; Tiruppattur taluk; North Arcot district; Tamil Nadu",C
sil-kurumba-1984-kurumbatheru,dialect:Kannada:sil-kurumba-1984-kurumbatheru:SNSK%2C%20Kurumbatheru%20Kannada,Kannada,kurumbatheru,"SNSK, Kurumbatheru Kannada",,,,S. Dravidian I,"Kurumbatheru hamlet; Kandikuppam village; Krishnagiri taluk; Dharmapuri district; Tamil Nadu",C
sil-kurumba-1984-thangiyadikuppam,dialect:Kannada:sil-kurumba-1984-thangiyadikuppam:SNSK%2C%20Thangiyadikuppam%20Kurumba,Kannada,thangiyadikuppam,"SNSK, Thangiyadikuppam Kurumba",,,,S. Dravidian I,"Thangiyadikuppam village; Kuppam taluk; Chittoor district; Andhra Pradesh",C
sil-kurumba-1985-beerajjanur,dialect:Kannada:sil-kurumba-1985-beerajjanur:SNSK%2C%20Beerajjanur%20Kurumba,Kannada,beerajjanur,"SNSK, Beerajjanur Kurumba",,,,S. Dravidian I,"Beerajjanur village; Krishnagiri taluk; Dharmapuri district; Tamil Nadu",C
sil-kurumba-1985-karmadai-kurumba,dialect:Kannada:sil-kurumba-1985-karmadai-kurumba:SNSK%2C%20Karmadai%20Kurumba,Kannada,karmadai_kurumba,"SNSK, Karmadai Kurumba",,,,S. Dravidian I,"Karamadai; Mettupalayam taluk; Coimbatore district; Tamil Nadu",C
sil-kurumba-1985-karmadai-vakkaliga,dialect:Kannada:sil-kurumba-1985-karmadai-vakkaliga:SNSK%2C%20Karmadai%20Vakkaliga,Kannada,karmadai_vakkaliga,"SNSK, Karmadai Vakkaliga",,,,S. Dravidian I,"Karmadai; Mettupalayam taluk; Coimbatore district; Tamil Nadu; control list",C
sil-kurumba-1985-kurumbapalayam,dialect:Kannada:sil-kurumba-1985-kurumbapalayam:SNSK%2C%20Kurumbapalayam%20Kurumba,Kannada,kurumbapalayam,"SNSK, Kurumbapalayam Kurumba",,,,S. Dravidian I,"Kurumbapalayam village; Coimbatore district; Tamil Nadu",C
sil-kurumba-1985-kalangal,dialect:Kannada:sil-kurumba-1985-kalangal:SNSK%2C%20Kalangal%20Kurumba,Kannada,kalangal,"SNSK, Kalangal Kurumba",,,,S. Dravidian I,"Kalangal village; Palladam taluk; Coimbatore district; Tamil Nadu",C
sil-kurumba-1985-masinagudi-jennu,dialect:Kannada:sil-kurumba-1985-masinagudi-jennu:JNSK%2C%20Masinagudi%20Jennu%20Kurumba,Kannada,masinagudi_jennu,"JNSK, Masinagudi Jennu Kurumba",,,,S. Dravidian I,"Masinagudi village; Gudalur taluk; Nilgiris district; Tamil Nadu",C
sil-kurumba-1985-maddur-betta,dialect:BettaKurumba:sil-kurumba-1985-maddur-betta:NST%2C%20Maddur%20Colony%20Betta%20Kurumba,BettaKurumba,maddur_betta,"NST, Maddur Colony Betta Kurumba",,,,S. Dravidian I,"Maddur Colony; Gundlupet taluk; Mysore district; Karnataka",C
```

## Installed file, sound profile, and routing

Do not copy all of the frozen source-local staged output: its 4,738 rows include
1,534 attestations from the four explicitly marked comparison-control lists.
`install_target_forms.py` proves the scope of every frozen staged row against
the 10,450-row audit and emits `installed_target_forms.csv`; only its 3,204
`Scope=target` attestations are copied, without changing relative row order, to
`data/other/forms/20260828-sil-kurumba.csv`. All controls, 5,710 printed dashes,
the ambiguous Pudukkottai cell, and the illegible Kotagiri Alu cell remain in
`shared_integration_audit.csv` with exact coordinates and exclusion reasons.
The frozen 4,738-row `staged_forms.csv` remains byte-unchanged for reproducibility.

`conversion/sil-kurumba-2012.txt` is built from the inventory of manually reviewed
`Manual_Form` symbols; its `IPA` column must cover every attested source form.
Do not derive mappings from the OCR scaffold. Preserve source IPA in raw `Form`
and `Phonemic`; compiled `Original`/`Phonemic` retain it while compiled `Form`
receives the reviewed house conversion.

`make_cldf.py` routes the immutable citation key `blairetal2012kurumba` to
`sil-kurumba-2012`, and `tests/test_sound_profiles.py` registers:

```python
"sil-kurumba-2012": ["20260828-sil-kurumba.csv"],
```

Representative assertions cover difficult source symbols and every installed
form is exercised against the profile. No etymology/edge routing is needed:
similarity groups are explicitly
non-etymological and all `Parameter_ID`, `Cognateset`, and `Etymology` fields
remain blank.

## Deferred commands and completion counts

```sh
python3 data/other/forms/raw_data/sil_kurumba_2012/import_kurumba.py --verify-pdf --stage
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_sil_kurumba_2012.py
make all
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

Final invariant counts are 19 lists, 550 prompts, 220 data pages, 10,450 cells,
8,250 target cells, and 2,200 comparison cells. The frozen manual audit contains
4,738 attestations and 5,710 printed-dash blanks, plus one ambiguous and one
illegible target cell. Shared installation contains exactly 3,204 target
attestations; 1,534 control attestations and every non-attested cell are audit-only.
