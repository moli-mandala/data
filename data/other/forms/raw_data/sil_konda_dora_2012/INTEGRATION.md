# Shared integration proposal

This source-local task does not edit shared registries or generated CLDF. Apply
the following after the parallel source packages have landed.

## Bibliography

Add to `cldf/sources.bib`:

```bibtex
@techreport{blair-george2012kondadora,
  author = {Blair, Frank and George, Jacob},
  title = {Multilingualism Among the Konda Dora},
  year = {2012},
  number = {2012-016},
  institution = {SIL International},
  address = {Dallas},
  series = {SIL Electronic Survey Reports},
  url = {https://www.sil.org/resources/archives/49120},
  note = {Report created 1987; Appendix 9.5 image-scan wordlists manually transcribed and visually verified cell by cell; PDF SHA-256 6e0a3e5522a45752938f8279753d07b4e29d7b76ca73e88f71c4e283dfd0f533; no OCR-derived form is installed}
}
```

## Language and dialects

Reuse the existing language row:

```csv
Konda,Konda,kond1295,18.27,82.93,S. Dravidian II,,
```

Do not create new base languages for source labels “Koraput Konda,” “Visakh
Konda,” or AKA “Kubi.” Add these two site-level rows to `cldf/dialects.csv`.
The report gives localities but not defensible coordinates, so latitude and
longitude are blank and quality is C.

```csv
sil-konda-dora-1987-koraput,dialect:Konda:sil-konda-dora-1987-koraput:Koraput%20Konda%20%28Pansawalsa%29,Konda,K,Koraput Konda (Pansawalsa),kond1295,,,S. Dravidian II,"Pansawalsa, Potangi, Koraput District, Orissa; speaker Panjidasu, male, 40; recorded February 1987 by J. George; source code K",C
sil-konda-dora-1987-visakh,dialect:Konda:sil-konda-dora-1987-visakh:Visakh%20Konda%20%28Lakshmipuram%29,Konda,V,Visakh Konda (Lakshmipuram),kond1295,,,S. Dravidian II,"Lakshmipuram, Paderu, Visakh District, Andhra Pradesh; speaker Devadas, male, 35; recorded January 1987 by J. George; source code V",C
```

Telugu and Adivasi Oriya/Kotia Oriya are controls and require no dialect rows
for this source.

## Profile and routing

- Add `sil-konda-dora` to the profile-name allowlist in `make_cldf.py`.
- Route citation key `blair-george2012kondadora` to
  `conversion/sil-konda-dora.txt` in `make_cldf.py` and any parallel path in
  `utils.py`.
- Preserve exact source transcription as `Original`; conversion maps source
  colon to `ː`, source `?` to `ʔ`, source `ɽ` to Jambu `ṛ`, `dz` to Jambu
  `j`, and standalone `j` to Jambu `y`.
- Register the audit and manifest in the shared source-audit registry.
- Do not turn similarity-group digits into cognate sets or graph edges.

## Commands and expected results

```sh
python data/other/forms/raw_data/sil_konda_dora_2012/import_konda_dora.py --install
UV_CACHE_DIR=/tmp/uv-cache uv run --with pytest --with segments --with pypdf \
  pytest -q tests/test_sil_konda_dora_2012.py
make all
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

Expected source-local results: 214 prompts, four lists, 856 manually reviewed
cells, 727 attested cells, 129 confirmed blanks (43 target, 86 control), 342
excluded attested control cells, 385 attested target cells, 452 installed forms
after source-defined expansion (231 Koraput, 221 Visakh), and 856 audit rows.
There are zero ambiguous, clipped, illegible, or unresolved readings. Full
build, full suite, generated CLDF, and browser QA remain deferred.
