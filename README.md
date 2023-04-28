# moli-mandala/data

![Status](https://github.com/moli-mandala/data/actions/workflows/python-app.yml/badge.svg)

This is a CLDF database for the Jambu application, containing historical linguistic data for many languages of South Asia. It also contains the underlying raw data and scripts used to produce/update the CLDF database.

## How it works

Before doing things, install dependencies in a fresh environment with `pip install -r requirements.txt` (with Python 3.9.12).

To recreate the CLDF database from raw data, just run `make parse` in root. To verify the output is valid CLDF, run `python -m pytest` in root folder.

### CLDF structure

The final CLDF database is in `cldf/`. It includes the following:

- `forms.csv`: Lemmata.
- `languages.csv`: List of languages, metadata (coordinates, Glottolog ID) for each.
- `parameters.csv`: Entries, including headwords and etymological notes.
- `sources.bib`: References in BibTeX format.

The structure is more formally defined in `cldf/Wordlist-metadata.json`.

### Raw data

**Raw data is organised under `data/`**. The script `make_cldf.py` builds the CLDF database in `cldf/` from the raw data. Raw data is all stored in CSV in order to be easy to edit and parse.

For raw data files that list lemmata, the columns are:
1. Language ID
2. Param ID (entry)
3. Lemma (normalised)
4. Gloss
5. Native script form
6. Phonemic form (in IPA)
7. Notes/comments
8. References

For raw data files that list parameters (entries), the columns are:
1. Param ID
2. Language of the headword (e.g. "Indo-Aryan", "Proto-Dravidian")
3. Form
4. Gloss
5. References

Etymological notes for *all* params (as written up/collated by us) are stored in `data/etymologies.csv`. The columns are just the Param ID and Markdown-formatted notes.

Finally, some sources have unusual orthographies that we need to convert to the Sāmapriya-n system. The profiles used by the `segments` library to do so are stored as `conversions/*.txt`; these give substitution rules for orthographic normalisation.

#### DEDR

The DEDR and related parsing scripts are all in `data/dedr/`. Originally, Suresh supplied a SQL database scraped from the online version (`data/dedr/dedr_new_entry_oct2013_edited.sql`) which was converted into a CSV at (`data/dedr/dedr.csv`). **These are now deprecated**.

The current CSV format of the DEDR is generated using `data/dedr/parse.py`, which scrapes the website and caches it in `data/dedr/dedr.pickle`, and then divides the entry into language spans (e.g. *Tam. word 'meaning', word2 'meaning2'...* is one span) and parses each span into forms and associated references and glosses using complicated regexes. The output is at `data/dedr/dedr_new.csv`.

The helper file `data/dedr/abbrevs.py` includes information about what each language tag and reference abbreviation corresponds to in the CLDF (e.g. Mal. = Malayalam).

Headwords for entries are to be stored in `data/dedr/params.csv`. Finally, Proto-Dravidian reconstructions are housed in `data/dedr/pdr.csv`.

#### CDIAL

The CDIAL directory is `data/cdial/` and is basically identically structured to the DEDR directory. Cache at `data/cdial/cdial.pickle`, parse script is `data/cdial/parse.py`, helper info in `data/cdial/abbrevs.py`, and params are at `data/cdial/params.csv`.

#### DBIA

Under development at `data/dbia/`.

#### Munda
- `data/munda/forms.csv`
- `data/munda/params.csv`

#### Others

- `data/others/forms/*.csv`: New lemmata extracted from various sources.
- `data/others/params/*.csv`: New entries.