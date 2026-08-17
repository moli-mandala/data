# moli-mandala/data

![Status](https://github.com/moli-mandala/data/actions/workflows/python-app.yml/badge.svg)

This is a CLDF database for the Jambu application, containing historical linguistic data for many languages of South Asia. It also contains the underlying raw data and scripts used to produce/update the CLDF database.

## How it works

Before doing things, install dependencies in a fresh environment with `pip install -r requirements.txt` (with Python 3.9.12).

To recreate the CLDF database from raw data, just run `make parse` in root. To verify the output is valid CLDF, run `python -m pytest` in root folder.

### CLDF structure

The final CLDF database is in `cldf/`. It includes the following:

- `forms.csv`: Lemmata.
- `languages.csv`: Base languages and their metadata (coordinates, Glottolog ID).
- `dialects.csv`: Dialect-tag metadata and the source lect IDs normalized to each base language.
- `parameters.csv`: Entries, including headwords and etymological notes.
- `sources.bib`: References in BibTeX format.

The structure is more formally defined in `cldf/Wordlist-metadata.json`.

Source prose may additionally be supplied in the optional `cldf/entry-texts.csv` sidecar with
columns `Form_ID, Position, Kind, Format, Content, Source`. `Position` orders blocks on an entry;
`Kind` is a semantic label such as `etymology`, `comparison`, or `usage`; `Format` is `text`,
`markdown`, or trusted `html`; and `Source` uses the same CLDF citation syntax as forms. The web
database materializes legacy `Etymology` values into this shape when no explicit blocks exist.

### Raw data

**Raw data is organised under `data/`**. The script `make_cldf.py` builds the CLDF database in `cldf/` from the raw data. Raw data is all stored in CSV in order to be easy to edit and parse.

Run the complete data build with `make all`. It executes `make_cldf.py`, `link_refs.py`,
`unify_cldf.py`, `assign_form_ids.py`, and finally `align.py`; alignment must use the final unified `Origin_ID` graph so
borrowings and redirected entries are aligned to the same ancestors displayed by the application.

Attested forms receive persistent opaque IDs in `assign_form_ids.py`. The committed
`data/form-identities.csv` registry keeps those IDs stable across input reordering, etymological
reassignment, and transcription-profile changes; legacy generated IDs remain resolvable through
`cldf/form-id-aliases.csv`. Manual etymology links are stored separately in
`data/etymology-assignments.csv` and applied against persistent form IDs.

For raw data files that list lemmata, the columns are:
1. Language ID
2. Param ID (entry)
3. Lemma (normalised)
4. Gloss
5. Native script form
6. Phonemic form (in IPA)
7. Notes/comments
8. References

References use CLDF source locators when a citation points to a particular page, column, entry, or
item: `source-id[p. 42]` or `source-id[p. 42, col. 2]`. Multiple citations are separated with `;`.
Keep source location, raw OCR, parser-review flags, alternate-form labels, and other reproducible
parse metadata in the reference locator, audit output, relation columns, or tags as appropriate.
The Notes column is reserved for genuine editorial information that is not represented elsewhere.

For raw data files that list parameters (entries), the columns are:
1. Param ID
2. Language of the headword (e.g. "Indo-Aryan", "Proto-Dravidian")
3. Form
4. Gloss
5. References

Etymological notes for *all* params (as written up/collated by us) are stored in `data/etymologies.csv`. The columns are just the Param ID and Markdown-formatted notes.

Cross-dictionary loan relationships are curated in `data/borrowings.csv`. `Borrower_ID` is the borrowed entry and `Source_ID` is its source etymon; `unify_cldf.py` writes these to `Origin_ID`, `Relation`, and `Borrowed_From` in the unified form table.

Proto-Nuristani cognate links are curated in `data/nuristani_cognates.csv`. Each row joins a
Proto-Nuristani entry and an Indo-Aryan entry through a shared, intentionally blank
Proto-Indo-Iranian `Ancestor_ID`; `make_cldf.py` creates those placeholder nodes and
`unify_cldf.py` attaches both descendants as reflexes. Cases without a sufficiently clear parsed
Indo-Aryan match are recorded in `data/nuristani_cognates_uncertain.csv`.
When Strand places a Proto-Nuristani head beneath OIA rather than PAr, the reviewed donor is stored
in `data/nuristani_borrowings.csv`; `unify_cldf.py` attaches the PNur head directly to that
Indo-Aryan entry with `Relation=borrowed`. Its attested descendants are flattened beside the PNur
head as direct borrowings from the same Indo-Aryan donor, avoiding a claim that borrowing occurred
specifically at the Proto-Nuristani stage.

Finally, some sources have unusual orthographies that we need to convert to the Sāmapriya-n system. The profiles used by the `segments` library to do so are stored as `conversions/*.txt`; these give substitution rules for orthographic normalisation.

#### DEDR

The DEDR and related parsing scripts are all in `data/dedr/`. Originally, Suresh supplied a SQL database scraped from the online version (`data/dedr/dedr_new_entry_oct2013_edited.sql`) which was converted into a CSV at (`data/dedr/dedr.csv`). **These are now deprecated**.

The current CSV format of the DEDR is generated using `data/dedr/parse.py`, which scrapes the website and caches it in `data/dedr/dedr.pickle`, and then divides the entry into language spans (e.g. *Tam. word 'meaning', word2 'meaning2'...* is one span) and parses each span into forms and associated references and glosses using complicated regexes. The output is at `data/dedr/dedr_new.csv`.

The helper file `data/dedr/abbrevs.py` includes information about what each language tag and reference abbreviation corresponds to in the CLDF (e.g. Mal. = Malayalam).

Headwords for entries are to be stored in `data/dedr/params.csv`. Finally, Proto-Dravidian reconstructions are housed in `data/dedr/pdr.csv`.

#### CDIAL

The CDIAL directory is `data/cdial/` and is basically identically structured to the DEDR directory. Cache at `data/cdial/cdial.pickle`, parse script is `data/cdial/parse.py`, helper info in `data/cdial/abbrevs.py`, and params are at `data/cdial/params.csv`.

#### Dictionary of Gāndhārī

`data/other/forms/raw_data/gandhari_org.py` snapshots the Sanskrit-bearing articles exposed by
the public Gandhari.org dictionary API. It emits only unique, accent-normalized Sanskrit → CDIAL
head matches into `data/other/forms/20260805-gandhari-org.csv`; ambiguous and unmatched articles
are written to `tmp/gandhari-org-audit.csv`. Article JSON is cached under
`tmp/gandhari-org-cache/`, so refreshes and interrupted downloads are resumable.

#### Kullui dictionary

`data/other/forms/raw_data/kullui_org.py` snapshots the public JSON API used by
`kullui.org`. The live database (version 3.1.0 when ingested) is newer and richer than the July
2023 PDF export, so it is the canonical input. Every article is retained, including
unetymologised entries; explicitly identified Old Indo-Aryan and Sanskrit protoforms are linked
only when they have one exact, accent-normalized CDIAL head match. Article JSON is cached under
`tmp/kullui-org-cache/`, and all match outcomes are written to `tmp/kullui-org-audit.csv`.

#### Linguistic Survey of India comparative vocabulary

`data/other/forms/raw_data/grierson_lsi.py` imports the CC-BY-4.0 Lexibank v1.0
retrostandardization of Grierson's 1928 *Linguistic Survey of India: Comparative Vocabulary*
into `data/other/forms/20260813-grierson-lsi.csv`. Historical source varieties are retained as
`LSI-` aliases in `cldf/dialects.csv`, normalized onto existing Jambu parent languages during the
build, and tagged with their exact LSI lect label. The current mapping admits 28,552 forms from
153 varieties; unmatched comparative controls and distinct languages with no existing parent are
documented in `data/other/forms/raw_data/20260813-grierson-lsi-audit.csv` rather than forced into
an inappropriate language. Forms are unetymologised, carry immutable upstream keys, and cite the
printed page range plus upstream form and concept IDs. Upstream Glottolog-derived coordinates are
retained as dialect metadata with an explicit warning that they are not historical survey sites.

#### Toda dictionary

`data/other/forms/raw_data/bhaskararao_toda.py` extracts all 7,560 entries in Bhaskararao and
Kobayashi's 2025 *Toda Dictionary* into
`data/other/forms/20260813-bhaskararao-toda.csv`. The repository PDF appears image-only to ordinary
PDF libraries, but contains a Unicode text layer outside its visible crop box. The importer uses
Ghostscript's `txtwrite` XML output and page coordinates to recover that layer without OCR, retain
Toda's underlines, ogoneks, retroflex dots, and vowel length, and discard the duplicated adjacent
page outside the crop box. Printed S2/alternate stems become variant rows; every DEDR citation is
resolved to its etymon. The complete source text and parse decisions are preserved in
`data/other/forms/raw_data/20260813-bhaskararao-toda-audit.csv`.

#### Brahui texts glossary

`data/other/forms/raw_data/ali_kobayashi_brahui.py` extracts all 3,483 entries in the glossary
to Ali and Kobayashi's 2024 *Brahui Texts* (minor revision, 2025) into
`data/other/forms/20260813-ali-kobayashi-brahui.csv`. The importer reads the PDF's two-column
Unicode text layer without OCR, using bold spans to separate headwords from grammatical labels
and definitions. It retains the source transcription in `Phonemic`, printed page locators,
loan-language labels, and the three entries explicitly marked Rakhshān. Parse decisions are
preserved in `data/other/forms/raw_data/20260813-ali-kobayashi-brahui-audit.csv`.

#### DBIA

`data/dbia/parse.py` conservatively extracts Emeneau and Burrow's *Dravidian Borrowings from
Indo-Aryan* into `forms.csv` and `params.csv`. It preserves each complete OCR entry, records
two-column boundary repairs and match decisions in `parse_audit.csv`, and writes high-confidence
normalized headword matches to `cdial_redirects.csv`. During unification, matched `dbiaN` entries remain as
resolvable redirect stubs while their Dravidian loans and source transcription are folded into the
canonical CDIAL entry.

#### Munda
- `data/munda/forms.csv`
- `data/munda/params.csv`

#### Others

- `data/others/forms/*.csv`: New lemmata extracted from various sources.
- `data/others/params/*.csv`: New entries.
