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
`unify_cldf.py` attaches both descendants as reflexes. Because CDIAL classifies Nuristani inside
Indo-Aryan, its Nuristani reflexes on these inherited entries are reparented from the Indo-Aryan
sibling to Strand's Proto-Nuristani head. When several reviewed PNur heads correspond to one CDIAL
entry, the build routes each CDIAL reflex using same-language Strand evidence before normalized
form similarity. Cases without a sufficiently clear parsed Indo-Aryan match are recorded in
`data/nuristani_cognates_uncertain.csv`.
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
the public Gandhari.org dictionary API. The pinned 2026-08-05 snapshot contains 5,807 articles:
1,512 unique, accent-normalized Sanskrit → CDIAL head matches are installed in
`data/other/forms/20260805-gandhari-org.csv`; 371 ambiguous matches, 3,923 unmatched articles, and
one article without a parsed Sanskrit etymon remain in the checked-in
`data/other/forms/raw_data/20260805-gandhari-org-audit.csv`. Article JSON is cached under
`tmp/gandhari-org-cache/`, so refreshes and interrupted downloads are resumable and the pinned
snapshot can be rebuilt offline.

Installed rows represent dictionary lemmata, not full paradigms. The importer maps the article's
part of speech, gender, and pronominal subtype to canonical tags and cites the stable article ID,
homograph number when present, and lemma. Gandhari.org supplies no printed page/column metadata
for these records. Declined/conjugated forms and corpus attestations are retained in the audit but
intentionally excluded from Jambu's Notes/reference display. Reuse terms were not stated on the
source site when the snapshot was taken.

#### Kullui dictionary

`data/other/forms/raw_data/kullui_org.py` snapshots the public JSON API used by
`kullui.org`. The live database (version 3.1.0 when ingested) is newer and richer than the July
2023 PDF export, so it is the canonical input. Every article is retained, including
unetymologised entries; explicitly identified Old Indo-Aryan and Sanskrit protoforms are linked
only when they have one exact, accent-normalized CDIAL head match. Article JSON is cached under
`tmp/kullui-org-cache/`, and all match outcomes are written to `tmp/kullui-org-audit.csv`.

#### Nuristani Etymological Dictionary (NurED)

`data/other/forms/raw_data/nured_org.py` snapshots the live CC-BY-SA-4.0 NurED MediaWiki by stable
page ID and exact revision ID. The 2026-08-18 snapshot inventories all 875 namespace-0 pages,
excludes 770 hard redirects, and audits all 105 nonredirect pages. Its target scope is the 22
Proto-Nuristani and 25 Middle Indo-Aryan loanword articles; the remaining 58 pages are site or
reference material. Forty-five articles are attached as 46 complete, source-linked HTML etymology
blocks to corresponding CDIAL or Strand PNur heads. The PNur barley article deliberately appears
on both Strand branches that NurED groups together. Two PNur articles without a corresponding
existing PNur head remain visible in the audit rather than being matched by phonetic similarity.

The installed sidecar is `data/other/entry_texts/20260818-nured-org.csv`; raw wikitext, sanitized
rendered HTML, checksums, categories, revision dates, match candidates, and every exclusion are in
`data/other/forms/raw_data/20260818-nured-org-audit.csv`. Reviewed exceptional targets live in the
small `20260818-nured-org-targets.csv` overlay. No form transcription, language/dialect record, or
new graph edge is introduced, so a sound profile is intentionally inapplicable. A weekly GitHub
Actions refresh opens a review PR when NurED revisions change; `--offline --install` rebuilds the
sidecar from the checked-in snapshot without network access.

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
The build converts the upstream CLTS segmentation to Jambu house transcription for the displayed
form, preserves that segmentation as `Phonemic`, and retains Grierson's normalized spelling as
`Original`.

#### Toda dictionary

`data/other/forms/raw_data/bhaskararao_toda.py` audits all 7,560 entries in Bhaskararao and
Kobayashi's 2025 *Toda Dictionary* into
`data/other/forms/20260813-bhaskararao-toda.csv`; 7,558 readable entries are installed and two
heads represented only by corrupt replacement glyphs remain audit-only. The repository PDF appears image-only to ordinary
PDF libraries, but contains a Unicode text layer outside its visible crop box. The importer uses
Ghostscript's `txtwrite` XML output and page coordinates to recover that layer without OCR, retain
Toda's underlines, ogoneks, retroflex dots, and vowel length, and discard the duplicated adjacent
page outside the crop box. Printed S2/alternate stems become variant rows; every DEDR citation is
resolved to its etymon. The complete source text and parse decisions are preserved in
`data/other/forms/raw_data/20260813-bhaskararao-toda-audit.csv`.

#### Marati of Kasargod survey vocabulary

`data/other/forms/raw_data/ghatage_survey.py` extracts the vocabulary on printed pages 136--168
of Ghatage's 1970 *Marati of Kasargod* into
`data/other/forms/20260817-ghatage-marati-kasargod.csv`. The importer uses the rendered scan and
the Tesseract Latin-script model because the public mirror's text layer loses both column structure
and phonetic characters. It audits all 1,244 detected or manually recovered lexical records and
emits 1,271 installed rows after splitting printed variants. Every installed row has an exact
printed-page/entry locator, a stable source key, and an `ocr-review` tag; no OCR head is silently
represented as human-verified.

The 129 source-image-verified corrections and recoveries are stored in
`data/other/forms/raw_data/20260817-ghatage-marati-kasargod-corrections.csv`; the remaining 1,115
records are structurally accepted but retain explicit transcription uncertainty in
`data/other/forms/raw_data/20260817-ghatage-marati-kasargod-audit.csv`. One corrupt alternate form
is audit-only while its readable main form remains installed. A deterministic 20-record seed-1970
sample is recorded in `data/other/forms/raw_data/20260817-ghatage-marati-kasargod-sample.csv`; all
20 final readings pass source-image review after 14 original OCR readings required material
correction. The dedicated `conversion/ghatage.txt` profile preserves the source's central and open
vowels, vowel length, retroflexes, palatals, and nasals rather than routing it through the generic
Marathi profile. Scan provenance, page alignment, and the reconstructed-PDF digest are documented
in `data/other/forms/raw_data/ghatage_survey_sources.md`.

#### Southworth's Dravidian element in Marathi

`data/other/forms/raw_data/southworth_marathi.py` ingests printed/PDF pp. 9--10 of Franklin
Southworth's 2005 paper: Table 1's 25 Marathi words of proposed Dravidian origin and Table 2's 23
comparative distribution records. The installed form file contains 25 Marathi rows plus five
separately printed Old Marathi forms. Every form is linked as a borrowing to the source's explicit
DEDR ID; `phaḷ` remains tagged uncertain because Southworth marks its origin controversial.

The PDF's legacy-font text layer corrupts the linguistic symbols, so the pages were rendered at
400 dpi, OCRed with Tesseract 5 (`eng`, PSM 6), and fully checked against the source images. Raw
OCR, exact distribution marks, source-image corrections, stable page/table keys, and the two
citation anomalies are retained in
`data/other/forms/raw_data/20260818-southworth2005m-audit.csv`. Printed Table 2 item 20 cites
Turner 5634 for `taḍāga` 'pool'; because that entry is actually `*taḍapphaḍ` 'agitate', the unique
headword-and-gloss match is installed on 5635 with a typed correction note. Item 17's printed
3276--3278 range is attached only to matching 3277--3278; conflicting 3276 remains audit-visible.
Table 2's plus/minus/question cells are comparison blocks on existing CDIAL entries, not invented
forms for the column languages. Southworth's `@` variable adjective ending is preserved, while
the dedicated sound profile applies only the paper's explicit vowel-length rules. The final
counts, exclusions, transcription decisions, validation, and representative browser entries are
recorded in `data/other/forms/raw_data/20260818-southworth2005m-ingestion.md`.

#### Brahui texts glossary

`data/other/forms/raw_data/ali_kobayashi_brahui.py` extracts all 3,483 entries in the glossary
to Ali and Kobayashi's 2024 *Brahui Texts* (minor revision, 2025) into
`data/other/forms/20260813-ali-kobayashi-brahui.csv`. The importer reads the PDF's two-column
Unicode text layer without OCR, using bold spans to separate headwords from grammatical labels
and definitions. It retains the source transcription in `Phonemic`, printed page locators,
loan-language labels, and the three entries explicitly marked Rakhshān. Parse decisions are
preserved in `data/other/forms/raw_data/20260813-ali-kobayashi-brahui-audit.csv`.

#### Thari thesis vocabulary

`data/other/forms/raw_data/thari.py` builds a review queue for Bhawnani's 1979 Thari vocabulary
after the previously hand-reviewed entry *gā̃* 'village'. It uses the ABBYY word coordinates
from the standalone vocabulary PDF for row segmentation and checks them against the duplicate
pages embedded in the annexures. The embedded text is **not** accepted as a transcription: on
held-out printed p. 200 it badly corrupts ordinary letters, while fresh Tesseract OCR still has
a 32.6% base-character error rate and drops contrastive diacritics. A source-specific,
form-only Tesseract experiment lowers base-character error to 10.3%, but cannot emit the full
transcription alphabet. Fine-tuning the Kraken CATMuS-Print Tiny model on 196 reviewed form
crops reaches 78.5% full-Unicode character accuracy and 50.0% exact-word accuracy on 14 held-out
forms from printed p. 200. This is a useful improvement but does not yet meet the bar for
ingestion. Consequently the script marks the 2,583 continuation rows `needs_review`
in `data/other/forms/raw_data/20260817-thari-audit.csv` and refuses `--install`. The exact
held-out comparisons are recorded in
`data/other/forms/raw_data/20260817-thari-calibration.csv`; training details and the comparison
with the Old Punjabi OCR pipeline are in
`data/other/forms/raw_data/20260817-thari-kraken.md`.

#### Nihali lexicons

`data/other/forms/raw_data/nihali_database.py` installs the reviewed spreadsheet *The Nihali
database* as the canonical Nihali import. The snapshot is pinned by Drive modification time and
SHA-256, and contributes 4,065 form variants from 3,976 lexical records: Mundlay (1,707 variants),
Nagaraja (1,761), Bhattacharya (407), and Konow (190). The Contact, Roots, and Dravidian tabs are
analysis sidecars rather than independent attestations; all 59 excluded rows remain in the audit,
and twelve Dravidian analyses are merged into their matching Nagaraja records.

The browser database exposes 4,063 Nihali rows: its standard exact-attestation compactor merges
the two duplicate Nagaraja spellings `gegeliya` and `gengeliya` from source IDs 615 and 621 while
retaining both source locators and durable aliases. The CLDF and ingestion audit retain all 4,065
source variants.

The replacement preserves 3,104 prior immutable keys after exact or conservative reconciliation.
The older Mundlay OCR and Wiktionary snapshot audits remain for provenance, but
`raw_data/nihali.py` is a superseded reconstruction tool and no longer writes canonical outputs by
default. Source and editor etymologies remain labeled; only unambiguous printed CDIAL/DEDR IDs are
turned into borrowing hypotheses. A private-use glyph in one Nagaraja form is preserved and marked
uncertain rather than silently guessed. The canonical audit and key map are
`data/other/forms/raw_data/20260817-nihali-database-audit.csv` and
`data/other/forms/raw_data/20260817-nihali-database-key-map.csv`.

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
