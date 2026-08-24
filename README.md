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

Source-attributed comparisons between dictionary articles live in the optional
`cldf/comparisons.csv` sidecar. Its columns are `ID, Entry_ID, Compared_Entry_ID, Relation,
Direction, Confidence, Source, Evidence`. `Entry_ID` is the article that prints the claim;
`Direction` is therefore source-relative (`entry-from-compared`, `compared-from-entry`, or
`undetermined`). `Confidence` describes the certainty of the printed wording, not confidence in
resolving an ID. These rows are scholarly comparisons, not accepted ancestry edges.
Reproducibly extracted rows are stored in `data/cross-family-comparisons.csv`; curated editorial
comparisons live separately in `data/manual-cross-family-comparisons.csv`, and `make_cldf.py`
validates and combines both files into the compiled sidecar.

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

Cross-dictionary loan relationships accepted by Jambu's editors are curated in
`data/borrowings.csv`. `Borrower_ID` is the borrowed entry and `Source_ID` is its source etymon;
`unify_cldf.py` writes these to the ancestry graph. This is deliberately stricter than printed
cross-family comparisons, which remain in `data/cross-family-comparisons.csv` even when the source
hedges or leaves direction unresolved.

Burushaski--Indo-Aryan links are migrated under that stricter policy by
`burushaski_comparisons.py`. A Burushaski attestation whose legacy ancestry path reaches a numeric
CDIAL article is rehomed beneath a stable Proto-Burushaski grouping node; the grouping node's
`Form` and `Original` are intentionally blank because Jambu does not reconstruct Proto-Burushaski.
This applies to all PBr comparative sets, including the pre-existing Berger/HKAT/Yoshioka groups.
The PBr--CDIAL link is emitted only in `cldf/comparisons.csv` as `related`
with undetermined direction. Exact Berger page locators and etymology snippets are retained;
CDIAL cross-reference chains cite both the article that prints the Burushaski item and the terminal
comparison endpoint. Backstrom contributes lexical attestations, but its legacy Jambu targets are
explicitly marked as editorial links rather than claims made by the wordlist source.

The current migration accounts for 787 Burushaski attestations in 524 PBr grouping sets and 533
source-attributed comparisons: 565 attestations from the Berger OCR ingest, 39 from the earlier
hand-entered Berger tranche, 162 from Backstrom, and 21 printed by CDIAL. The complete decision log
is `data/burushaski-indo-aryan-comparisons-audit.csv`; the deterministic, source-stratified checked
sample is `data/burushaski-indo-aryan-comparisons-sample.csv`.

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

DEDR article prose is preserved separately from that reflex inventory. Run
`python -m data.dedr.entry_texts` for a dry run or add `--install` to regenerate
`data/other/entry_texts/20260819-dedr.csv`, `data/dedr/entry-references.csv`, and the
per-article audit/sample/manifest beside the importer. The extractor retains same-family and
otherwise unresolved comparisons plus editorial/source notes, but excludes ordinary reflex runs
and claims already represented in the structured cross-family table. Printed `DED`, `DED(S)`,
`DED(N)`, `DED(S, N)`, `DEDS`, and `DEN` numbers become locators on the entry's `dedr` citation.
Four repeated website article numbers are retained in the audit as excluded duplicate source
records rather than attaching the wrong second article to the canonical entry.

DEDR's Sanskrit/Indo-Aryan comparison lemmata are not reflexes and are excluded from that output.
`data/cross_family.py` instead parses their article-level CDIAL links, source wording, direction,
and confidence into `data/cross-family-comparisons.csv`; run it without `--install` for an offline
dry run under `tmp/cross-family-comparisons/`. The full installed/unresolved/excluded decision log
and fixed reviewed sample are checked in alongside the table.

The helper file `data/dedr/abbrevs.py` includes information about what each language tag and reference abbreviation corresponds to in the CLDF (e.g. Mal. = Malayalam).

Headwords for entries are stored in `data/dedr/params.csv`. The legacy curated Proto-Dravidian
reconstructions are housed in `data/dedr/pdr.csv`. Merriam and Fuls's CC-BY-4.0 *Dravidian
Database* v1.0 reconstruction table is installed separately at
`data/other/forms/20260718-merriam-dravidian-db.csv`, with its seven asserted reconstruction
levels represented as distinct proto-language IDs. Its reproducible importer, complete record
audit, manifest, and seeded sample are under `data/other/forms/raw_data/`. Integer records that
collapse a DEDR `N`/`NA` pair and records pointing to absent DEDR slots remain audited but
unlinked rather than being assigned heuristically.

The compiled DEDR entry header follows a fixed evidence hierarchy in `unify_cldf.py`: retain the
curated Krishnamurti/Pfeiffer head when present; otherwise use the first source-ordered Merriam
row explicitly classified as Proto-Dravidian; otherwise use the first surviving DEDR reflex as a
display-only head tagged `not-reconstructed`. The last form is deliberately left unstarred and is
not asserted as Proto-Dravidian. `cldf/pdr-headword-audit.csv` records the selected tier, source
form, citation, and four obsolete DEDR slots for which no defensible head is available.

#### CDIAL

The CDIAL directory is `data/cdial/` and is basically identically structured to the DEDR directory. Cache at `data/cdial/cdial.pickle`, parse script is `data/cdial/parse.py`, helper info in `data/cdial/abbrevs.py`, and params are at `data/cdial/params.csv`.

Likewise, cited Dravidian comparison forms are excluded from `cdial.csv` and parsed into the same
article-comparison table. Old DED/DEDS numbers are resolved through current DEDR footers; ambiguous
legacy numbers require an explicit reviewed row in `data/cross-family-comparison-overrides.csv`.

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
reference material. The importer parses the Nuristani section's 255 explicit `Form` templates into
263 source-keyed reflex rows in `data/other/forms/20260818-nured-org.csv`. These rows attach to a
corresponding Proto-Nuristani entry rather than directly to CDIAL. A Middle Indo-Aryan article uses
an existing compatible PNur borrowing sibling when one is available; otherwise it creates a stable
`nured-<page ID>` PNur reconstruction and records that reconstruction's borrowing from CDIAL.
Eighteen PNur entries are generated in this snapshot, sixteen of them as CDIAL borrowing siblings.
The two PNur articles that previously had no target now receive stable generated PNur entries.

Only each article's Commentary section and its referenced footnotes are installed in
`data/other/entry_texts/20260818-nured-org.csv`, always on the PNur target; the lexical and source
sections are not dumped into entry text. Raw wikitext, sanitized rendered HTML, checksums,
categories, revision dates, target routing, template counts, and every exclusion remain in
`data/other/forms/raw_data/20260818-nured-org-audit.csv`. Reviewed exceptional dictionary targets
live in the small `20260818-nured-org-targets.csv` overlay. Source spellings pass through the
lossless `conversion/nured.txt` profile, and source dialect labels are registered in
`cldf/dialects.csv`. A weekly GitHub Actions refresh opens a review PR when revisions change;
`--offline --install` reproduces forms, PNur parameters, borrowings, commentary, and audit outputs
from the checked-in snapshot without network access.

#### Mayrhofer's KEWA

`data/other/forms/raw_data/kewa.py` snapshots the 2021 version-1.0 web image edition of Manfred
Mayrhofer's *Kurzgefasstes etymologisches Wörterbuch des Altindischen*. The index supplies 9,587
stable article IDs, exact volume/page locators, reviewed Sanskrit headwords, and one tightly
cropped image per article; the article bodies themselves are image-only. The importer pins the
index checksum, caches and verifies every image under ignored `tmp/kewa-cache/`, and runs
Tesseract 5 with the `script/Latin` model, page-segmentation mode 6, and a fixed 300-dpi hint.

Every article, image checksum, raw OCR result, head match, and exclusion is retained in
`data/other/forms/raw_data/20260818-kewa-audit.csv`. KEWA contributes no invented attestation,
language, transcription, sound-profile mapping, or graph edge: conservatively matched Sanskrit
articles become source-attributed blocks in `data/other/entry_texts/20260818-kewa.csv`. Each block
contains the exact cropped source scan and links to its stable article anchor. OCR is audit-only:
no OCR text enters the database, affects a match, or appears to readers. Accent-sensitive exact
matches take precedence; accent-neutral matches must be unique; where main-dictionary senses
collide, only a sole accented index match can win and all other cases remain unresolved. Matching
a KEWA article never marks the independently transcribed CDIAL headword as OCR-derived. The site
states that scanning was done with the author's permission but gives no explicit reuse licence.

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
forms for the column languages. Seven Table 2A items also print both a Table 1 DEDR source and a
Turner/CDIAL target; `data/cross_family.py` exposes exactly those checked pairs as
Southworth-attributed cross-family comparisons. Six are explicit high-confidence loan claims;
`phaḷ`/CDIAL 9051 remains low-confidence because the paper marks its Dravidian origin
controversial. Rows lacking either printed endpoint remain prose/audit evidence rather than
inferred links. Southworth's `@` variable adjective ending is preserved, while the dedicated
sound profile applies only the paper's explicit vowel-length rules. The final counts, exclusions,
transcription decisions, validation, and representative browser entries are recorded in
`data/other/forms/raw_data/20260818-southworth2005m-ingestion.md`.

#### Emeneau's new Brahui etymologies

`data/other/forms/raw_data/emeneau_brahui_1997.py` is the first page-agent pilot for an
unstructured comparative article. Emeneau's eight printed pages (440--447) were rendered and
assigned one at a time to `gpt-5.6-luna` under the checked extraction contract in
`data/other/forms/raw_data/emeneau_brahui_1997_prompt.md`. The checked-in page JSON contains 76
raw claim records: pages 440 and 447 correctly contribute zero, while lexical attestations,
DEDR/CDIAL assignments, rejections, reassignments, sound changes, and unresolved comparisons are
typed separately on pages 441--446.

Agent output is evidence, not installation authority. The importer applies an image-checked
reconciliation layer that records 18 corrections, including `bāšt`, `taṛifing`, and `cīkap-`,
canonicalizes target IDs, and deduplicates cross-page discussion. It emits 19 rich form rows and
35 source-attributed entry-text blocks. Accepted claims receive rank-1 edges; tentative analyses
for `pužža`, `kūžing`, `pisfing`, `šupping`, and `dūī` remain unlinked forms with rank-2/rank-3
hypotheses. The overlay also reassigns Brahui `(h)ullī` from DEDR 500 to 701, links `dū` as a
probable northwestern Indo-Aryan borrowing from CDIAL 6586, and models Gadaba `cīkap-` as a loan
from the existing Telugu `cīk-` reflex. The homonymous `taṛifing` 'turn sour (milk)' remains
unetymologized.

Emeneau's underlined `gh` is preserved in `Original` and converted to display `ɣ` by
`conversion/emeneau-brahui.txt`. The publisher PDF is not redistributed; its SHA-256, rights note,
page counts, per-page record counts, audit, deterministic 20-record sample, and all reconciliation
decisions are recorded in the source manifest and audit beside the importer.

#### Burrow and Emeneau's Dravidian Etymological Notes, part I

`data/other/forms/raw_data/burrow_emeneau_1972_den1.py` extends the page-agent experiment to a
substantially denser unstructured source: all 22 printed pages (397--418) of Burrow and Emeneau's
1972 *Dravidian Etymological Notes*, part I. Each rendered page was assigned in isolation to
`gpt-5.6-luna` under `burrow_emeneau_1972_den1_prompt.md`. Pages 397--398 correctly yield no
lexical records; page 399 preserves the bibliography/lexicon boundary and begins with 20 numbered
segments in its lower-right column. The complete raw layer contains 1,154 numbered page segments
and 1,324 nested form candidates across the DED and DEDS sections.

This larger trial draws a sharper boundary between structural extraction and diplomatic
transcription. Luna reliably recovers entry boundaries, operations, link targets, form grouping,
and uncertainty, but often drops length, retroflexion, or other dense Dravidianist diacritics.
The importer therefore publishes only 709 active/corrected forms whose language, old-entry
resolution, and transcription are independently corroborated by the later DEDR. It maps 467 old
entry groups to current DEDR targets and records every later split rather than choosing a branch
by number alone. Of those forms, 286 carry unambiguous registered dialect IDs (including Thiyya,
Gondi locality codes, and Onti/Mudu/Tappu Koraga); bibliographic sigla and mixed-dialect labels
remain at the base-language level. Representative reconciliations include old 435 `iḷusan` to
d512, old 694 `talay-ēru` to d811, old 2127 `jicoṇa` to d800, and old 3722 `boḷi` to d4556.

The 2,478-row audit accounts for every numbered segment and every nested form. It keeps 304
uncorroborated transcriptions, 153 comparison-only forms, 88 queried forms, 43 deletions, ten
loans, six unresolved split targets, and one combined variant field audit-only; two exact source
duplicates are collapsed. Page-agent running prose is likewise retained in JSON and audit rather
than published as a diplomatic entry-text block. Those explicit exclusions are the principal
result of the pilot: page-wise cheap-agent extraction is viable for structure, but requires a
second transcription/reconciliation pass before uncrosschecked forms or prose can enter Jambu.
The publisher PDF is not redistributed; its digest, rights note, source scope, corrections,
exclusions, and deterministic sample are recorded in the manifest beside the importer.

#### Burrow and Emeneau's Dravidian Etymological Notes, part II

`data/other/forms/raw_data/burrow_emeneau_1972_den2.py` continues the page-isolated Luna pilot
with all 17 printed article pages (475--491) of part II. The five lexical pages contribute 119
page-local numbered segments and 448 split form candidates; the twelve Dravidian and English
index pages are retained as explicitly non-lexical JSON with no form rows. The scan's labels such
as `S21` are plain-text encodings of printed S²1 (part-II new entry 1), not historical DEDS 21.
The importer therefore resolves the S² sequence by conservative current-DEDR language, form, and
gloss corroboration rather than by the old-number map used for part I.

That reconciliation installs 159 DEDR-corroborated forms (161 compiled citation attestations),
including Tamil `accu` under d49,
Kota `koyk` under d2121, Tulu `sūri` under d2728, Kui `tōṛa (tōṛi-)` under d3523, and Kodagu
`pu·ḷï` 'mist' under d4375. The last case is an important guardrail: the page agent's unmarked
`puḷi` is an exact homonym of a 'sour' form in d4322, but the source gloss plus the fuller DEDR
transcription recover the correct entry. Thirty-nine installed rows use unambiguous registered
dialect IDs.

The 567-row audit also keeps 46 structurally extracted but transcription-unreconciled DEDS forms,
25 DEDS forms without a current language/form target, and 66 queried, comparative, deleted,
loan, or otherwise non-reflex candidates out of the reflex table. All 152 active DBIA
addition/correction candidates are preserved but deferred to a separate loan-entry reconciliation;
the structural pilot does not disguise them as inherited DEDR reflexes. Page-agent running prose
remains raw/audit evidence pending diplomatic image review. The publisher PDF is not redistributed;
its SHA-256, rights note, page scopes, agent corrections, exclusions, and deterministic sample live
beside the importer.

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
