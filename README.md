# moli-mandala/data

The CUJ Asur dictionary draft adds **2,106 Asuri source rows**, including native
spelling, 185 explicit variant links and one two-component compound. See the
[source package](data/other/forms/raw_data/cuj_asur_2020/README.md) for extraction,
transcription, exclusions and audit evidence. Source-local checks pass; full
compiled integration and browser QA are still pending.


![Status](https://github.com/moli-mandala/data/actions/workflows/python-app.yml/badge.svg)

This is a CLDF database for the Jambu application, containing historical linguistic data for many languages of South Asia. It also contains the underlying raw data and scripts used to produce/update the CLDF database.

Peterson et al. (2024), Appendix 1, adds 246 Odisha Turi forms to the source
inputs, extending a language with only 23 rows in the pre-ingest compiled data.
The [source package](data/other/forms/raw_data/peterson_turi_2024/README.md)
records the pinned publisher edition, 275-record audit, transcription decisions,
12 passing focused tests, and outstanding full-build/browser gates.

Zoller (2023), *Linguistic data I–III* (pp. 519–1035), contributes 17,754 resolved
attestations in 328 languages after separating its West Pahari varieties. The importer decodes the PDF's TeX phonetic fonts,
preserves heterogeneous source transcriptions, and audits all 4,377 source records.
See the [source package](data/other/forms/raw_data/zoller_2023/README.md) and
[validation review](source_checklists/20260913-zoller-review.md) for scope, exclusions,
language decisions and reproducible extraction.

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
Raw sidecars in `data/other/entry_texts/` may additionally use `Entry_Key` with a blank
`Form_ID`. The build resolves that key against emitted forms before producing the standard
six-column sidecar; missing, ambiguous, or conflicting targets fail rather than losing prose.

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
`cldf/form-id-aliases.csv`. Manual etymology links are stored next to the source they annotate,
one `data/other/forms/etymologies/<source>.csv` per lexical source (dictionary-entry children go
to `data/<dictionary>/etymologies.csv`), and applied against persistent form IDs. See
`etymology_assignments.py`.

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
`form_note_policy.py` enforces that boundary for legacy and batch importers whose local source
rows still retain audit prose in column 7: structured leading tokens are promoted to `Tags`, exact
page/item IDs remain in `Source`, and the residual provenance is omitted from compiled CLDF and
the browser database. The checked-in source/audit artifacts remain the reproducibility record.

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

The earlier compiled migration snapshot accounts for 654 direct Burushaski claim attestations in 418 PBr grouping
sets and 426 source-attributed comparisons: 431 attestations from the cleaned Berger OCR ingest, 39 from the
hand-entered Berger tranche, 162 from Backstrom, and 22 printed by CDIAL. The complete decision log
is `data/burushaski-indo-aryan-comparisons-audit.csv`; the deterministic, source-stratified checked
sample is `data/burushaski-indo-aryan-comparisons-sample.csv`.
These are historical build counts; the September Berger source repairs, including withheld
tentative manual links, await the consolidated full rebuild.

Berger's complete Burushaski--German dictionary (1998, printed pp. 9--486) is reconstructed by
`data/other/forms/raw_data/berger_cleanup.py` from the 300 dpi cache produced by `berger.py`.
The September repair measures the four column margins on each scan and uses native PDF fonts
to distinguish lexical text from German definitions and grammatical labels. Surviving keys are
anchored to their original physical lines; retired parsing fragments have reviewed source-key
redirects. The original August crosswalk, translations and audit remain historical evidence.
Current source notes, full grammar, class/number tags, attested paradigms, checked corrections,
translation hashes, per-record audit, unresolved references and counts live in
`data/other/forms/raw_data/berger_2026/`. Questionable accepted cognate bindings use explicitly
tagged original evidence instead of uncertain article matches. OCR and machine translation
remain unreviewed outside the recorded image checks, including the old cache's conflation of
`g` and `ġ`. See [the repair report](source_checklists/20260914-berger-repair.md) for validation
and deferred integration gates. `make berger` uses the canonical repair importer; a run without
`--install` is read-only unless an explicit proposal output directory is supplied.

Nuristani evidence uses CDIAL entries as editorial grouping points wherever a correspondence is
available. `nuristani_grouping.py` runs at the end of `assign_form_ids.py`, after durable IDs and
curated assignments. Proto-Nuristani reconstructions and their former attested descendants become
siblings directly beneath the corresponding Indo-Aryan entry. Attested variants retain their
lexical variant target. The grouping leaves inheritance from PII versus borrowing from IA
undetermined; its schema-compatible `reflex` edge has `grouping:cdial` in `Note`, and the form has
an `etymology-group` tag which the site renders as “Grouped with.”

`data/nuristani_cognates.csv` and `data/nuristani_borrowings.csv` retain the previously reviewed
correspondences and historical source interpretations; they no longer generate contrasting trees.
`data/nuristani_cdial_groups.csv` supplies additional CDIAL correspondences for Strand's standalone
PII heads, with the matching inventory in `data/other/analysis/nuristani-grouping/`. Real
reconstructions retain their text and source attribution as grouped siblings. The old blank
`pii-*` grouping nodes become redirects to CDIAL, preserving public IDs without retaining artificial
ancestry. Heads without a resolved CDIAL correspondence keep their existing groups.

For an existing built checkout, `python nuristani_grouping.py --install` applies the same pass with
bounded form loading, validates all graph endpoints and accepted ancestry cycles, checks
idempotence, and regenerates only alignments whose targets changed. It keeps pre-migration files
and a change audit in `tmp/nuristani-cdial-grouping/`. This is a graph migration, not a re-ingestion
or automatic merger of independent source records.

Finally, some sources have unusual orthographies that we need to convert to the Sāmapriya-n system. The profiles used by the `segments` library to do so are stored as `conversions/*.txt`; these give substitution rules for orthographic normalisation.

**House transcription policy (2026-09-19).** Every profile writes the house transcription
that `conversion/cdial.txt` produces for Turner, and `profile_policy.py` enforces it
(`uv run python profile_policy.py check`, also run by `tests/test_sound_profiles.py`;
`fix` rewrites the profiles and adds a `Vː`/`Cː` rule for every length bigram that occurs in the
sources routed through a profile). Vowels follow Turner's phonemic system: the house pairs are
a/ā, i/ī, u/ū, so a source's IPA `ʌ` (and, in Indo-Aryan and Dravidian survey IPA, `ə`) is `a`,
`ɪ` is `i`, `ʊ` is `u`, and the vowel the source writes as long — plain `a`/`i`/`u` in the
SIL-style transcriptions that reserve `ʌ`/`ɪ`/`ʊ` for the short series, `aː`/`ɑ` elsewhere — is
`ā`/`ī`/`ū`. Which convention each source follows is a per-profile decision recorded in
`profile_policy.py` (`LONG_A`, `LONG_I`, `LONG_U`, `SCHWA`), derived from every source's
vowel inventory and reviewed by hand; `ə` stays `ə` where Turner writes it (Dardic, Nuristani,
Kashmiri, Burushaski, Romani) and where it is a phoneme (Tibeto-Burman, Munda, Kusunda, Nihali,
scholarly Dravidian). `e ɛ o ɔ æ ɨ` keep their quality (no more `ɛ`→`e`, `ɔ`→`o`); a long
consonant is doubled. Consonants use the Indological letters (`tʃ dʒ ʃ ʒ ʂ ʈ ɖ ɳ ɽ ɭ ɲ ŋ` →
`c j ś ź ṣ ṭ ḍ ṇ ṛ ḷ ñ ŋ`, `ʱ`/`Ch` → `Cʰ`, IPA `j` → `y` where the source writes the affricate
as `dʒ`, `w`/`ʋ` → `v`, `ṃ` → `ṁ`, `ṅ` → `ŋ`); the Marathi sources that contrast dental c/j with
palatal č/ǰ (SDML, Ghatage) write the dental pair as the house `ʦ`/`ʣ`. Symbols with no house
counterpart (`ɐ ɑ ɤ ɯ ø ʔ ʕ`, breathy `̤`, half-length `ˑ`, tone and stress marks) stay as the
profile author chose; DEDR's `è`, Emeneau's Toda `ü` and the Chhattisgarhi/Magahi `ɔ`/`æ`
(= Hindi au/ai) are explicit exemptions. `Original`/`Phonemic` always keep the source's own
spelling, and form IDs are fingerprinted on `Original`, so a profile change never re-mints an ID.

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
SHA-256, and contributes 4,087 form variants from 3,976 lexical records: Mundlay (1,707 variants),
Nagaraja (1,783), Bhattacharya (407), and Konow (190). Nagaraja's optional vowel length (`ga(ː)ra`)
is installed as two forms, the long reading first and the short one as its `:short` variant, as
CDIAL's `ā̆` is. The Contact, Roots, and Dravidian tabs are
analysis sidecars rather than independent attestations; all 59 excluded rows remain in the audit,
and twelve Dravidian analyses are merged into their matching Nagaraja records.

The browser database exposes 4,085 Nihali rows: its standard exact-attestation compactor merges
the two duplicate Nagaraja spellings `gegeliya` and `gengeliya` from source IDs 615 and 621 while
retaining both source locators and durable aliases. The CLDF and ingestion audit retain all 4,087
source variants.

The replacement preserves 3,104 prior immutable keys after exact or conservative reconciliation.
The older Mundlay OCR and Wiktionary snapshot audits remain for provenance, but
`raw_data/nihali.py` is a superseded reconstruction tool and no longer writes canonical outputs by
default. Source and editor etymologies remain labeled; only unambiguous printed CDIAL/DEDR IDs are
turned into borrowing hypotheses. A private-use glyph in one Nagaraja form is preserved and marked
uncertain rather than silently guessed. Mundlay, Nagaraja and Bhattacharya share the `nihali`
sound profile; Konow follows the Linguistic Survey's romanisation (`ch` = c, `chh` = cʰ) and has
its own `nihali-konow` profile. The canonical audit and key map are
`data/other/forms/raw_data/20260817-nihali-database-audit.csv` and
`data/other/forms/raw_data/20260817-nihali-database-key-map.csv`.

#### Kalkoti (Hultman 2023)

`data/other/forms/raw_data/hultman_kalkoti_2023.py` installs every Kalkoti form in David Hultman's
open-access BA thesis *Topics in the grammar of Kalkoti* (Stockholm University, supervised by
Henrik Liljegren). It is the fullest description of Kalkoti that exists: 65 pages built on eleven
datasets recorded from four speakers between 2006 and 2023, and the first systematic account of the
language's tone system. Like Knobloch's Sauji sketch it is a grammar rather than a dictionary, so
its lexicon lives in numbered tables, interlinear examples and citations in running prose.

The PDF is XeTeX output with a real text layer, so no OCR is involved, but two properties of that
layer drive the extraction. **The text layer contains no space characters at all**: word boundaries
exist only as horizontal gaps, so words are recovered by measuring the gap between glyphs against
the font size rather than by splitting on whitespace. And, as in the Liljegren ingest, characters
are read in content-stream order rather than sorted by position, because sorting detaches the
combining tone accents from their vowel. Each table is declared with its page, band, column
x-ranges and printed line count, and a table that does not yield exactly that many lines raises.
The 877-record extraction is committed as `20260825-hultman-kalkoti-extract.psv`; `--pdf FILE`
verifies the checksums, asserts that a fresh extraction reproduces the snapshot byte for byte, and
checks that every prose citation is still on the page it cites.

925 raw records become 497 installed rows. Twenty-one tables, all 129 numbered interlinear
sub-examples, the phonological illustrations of examples (2)–(4) and the prose citations are
installed; the Palula, Gawri, Pashto and Urdu comparanda are kept as `Etymology` prose rather than
as forms, and Table 1 (datasets), Tables 2 and 8 (phoneme inventories), Table 18 (bare inflectional
suffixes) and Tables 22 and 27 (sentences the numbered examples already carry) are excluded.
Repeated citations fold on tone-free shape plus gloss, so `driṣ` 'see' is one row carrying all its
locators. **Every interlinear row cites the dataset item it was recorded in** — `p. 22, example 5a,
U23a-28` — so a reader can trace any form back to its speaker and session; the four consultants and
eleven datasets stay provenance rather than becoming dialects.

Tables 2 and 8 print each phoneme beside its orthographic spelling in angle brackets, so the
thesis's phonemic and phonetic citations are rewritten into the same practical orthography its
tables use, with the IPA retained in `Phonemic`. The rewrite reproduces the thesis's own spellings
where both appear (`/měːɕ/` → `meéš`, `/tʰæ̌ːl/` → `thää́l`, `[ʈɒŋgʊɾ]` → `ṭangur`), treats a long
nasal vowel as the allophone of /Vːn/ that the thesis says it is (`[pæ̃ːs]` → `pääns`), and drops
glottalization and devoicing as phonetic detail. Glosses are split with the thesis's own
Abbreviations page as the label vocabulary, so no label is left unmapped; a word the thesis glosses
by category alone — the copula, the negator, the pronouns — is given the meaning that category
states, with a note recording that. Hultman makes no reconstruction and cites no CDIAL number, so
every row is installed unlinked, and the donor languages of Tables 4, 5 and 13 are recorded as
prose plus a `loanword` tag rather than as graph edges.

Both Kalkoti sources share `conversion/kalkoti.txt`; see the Liljegren section below for how it
writes tone. Hultman's `ràat` 'night' beside `raat` 'blood' is the minimal pair that makes carrying
tone into the transcription necessary rather than decorative.

#### Kalkoti (Liljegren 2013)

`data/other/forms/raw_data/liljegren_kalkoti_2013.py` re-extracts Henrik Liljegren's open-access
article *Notes on Kalkoti: A Shina Language with Strong Kohistani Influences* (Linguistic Discovery
11(1): 129–160; doi `10.1349/PS1.1537-0852.A.423`). It supersedes the hand-typed 2022 snapshot at
the same installed path, `data/other/forms/20220913-kalkoti.csv`, and keeps the `kalkoti` citation
key, so every form ID, alias and hand-assigned CDIAL etymology survives the change. Kalkoti has no
dictionary; this article and two survey wordlists are the whole published record, and the article
is a sketch, so its lexicon sits in twenty numbered tables, seventeen interlinear examples, and
phonetic citations in running prose.

The PDF carries an Acrobat text layer, so no OCR is involved. Two properties of the typesetting
drive the extraction. First, every lexical table is a fixed grid: `GRIDS` records each table's
page, vertical band, column x-ranges and printed row count, and a table that does not yield exactly
that many rows raises, so a missed page break or column cannot pass silently. Second, characters
are read in **content-stream order rather than sorted by position**: Kalkoti tone is written with
combining grave and acute accents that Acrobat emits on their own baseline, drawn over the vowel
*after* the one they belong to, so any positional sort moves the tone onto the wrong mora. The
319-record extraction is committed as `20260825-liljegren-kalkoti-extract.psv`; `--pdf FILE`
verifies the file against its SHA-256 and SHA-512, asserts that a fresh extraction reproduces the
snapshot byte for byte, and checks that every prose citation is still on the page it cites. The PDF
is not redistributed.

365 raw records become 276 installed rows. Only the Kalkoti column is installed: the Palula, Gawri,
Sawi and Kohistani Shina comparanda are secondary citations of Baart (1997, 1999a), Buddruss
(1967), Liljegren (2008) and Schmidt & Kohistani (2008), which Jambu ingests from those works
directly, and they are kept verbatim in the audit's `Raw_Context`. Also excluded are the phoneme
inventories of Tables 7 and 10, the comparative tense/aspect Tables 15 and 21–24, and example (9),
which is Biori Palula. Repeated citations of one lexeme fold on form plus gloss, so `pitri`
'father's brother' is one row carrying its kinship-table, cluster-table and footnote-15 locators
together with its printed IPA, while `raat` 'blood' and `raat` 'night' stay apart.

The article prints Tables 8, 11 and 12 and the phonetic prose in IPA and everything else in the
broad transcription customary among Shina scholars. Because Tables 7 and 10 state the
correspondence between the two notations explicitly, the IPA is rewritten into that same broad
transcription and the printed IPA is kept in `Phonemic`; the rewrite reproduces the article's own
spellings where both appear (`/eːʂ/` → `eeṣ`, `/treːr/` → `treer`, `/drɑːm/` → `draam`). A
parenthesised `(ʔ)` is the prosodic glottal element of melodies 3 and 4 and is dropped; every other
parenthesised segment is a consonant some speakers drop and is kept.
`conversion/kalkoti.txt` then converts that broad transcription to house transcription,
and reproduces the 2022 hand-typed spellings exactly (`bään` → `bǣn`, `pheep` → `pʰēp`, `ic̣ii` →
`iʦ̣ī`).

**Tone is carried into the transcription.** `conversion/kalkoti.txt` extends the
convention `conversion/liljegren.txt` already uses for Palula accent: an acute on the first mora is
a falling contour (`šáak` → `śā̂k`) and an acute on the second a rising one (`taár` → `tā̌r`).
Kalkoti additionally has a marked low tone, written with a grave; Hultman (2023, p. 17) states that
it is a property of the whole syllable with no /V̀V/ versus /VV̀/ contrast, so both writings give a
single grave (`ḍä̀är` → `ḍǣ̀r`, `šaàk` → `śā̀k`), and grave plus acute is the low-rising contour
(`ɡòór` → `gō̌̀r`). Tone also appears on short vowels (`bä̀kaál` → `bæ̀kā̌l`). Only some of the
article's tables mark tone, so citations are folded on their tone-free shape and the row keeps the
marked spelling: `taár` in Table 13 and `taar` in Table 9 are one row, written `taár`. The tone
class itself is also kept as a typed note (`Liljegren analyses this word as carrying a high tone`).
Without this, minimal pairs such as Hultman's `ràat` 'night' beside `raat` 'blood' would collapse.

Grammatical labels become canonical tags: the pronoun paradigms of Tables 4 and 14 carry person,
case and deixis, the aspect pairs of Table 2 split into two rows apiece, and the two verb classes
the article names after Palula's are recorded as `Kalkoti-verb-class-L`, `-T` and `-suppletive`.
Interlinear glosses are split into a lexical definition and tags, with the polar-question clitic
`=ää` left off the headword and kept as `interr`. Table 13's eight printed Turner (1966) numbers
are linked directly; every other CDIAL link is an editorial decision carried over from the 2022
snapshot through `20260825-liljegren-kalkoti-etymology.csv`, which maps all 135 of them onto the
new `Entry_Key` values and fails the build if re-extraction drops one. Four comparanda in other
languages that the 2022 file used to anchor Jambu's own etyma `e34`–`e36` are carried through
verbatim and stay on the preservation profile. Re-extraction also corrects three readings the
article disproves: `im` is 'snow', not 'belly', and 'dust' and 'bone' have short vowels.

#### Sauji (Knobloch 2020)

`data/other/forms/raw_data/knobloch_sauji_2020.py` installs every glossed Sauji form in Nina
Knobloch's open-access MA thesis *A grammar sketch of Sauji* (Stockholm University; DiVA
`diva2:1440556`). The thesis is one of only four substantial descriptions of this Shina language of
Sau in Kunar, Afghanistan, and it is a grammar sketch rather than a dictionary, so its lexicon is
spread across numbered tables, interlinear examples and italicised citations in running prose.

The PDF carries a genuine LaTeX text layer, so no OCR is involved. The importer reads pdfplumber
word boxes and keys on the structure the thesis actually uses: object-language material is italic
and glosses are roman, each interlinear gloss sits at exactly its word's `x0`, and every table has
fixed column positions. Thirteen regions are extracted — Tables 2, 4, 6, 7, 8, 9, 10, 11 and the
Sauji column of Table 14, the syllable-structure examples in (1), all interlinear examples
including the three appendix texts, the italic-plus-quoted-gloss citations in prose, and a small
checked-in table of prose citations whose gloss precedes the form. The 906-record extraction is
committed as `20260825-knobloch-sauji-extract.psv`; `--pdf FILE` verifies the file against the
SHA-256 and SHA-512 published in the DiVA record and asserts that a fresh extraction reproduces
that snapshot exactly. The PDF itself is not redistributed.

926 raw records become 573 installed rows. Repeated attestations of one lexeme are folded on form
plus gloss, so `aw` 'and' is a single row carrying all 24 of its locators, while homographs such as
*si* 'bridge', 'together' and 'together with' stay apart. Interlinear glosses are split into a
lexical definition and canonical tags (`house-obl` becomes gloss *house* with `noun obl`), the
author's own hedges become `uncertain` rather than invented definitions, and the four verb and
three noun inflection classes the thesis assigns are recorded as `Sauji-verb-class-*` and
`Sauji-noun-class-*`. The thesis makes no etymological claim about individual words, so every row
is installed unlinked; the donor languages named in section 5.2.3 are kept as prose plus a
`loanword` tag rather than as graph edges. Forms Knobloch reproduces from Buddruss (1967) carry his
own citation alongside hers.

`conversion/knobloch-sauji.txt` reads both notations the thesis prints — broad IPA in the phonology
tables (retained in `Phonemic`) and the simplified Indo-Aryanist transcription everywhere else —
because the two never disagree about a grapheme. Source-side slips are repaired with the reason in
the audit: an unclosed gloss quote on p. 13, quotes printed around the form rather than the gloss
in the same table, ASCII colons for length, and one IPA script-g.

#### Zargari (Rezai Baghbidi 2003)

`data/other/forms/raw_data/rezai_baghbidi_zargari_2003.py` installs the Zargari lexical material in
Hassan Rezai Baghbidi's *The Zargari language: An endangered European Romani in Iran* (Romani
Studies 5th ser. 13/2: 123–148), based on his own fieldwork in Zargar village in 2000–1. The
article is a grammatical sketch rather than a dictionary, so its lexicon lives in glossed examples
inside numbered sections plus a handful of glossed list and table blocks. The publisher PDF is not
redistributed; the importer requires it, verifies its SHA-256, and refuses to run without it.

Extraction reads the publisher's Type 1 text layer, not OCR — but decodes it by `/Differences`
glyph name rather than through the embedded `ToUnicode` map, which silently drops the first element
of the `T_h`, `f_i`, `f_l`, `f_f` and `f_f_i` ligatures ("The" reads as "he", "field" as "ield") and
mangles oldstyle figures, small capitals and the Indological `ṭ`/`ṣ` and composite accent glyphs.

Every single-quoted gloss span in a numbered section is a source record. All 575 of them, plus 42
records attached to a span (a printed plural, a gendered counterpart, a perfect stem, a listed
synonym) and 89 entries from the glossed list and table blocks, carry an explicit status in
`data/other/forms/raw_data/20260825-rezai-baghbidi-zargari-audit.csv`: 444 ingested, 68 repeated
mentions folded into an existing record, and 194 excluded as clause examples, phrase examples,
metalinguistic prose, donor forms, or Hindi/Sanskrit/Qorbati/Seliyeri comparanda. The 444 ingested
units become 522 installed rows once printed alternates are expanded into `Variant_Of_Key` rows.

The editorial line is that the article's own glossing decides what is lexical: every isolated
Zargari word it glosses is installed, and a multi-word item only when it is printed inside a
lexical list (tagged `multiword-expression`). Clause and phrase examples, and the paradigm tables
that print no glosses at all — case suffixes, personal endings, personal pronouns, demonstratives,
possessives, reflexives, the copula, and the mediopassive and Azari Turkish loan conjugations —
stay out of the installed CSV and remain accounted for in the audit. Syllable dots and the
non-phonemic glottal onset the author himself calls unnecessary are dropped from `Form` while
printed stress acutes are kept; optional parenthesised segments (`bax(t)`, `(ā)kātu`, `ām(m)ā`)
become explicit head plus alternate rows.

Zargari is its own base language (`Zarg`) with Glottocode `zarg1238`; Zargar village survives as
its `zargari` dialect tag, carrying the village's own printed coordinates (36° 03′ N, 50° 23′ E,
quality `A`). The article prints no CDIAL, DEDR or other etymon identifier, so every row
is installed unlinked; its Azari Turkish, Persian, Arabic, Greek, Armenian, Early Romani, Hindi and
Sanskrit comparisons are kept as `Etymology` prose plus a `loanword` tag, never as graph edges.
Because the sketch cites the same shape in several sections under genuinely different analyses —
*ruv* 'wolf' beside the imperative *ruv* 'cry!', the adverb/postposition pairs *opro*, *teli*,
*ānglo*, *ānvro*, *bāšu*, *pālo*, *anvri* and *sar*, and *kāšt* 'wood' beside *kāšt* 'tree' — its
immutable span keys take part in the compiled deduper's key so those homographs survive the build.

`conversion/zargari.txt` maps the article's Persianist/Romanist transcription onto house
conventions: `j` is the palatal glide and becomes `y`, `dž` the voiced palatal affricate and becomes
`j`, `č`/`čh` become `c`/`cʰ`, `š`/`ž` become `ś`/`ź`, `γ` becomes `ɣ`, and the aspirate digraphs
become superscripts.

#### Gondi dialect survey (Beine 1994 / Rama et al. 2017)

`data/other/forms/raw_data/gondi_beine.py` installs the complete 46 sites x 210 concepts IPA word
list that Rama, Çöltekin and Sofroniev digitized from David K. Beine's 1994 San Diego State
University master's thesis, *A sociolinguistic survey of the Gondi-speaking communities of central
India*. The source is the supplementary repository
[PhyloStar/Gondi-Dialect-Analysis](https://github.com/PhyloStar/Gondi-Dialect-Analysis), pinned at
release `v1.0` (commit `f24fc74`, DOI `10.5281/zenodo.1220088`, licensed "Other (Open)" on Zenodo);
the importer verifies the SHA-256 of each upstream file it reads and fails loudly on drift, so the
snapshot itself is not committed. It is not an OCR ingest: the release is born-digital TSV.

Of the 9,660 source cells, 158 are printed `-----` (no word elicited at that site) and stay in the
audit; the remaining 9,502 become 10,264 installed rows because 762 cells list two or three
responses. Those responses are emitted as independent rows with their own `Entry_Key`
(`beine:<site>:<concept>:<n>`) rather than as `Variant_Of_Key` chains, since Beine's alternates are
frequently distinct lexemes — *kʰarab* beside *beshile* for 'bad' — not spellings of one word. The
source prints no etymological identifier of any kind, so every row is installed unlinked. Because
the same site answers 299 unrelated prompts with the same shape (*puro* 'above'/'all' at grp, *pir*
'belly'/'rain' at rui), the citation key takes part in the compiled deduper's key so those
homographs survive the build.

All 46 sites are registered in `cldf/dialects.csv` beneath the existing `Gondi` base language, using
the digitizers' own geolocation of Beine's site descriptions from `maps/gondi.kml` (quality `B`; no
Glottocode, because the sites are not Glottolog languoids). Each `Location` carries the source's
full site description — with its historical district and state names, so Bastar is recorded under
Madhya Pradesh and Adilabad under Andhra Pradesh as Beine wrote them — plus the five-way Glottolog
subgrouping of table 1 of the paper (Northwest Gondi > Northern/Southern; Southeast Gondi > Hill
Maria, Muria, Bison Horn Maria). Those subgroups are recorded as prose rather than as clades so the
frontend clade files stay untouched.

`conversion/gondi-beine.txt` maps Beine's IPA onto Jambu's Dravidianist house transcription with
complete coverage of the 184 attested graphemes. The one linguistically consequential decision is
that the source marks dentality only sporadically: `t̪`/`d̪`/`n̪`/`s̪` alternate freely with plain
`t`/`d`/`n`/`s` within a single site and even inside one cognate set (*wort̪itor* beside *wortitur*
for 'eat'), and Gondi has no third coronal series, so both spellings render as the house dental
while `Original` and `Phonemic` keep the source's own diacritics. `w`/`ʋ`/`v` likewise collapse to
`v` and `j`/`y` to `y`; `ː` becomes a macron on a vowel and a doubled letter on a consonant; vowel
qualities (`ʌ`, `ə`, `ɛ`, `ɪ`, `ɔ`, `ʊ`, `ɨ`, `ɵ`, `ɤ`) are carried over unreinterpreted, as are
half-length `ˑ`, nasality, non-syllabic marks and printed parentheses.

Two parts of the release are deliberately **not** installed and remain in the audit
(`data/other/forms/raw_data/20260825-gondi-beine-audit.csv`, one row per source cell plus one per
expanded response): the LingPy-derived ASJP and SCA recodings, which are computed from the same
IPA; and Taraka Rama's `Cognate Class` judgments. The latter are Gondi-internal and scoped to a
single concept, so representing them in the graph would mean minting roughly 700 headword-less
Proto-Gondi grouping nodes on the Proto-Burushaski pattern. That is a schema decision rather than a
parsing one, so every class label is preserved per record and nothing here asserts an edge the
source does not print. The Nexus matrices, MrBayes consensus trees and autoencoder code are
analysis results rather than lexemes and are out of scope.

#### Bhatri dialect survey (Beine 2017, SIL ESR 2017-005)

`data/other/forms/raw_data/beine_bhatri.py` installs Appendix A of Dave Beine's *A Sociolinguistic
Survey of the Bhatri-speaking Communities of Central India* (SIL Electronic Survey Report 2017-005,
[archive 71330](https://www.sil.org/resources/archives/71330)) — the complete 12 sites x 210
concepts comparative word list on printed pages 10--33. The fieldwork is from February--November
1989 in Bastar District, Madhya Pradesh (now Chhattisgarh) and Koraput District, Orissa (now
Odisha); the report was published unrevised in 2017. The importer verifies the PDF's SHA-256 and
fails loudly if the uncommitted file is absent or has drifted.

This is not an OCR ingest: the 2017 PDF carries a positioned CharisSIL text layer. It is not a
table, though, and ordinary extraction garbles it — column x-positions shift on printed page 13,
and combining marks are emitted on their own y-band. Each line is rebuilt by bucketing every
character onto the nearest dominant baseline and concatenating in *content-stream* order, the only
order that keeps a mark next to its base, and is then split on the twelve fixed uppercase site
codes. All 2,520 cells were cross-checked against an independent extractor (`pypdf`); the single
disagreement is the source typo noted below.

Of the 2,520 printed cells, 74 print an en dash (no word elicited) and two Halbi cells print a
stray combining diaeresis and no word at all; all 76 stay in the audit as `missing`. The remaining
2,444 become 2,492 installed rows, because 48 cells print two responses separated by the source's
`ʔ` (*mutⁿ* ʔ *pise̠b* for 'urine', *bato* ʔ *ṛastṛa* for 'path'). Those are distinct lexemes, so
they are emitted as independent rows with their own `Entry_Key` (`beine-bhatri:<site>:<item>:<n>`)
rather than as `Variant_Of_Key` chains. The key is qualified with the work because Beine's Gondi
lists already own the bare `beine:` namespace. The survey prints no etymological identifier, so
every row is installed unlinked and the source contributes no graph edges. As in the Gondi lists,
the citation key takes part in the compiled deduper's key, so the 76 same-lect homophones elicited
under different prompts survive the build as separate records (OAR *nak* is both 'nose' `nāk` and
'nail' `nakh`; OAR *aṭ* answers 'arm', 'week' and 'eight').

All twelve lects are installed, not only the nine Bhatri points: the three lects the survey tested
intelligibility against are elicited field word lists from named localities exactly like the rest.
`Bhatri` (`bhat1265`, Halbic — not `bhat1263`, which is Bhateali) and `AdivasiOriya` (`adiv1239`,
Eastern) are new base languages; Halbi from Bhatpal is filed under the existing `hal` and Oriya
from Cuttack under `Or`. All twelve survey points are registered in `cldf/dialects.csv` with the
printed site description in `Location`, keeping Beine's 1989 district and state names. Five
localities resolve to an OpenStreetMap point in the district the source names (quality `B`); the
other seven fall back to the tahsil or district headquarters Beine himself names and are marked
quality `C`, with the reason recorded per site in `data/dialect-coordinate-decisions.csv`. Beine's
Halbi site and Fran Woods' Halbi dictionary both say "Bhatpal" but name different tahsils, so they
are kept as separate dialects and the discrepancy is recorded rather than silently merged. The
`Bhatri` base-language point is a centroid of its nine survey sites and is marked quality `C`.

`conversion/beine-bhatri.txt` maps what the report calls "a modified International Phonetic
Alphabet" onto Jambu's house transcription. No key is printed, so every value was established
distributionally across the 2,520 cells, against the Indo-Aryan etyma, and against rendered page
images: `ˑ` after `t d n r l` is retroflexion (*tˑondˑ* 'mouth' → *ṭonḍ*); `ⁿ` after `t d` is
dentality, marked only sporadically (*hat*, *hatⁿ* and *atⁿ* all appear for 'arm'), so both
spellings give the house dental while `Original` and `Phonemic` keep the source's diacritics; `̽`
is a nasal vowel; `̂` is aspiration on `h` and non-syllabicity on a vowel (*soîla* beside *soila*
'sleep'); and an under-bar marks a look-alike letter used as a vowel symbol, `e̠` → `ə`, `v̠` → `ʌ`,
`c̠` → `ɔ`. The under-tick is retroflex on `r` (*r̩* → `ṛ`) but obligatory on `ʃ` and `ʒ` (186/186
and 146/146), where it therefore carries no information, so `tʃ̩` → `c` and `dʒ̩` → `j` while the
four Oriya tatsamas printing `ʃ̩` alone give `ṣ`. `.̩` is a sporadic juncture mark, not a segment
(*bol.̩a* beside *bola* 'speak').

Two residues are deliberately left uninterpreted and their 23 rows are tagged `uncertain`: the
under-tick on a vowel (*e̩k* beside *ek* 'one'), which is kept verbatim, and `ɵ̩`, printed in
exactly two cells where 'flower' and 'fruit' require /pʰ/ and rendered `ɸ`. Three defects of the
2017 typesetting are repaired mechanically and recorded per record in the audit: `U+FFFD` in 24
cells is a dotless i whose ToUnicode entry is missing (the page renders *boı̽si* 'buffalo', *sı̽ɡ*
'horns'); printed page 11 labels one Cuttack Oriya row `OC` instead of `OCU`; and the English
prompts print `ɡ` for g, `ʏ` for y, `ɪ` for I and `ʡ` for `?`, with the parentheses of 'we (incl.)'
broken into stray combining marks.

Only Appendix A is installed. The report body, Appendix B (interlinearised narrative texts and
their comprehension questions), Appendix C (recorded-text-test score sheets) and the References are
running text or test results rather than elicited lexemes and are out of scope; the audit
(`data/other/forms/raw_data/20260825-beine-bhatri-audit.csv`) carries one row per printed cell plus
one per expanded response.

#### Romani etymological appendix (Boretzky & Igla 1994)

`data/other/forms/raw_data/boretzky_igla_1994.py` installs the appendix *Etymologien* of Norbert
Boretzky and Birgit Igla's *Wörterbuch Romani-Deutsch-Englisch für den südosteuropäischen Raum*
(Wiesbaden: Harrassowitz, 1994), printed pages 311--338. The appendix is four alphabetical word
lists — `Indische Etyma` (pp. 311--328), `Iranische Etyma` (pp. 329--331), `Armenische Etyma`
(pp. 331--332) and `Griechische Etyma` (pp. 333--338) — each entry giving a Romani headword, an
italic grammatical label, a German gloss and a bracketed etymological note. The A--Z dictionary
proper, the two reverse indexes and the *Variantengrammatik* are not ingested.

The scanned volume is copyrighted and is not redistributed; it carries no text layer. Two Tesseract
passes over 400 dpi renders were used only for navigation and for discrepancy discovery, and every
printed entry was then read off the page images, because the OCR does not distinguish the source's
`č`/`ć` or `ž`/`ź` and does not read its `ř` or `ə` at all. The resulting hand-collated
transcription is checked in as
`data/other/forms/raw_data/20260825-boretzky-igla-etymologies-extract.psv`, one row per printed
entry with the source German gloss and an editorial English gloss in separate columns; the importer
reads that file and fails loudly if it is absent.

1093 printed entries expand to 1259 audit rows and 1140 installed rows. A printed entry may bundle
several headwords, so comma-separated alternates become variant rows linked by `Variant_Of_Key`
while a bundled distinct lexeme (`salo m Schwager; sali f Schwägerin`) becomes its own row; 34
lexemes listed in two appendices at once are merged into a single row that keeps both page locators
and both etymological analyses; and the 85 `s. X` cross-reference lines are pointers to a full entry
elsewhere in the appendix, so they are accounted for in the audit rather than installed.

Every form belongs to European Romani (`eur`), which is what the appendix covers — it deliberately
includes "alte Etyma aus allen Dialekten" and cites each word in the shape it has as a main entry in
the dictionary. Entries printed with an explicit dialect-group label take that group's canonical base
language instead: `(Sinti)` → `RomSint`, `(Arli)` and `(Bug.)` → `RomBalk`, `(Urs)` → `RomVlax`,
`(Caló)` → the registered Spanish-Romani dialect of `eur`; Arli, Bugurdži and Ursari are added to
`cldf/dialects.csv` as quality-`C` approximations. The appendix's other parenthesised labels — `(Sa)`,
`(So)`, `(Thes)`, `(Rozw)`, `(Col)`, `(Finck)`, `(Ješ)`, `(Lípa)`, `(Bar)`, `(Paspati)`, `(Bischoff)`,
`(Heinschink)` — are bibliographic, not dialectal: p. 311 states that a source is named for words
absent from the authors' own dictionary, so each becomes a secondary CLDF citation with its own
`cldf/sources.bib` record.

`conversion/boretzky-romani.txt` follows the conventions the Zargari and CDIAL Romani profiles
already use (`č` → c, `čh` → cʰ, `dž` → j, `š` → ś, `ž` → ź, `j` → y) and adds the second affricate
series this source contrasts with them (`ć` → ʨ, `ćh` → ʨʰ, `dź` → ʥ), plus `c` → ʦ, `ř` and `ə`.
Bound forms keep their hyphens, so the profile is routed through
`PRESERVE_SOURCE_PROFILE_INPUT`.

The etymological brackets are scholarly prose and are preserved verbatim in `Etymology`, labelled by
the list they come from. Only the Indic list can produce a `Parameter_ID`: the Old Indo-Aryan form
the bracket cites is matched against CDIAL headwords under a normalisation that folds Vedic accents,
reconstruction asterisks, morpheme hyphens, homonym numbers and Turner's `ē`/`ō`, but never vowel
length, retroflexion or sibilant quality. A link is taken only when that match is unique and the
source is actually asserting the etymology, which needs care: Boretzky & Igla routinely name an
etymon in order to reject it (`ai. śīrṇa- ... paßt lautlich nicht`), reject one clause before
proposing another in the next (`... gehören ... nicht dazu; eher < pa. garahati < ai. garhati`), and
offer two etyma they do not choose between (`ai. kṣuri-/churī`, `< ai. dhāpayati ... oder < ai.
sthāpayati`). Rejection is therefore scoped to the semicolon-delimited clause, and a clause naming
more than one candidate links none of them. Of 1140 rows, 286 carry a CDIAL link and the audit
records the outcome for every other one: 599 with no OIA etymon cited, 105 ambiguous, 101 unmatched,
32 source-alternatives, 17 rejected by the source. The Iranian, Armenian and Greek lists assert
borrowing from donors Jambu does not carry as nodes, so those claims stay in prose and the rows are
installed unlinked.

Per-record provenance is in `data/other/forms/raw_data/20260825-boretzky-igla-etymologies-audit.csv`
(one row per emitted headword, with the raw printed fields, the cited etymon and its match status)
and a seeded 20-entry extract in the `-sample.csv` beside it; regression tests are in
`tests/test_boretzky_igla.py`.

#### Gilgit Shina riddle glossary (Buddruss 1996)

`data/other/forms/raw_data/buddruss_shina_1996.py` installs the complete analytical glossary
(pp. 40–50) of Georg Buddruss's “Shina-Rätsel,” in Dieter B. Kapp (ed.), *Nānāvidhaikatā:
Festschrift für Hermann Berger* (Wiesbaden: Harrassowitz, 1996), pp. 29–54. The 31-page Stanford
ILL scan (request 446828, supplied by Cornell/Olin) is copyrighted and is not redistributed; the
importer can verify its SHA-256 on demand. The embedded PDF text layer was useful for navigation
but systematically mis-maps several specialist characters, so the 296 analytical headword units
and their 15 additional headline alternates were collated manually against 300 dpi renders.

Every glossary headword is installed, including `wáaku`, which Buddruss explicitly calls an
unintelligible word in riddle 12 and which is therefore retained with an `uncertain` tag. Explicit
headline alternates become `Variant_Of_Key` rows, and the reciprocal `agúl = hagúl` references are
represented by making `hagúl` a variant of `agúl`. Feminine, plural, case, finite, and participial
examples remain prose rather than becoming forms. Excluded from the lexical table are the 58
running riddles and translations, the glossary preface, comparison-only non-Shina forms, the
bibliography, and the closing summary.

Turner/CDIAL links follow the source conservatively: only direct, unambiguous assignments are
linked. Competing numbers (`ai` “goat” with T. 145 and 887), questioned comparisons, IDs cited only
for a compound component, and rejected alternatives remain unlinked editorial prose. The decimal
T. 7934.2 on `phapaáo~` links to its integer CDIAL parent 7934. All forms use canonical Shina (`Sh`)
with the registered Gilgit dialect tag (`dialect:Sh:gil:Gilgit`).

`conversion/buddruss-shina.txt` preserves Buddruss's double-vowel quantity and tone system while
mapping it to house transcription: `aá` is rising, `áa` falling, and nasal marks are encoded in the
checked-in source layer with `~` immediately after the vowel sequence. The manifest, full audit,
and deterministic 25-row review sample are beside the importer.

#### Gilgit Shina glossary (Degener 2008)

`data/other/forms/raw_data/degener_shina_2008.py` installs the complete glossary (pp. 243–315) of
Almuth Degener's *Shina-Texte aus Gilgit (Nord-Pakistan): Sprichwörter und Materialien zum
Volksglauben, gesammelt von Mohammad Amin Zia* (Beiträge zur Indologie 41, Wiesbaden:
Harrassowitz, 2008), the reference lexicon to Mohammad Amin Zia's collection of 780-odd Gilgit
Shina proverbs and folk-belief texts. The two Stanford ILL scans that together cover the glossary
(requests 446831 and 447377) are not redistributed; the importer verifies their SHA-256s on demand.

The checked-in raw layer is `20260827-degener-shina-transcription.txt`, a complete
paragraph-by-paragraph verbatim transcription of the glossary produced by manual collation against
300 dpi renders with zoomed band verification (Tesseract was used only for per-page digit
cross-checks; every page passes that check). Headword paragraphs, indented attestation/inflection
sub-paragraphs, and page-break continuations are marked distinctly, and a deterministic 25-entry
sample from the fully reviewed census has 0/25 material errors. The one mechanical normalization is
documented in place: the italic font draws the haček of `ǰ` as a rounded arc, which one
transcription pass had recorded as a breve. Sixteen uncertain readings across fifteen headword
records (mostly Burushaski and Indus Kohistani comparanda, plus one inflected Shina form retained
only as audit prose) remain flagged `⟦…⟧` in the raw layer; the fifteen installed headword records
carry an `uncertain` tag.

All 1561 printed headword paragraphs are accounted for: 1521 install directly, 32 printed
cross-reference headwords resolve to a unique target and install as its variants, and 8
cross-references whose printed target is a homonym pair (or a two-target listing) stay audit-only.
Headline alternates (`maphéer, maféer`; `kháčo/ kháči th-`) become `Variant_Of_Key` rows; light-verb
constructions (`ẓan th-`, `bal b-`) are single headwords. English glosses were translated
editorially from Degener's German and live in `20260827-degener-shina-editorial.csv` alongside
per-entry resolutions; the raw German gloss is preserved on every row in `Notes`, and attestation
numbers and inflected forms remain audit prose, following the Buddruss Waigali/Wama precedent.

Degener's bracketed etymologies are linked conservatively: an entry links to CDIAL only when it
prints exactly one unhedged, stand-alone `T. N` claim, wherever it stands in the bracket
(`[Bur. oq, T. 2538]` links; `T. 145, 887`, `zu T. 6298`, `vgl. T. 5055`, and `T. 14154?` stay
prose), yielding 529 linked rows. A printed decimal sub-number such as `T. 11503.3` links to its
integer CDIAL parent with the printed form preserved in the etymology prose. Loan etymologies
(`← Ar.-Ur.`, `← Pers.-Ur.`, `← Tib.`) keep their arrow prose plus a `loanword` tag; Burushaski
and Indus Kohistani comparanda remain prose and are candidates for the Proto-Burushaski
comparison layer in a later pass.

All forms are canonical Shina (`Sh`) under the registered Gilgit dialect tag
(`dialect:Sh:gil:Gilgit`); Zia's collection and the informants remain provenance.
`conversion/degener-shina.txt` maps Berger's orthography as used by Degener onto house
conventions: doubled vowels become macrons, with the mora position of the acute mapped onto the
CDIAL-style accents (`aá` rising → `ā́`, `áa` falling → `ā̀` — a mapping that merits linguistic
review), the spacing nasalization tilde becomes the combining tilde, and the affricates map as
`ċ` → `ʦ`, `č`/`čh` → `c`/`cʰ`, `c̣`/`c̣h` → `ʦ̣`/`ʦ̣ʰ`, `ǰ` → `j`.

#### DBIA

`data/dbia/parse.py` conservatively extracts Emeneau and Burrow's *Dravidian Borrowings from
Indo-Aryan* into `forms.csv` and `params.csv`. It preserves each complete OCR entry, records
two-column boundary repairs and match decisions in `parse_audit.csv`, and writes high-confidence
normalized headword matches to `cdial_redirects.csv`. During unification, matched `dbiaN` entries remain as
resolvable redirect stubs while their Dravidian loans and source transcription are folded into the
canonical CDIAL entry.

#### Munda

`data/other/forms/raw_data/pinnow_munda_1959.py` imports the structured SEAlang index of
Heinz-Jürgen Pinnow's *Versuch einer historischen Lautlehre der Kharia-Sprache* (1959).
The snapshot contains 3,340 stable HTML records across fourteen Munda language labels and uses
semantic fields for source IPA, gloss, language, and database ID; extraction therefore requires
no OCR. Top-level comma alternants produce 4,051 installed rows; the sole record explicitly
marked `MISSING` and one exactly repeated alternant remain audit-only. Historical `Bodo-Gadaba` and `Bondo` labels map to canonical
Gutob and Remo; Asuri, Birhor, and Turi are independently registered languages. The source gives
no locality or dialect field, so no site distinction is invented.

Pinnow's 553 source set labels are preserved on every record. Proto-Munda linkage is deliberately
narrower: a form is attached to a Rau etymon only when Rau's 2019 table prints the same Pinnow set
number. Duplicate Rau cross-references (`V3`, `V278`) are disambiguated only where the source form
or gloss uniquely selects a branch; three mixed hill/forest alternants remain unlinked. The
identity profile `conversion/pinnow-munda.txt` preserves the Unicode source transcription in both
display and `Phonemic`. The complete per-variant audit, deterministic 20-record review sample, and
manifest sit beside the importer; `--offline --install` rebuilds from that checked-in audit.

`data/other/forms/raw_data/munda_proto_kherwarian_1968.py` imports the born-digital
SEAlang index of Ram Dayal Munda's *Proto-Kherwarian Phonology* (1968). Its 2,768
semantic HTML records comprise 920 Proto-Kherwarian reconstructions, 923 reconstructed
pre-Mundari forms, and 925 Santali forms; splitting only source-level spaced-tilde
alternants yields 920 parameters and 2,919 form rows without OCR. Proto-Kherwarian and
Pre-Mundari are registered as historical/reconstructed Munda stages without invented
Glottocodes or coordinates.

Reflex alignment requires a shared source locator and gloss. Eleven damaged locator
matches are recovered only where the exact gloss selects one reconstruction, and two
further shifted records have uniquely compatible forms and meanings recorded explicitly
in the audit. Three comparison records lacking any source reconstruction remain installed
and unlinked. Slash alternatives and optional segments are preserved as printed; the
identity profile `conversion/munda-proto-kherwarian.txt` checks the complete Unicode
inventory. A full audit, deterministic 20-record sample, manifest, and offline rebuild are
checked in beside the importer.

`data/other/forms/raw_data/zide_sora_gorum_1982.py` imports the structured SEAlang
index of Arlene R. K. Zide's *A Reconstruction of Sora-Gorum Morphology* (1982).
The 1,750 records contain 953 Sora and 797 Juray entries in 1,011 stable comparison
groups; top-level comma and semicolon alternants yield 2,057 installed rows without
OCR. Juray is registered independently from Juang using Glottocode `jura1242` and a
quality-C modern Glottolog coordinate.

The indexed view exposes comparison forms but no protoforms. Its 739 two-lect groups
and 272 singletons are therefore preserved in `Cognateset`, while every row remains an
unlinked graph node: neither a Sora–Gorum reconstruction nor Proto-Munda ancestry is
invented. The identity profile `conversion/zide-sora-juray.txt`, complete audit,
deterministic 20-record review sample, manifest, and offline rebuild preserve and test
the source's Unicode transcription and stable identifiers.

`data/other/forms/raw_data/bhattacharya_bonda_1968.py` imports the born-digital
SEAlang index of Sudhibhushan Bhattacharya's *A Bonda Dictionary* (1968). All 2,881
records are accounted for: 2,716 Plains Bondo and 165 Hill Bondo records expand to
3,330 installed form rows, while one exactly repeated alternant is retained audit-only.
Both named varieties are registered as dialects of canonical Remo (`re`); the source
index supplies no locality, so no coordinates are invented.

The importer separates 550 explicit `ETY:` note segments from dictionary commentary without
promoting abbreviated comparisons into unsupported graph ancestry or borrowing edges.
Twenty-seven uniquely recoverable printed `see` references become source-internal
variant links. Eight absent, malformed, or ambiguous targets remain unlinked and
glossless, and one multiple-target reference receives only the targets' shared lexical
definition. Question mark is preserved as a transcription symbol; only three terminal
`(E?)` provenance/query markers are removed from `Form`, retained in the audit, and
tagged `uncertain`. The identity profile `conversion/bhattacharya-bonda.txt`, complete
audit, stratified 20-record review sample, manifest, and offline rebuild preserve these
decisions without OCR.

`data/other/forms/raw_data/bahl_korwa_1962.py` imports the born-digital SEAlang
index of Kali Charan Bahl's unpublished *Korwa Vocabulary* (1962). Of 1,792 stable
source records, one empty row remains audit-only; the 1,791 lexical records yield
1,830 installed Korwa forms after 39 comma-separated alternants are split. The index
labels every record simply Korwa, so forms use canonical `kw` without an invented
dialect or locality.

The full dictionary supersedes 57 hand-entered BAHL excerpts from Rau's Proto-Munda
dataset only where a unique normalized source form also has a compatible meaning.
Those source records retain their Proto-Munda parameter links and exact keyed locators;
the one two-form entry *goej, goeˀ* 'to die' yields two m51-linked variants. Ten Rau
citations whose form is absent or whose dictionary meaning conflicts remain as separate
legacy evidence. Explicit `Cf.`, component, Santali, and Hindi comparisons are preserved
as source etymology prose without new graph edges, while usage/editorial notes remain
notes and six queried records are tagged `uncertain`. The identity profile
`conversion/bahl-korwa.txt`, complete audit, stratified 20-record review sample,
manifest, and offline rebuild preserve these decisions without OCR.

`data/other/forms/raw_data/pinnow_juang_1960.py` imports the born-digital
SEAlang index of Heinz-Jürgen Pinnow's unpublished *Beiträge zur Kenntnis der
Juang-Sprache* (1960). Its 1,658 stable Juang records yield 1,818 installed
forms after comma-separated alternants are split; six exact within-record
repetitions remain audit-only. The source gives no dialect or locality field,
so every form uses canonical Juang (`ju`) without invented site metadata.

Sixty-six keyed records replace legacy PJDW excerpts only after a unique
normalized form and compatible meaning identify the source entry. They yield
72 linked variants; seven absent, conflicting, or source-glossless Rau claims
remain as separate legacy evidence. Hash-prefixed comparisons and explicit
`Cf.` prose are preserved as source etymology commentary, not promoted to new
graph edges. The 185 entries printed with no gloss or only `?` remain
intentionally glossless, while terminal `??` form markers are retained in the
audit, removed from display forms, and tagged `uncertain`. Two terminal Elwin/source
markers move to Notes so their internal commas cannot become false forms. The identity profile,
complete audit, deterministic 20-record review, manifest, and offline rebuild
make the no-OCR import reproducible.

- `data/munda/forms.csv`
- `data/munda/params.csv`

#### Others

- `data/others/forms/*.csv`: New lemmata extracted from various sources.
- `data/others/params/*.csv`: New entries.

### Approved Kalkoti donor supplement (2026-09-09)

The 43 donor heads approved in Kalkoti batch 17 are reproducibly emitted by
`data/other/params/raw_data/kalkoti_donors.py` from its source-attributed audit.
They support 68 analyses of 84 existing Kalkoti records in the curated assignment overlay,
including component and derived relations. This is a selected OPED/Platts/Hultman/editorial
supplement, not a complete dictionary import. Reviewed donor transcription is preserved with
NFC only; transmission qualifications, source variants and stable donor IDs remain in the audit.
See `source_checklists/20260909-kalkoti-donors.md` for scope, validation and outstanding global
suite failures. Browser refresh is separate and user-triggered.

### Shina and Brokskat donor-head supplement (2026-09-10)

`data/other/params/raw_data/sh_bro_donors.py` reproducibly emits 94 selected donor
heads and audits 29 reuses of the existing Shinaic supplement. The checked audit
preserves Degener's exact comparisons, CDIAL's Domaaki weaving donor, selected
Platts/Rekhta senses and two pinned OPED entries. These curated donor heads do
not by themselves decide unresolved borrowing routes. The 838 approved numbered
Shina/Brokskat analyses are saved separately in the etymology overlay. See
`source_checklists/20260910-sh-bro-donors.md` for source scope, transcription,
stable identities, successful donor-specific checks and the qualified global test results.

### Approved Dameli donor supplement (2026-09-10)

`data/other/params/raw_data/dameli_donors.py` reproduces 33 selected donor heads
(18 Pashto, 15 Urdu) from the adjacent source-attributed audit. OPED's archived
XML, Platts, Perder's explicit donor tables and seven existing survey attestations
supply the evidence. A rich self-attestation file preserves available native text
and grammatical tags; an explicit preservation profile prevents double conversion.
Another 22 donor heads are reused, supporting 114 approved overlay rows for 106
Dameli records, including six co-wife compounds. This is selected donor coverage,
not a full dictionary import. Uncertain transmission remains qualified.
See `source_checklists/20260910-dameli-donors.md` for source scope, excluded variants,
stable IDs, audit and validation limits: corrected full build blocked by disk space,
global tests not clean, original survey-page checks outstanding. No browser DB refresh.

### Mewari-reference donor supplement (2026-09-11)

`data/other/params/raw_data/mewari_donors.py` reproduces 13 selected dictionary
heads and audits 9 existing-head reuses. Platts (1884), Monier-Williams (1899),
and one Hindi cabbage cell in the CSTT agriculture glossary supply the lexical
evidence. The adjacent audit preserves native text, original spellings, sense
exclusions, and explicit transcription choices. This follows the curated
parameter-head pattern used for Pashto donors; no new survey attestations or
dialects are fabricated. The authorized overlay adds 94 links for 88 records
from reference `mewari`, including borrowed Sanskrit saptāha. Ambiguous responses
remain unlinked. See `source_checklists/20260911-mewari-donors.md` for validation.

### Mewari regional families (2026-09-11)

`data/other/params/raw_data/mewari_families.py` reproduces seven user-reviewed
regional lexical-family heads from existing Mewari and Marwari records. The
adjacent audit preserves all members and stable IDs; `data/etymologies.csv`
explains each grouping. Batch 008 saves 157 links on 151 records. See
`source_checklists/20260911-mewari-families.md` for scope and data-only validation.
Builds and browser refreshes were excluded by user instruction.

### Mewari batch 009 heads (2026-09-11)

`data/other/params/raw_data/mewari_batch009_heads.py` reproduces three regional
family heads, one Sanskrit donor and one extracted borrowed component. Its
adjacent audit and `source_checklists/20260911-mewari-batch009-heads.md` preserve
provenance and data-only validation. Batch 009 saves 54 links on 44 survey records
and one component node; one survey record remains only partly analysed.
No build or browser refresh was run.


### Nirmaan Mewari dictionary (January 2018)

The new `nirmaan2018mewari` source is prepared in
`data/other/forms/20260911-nirmaan-mewari.csv`: 6,406 articles become 6,535 numbered-sense/
pronunciation rows and 484 source-asserted variant links. It is distinct from the existing
`mewari` survey. `raw_data/nirmaan_mewari.py` extracts pinned PDF font spans without OCR;
its audit retains raw text, IPA, Devanagari, senses and unresolved relations. A dedicated
IPA-to-house profile, Kapasan-area Mewari dialect, bibliography record, regression fixtures
and fresh seeded 20-entry visual audit accompany the source. Twenty-seven articles retain
unresolved variant/compound scope; no new etymologies are inferred. The all-rights-reserved
PDF is not bundled, and no redistribution licence has been established.

The user subsequently authorized a **local database rebuild**. Compilation and browser DB
staging succeeded; all 6,535 dictionary IDs and 484 variant links were verified, with no old
form IDs removed. The compressed artifact is 43.86 MB (cache version 31), and representative
source, entry, variant and dialect views passed browser QA. Broad test-suite exceptions
remain documented in the review; nothing was deployed.
See [the ingestion review](source_checklists/20260911-nirmaan-mewari-review.md) for counts,
transcription decisions, focused validation, pre-existing dialect-test failures, rights status,
unresolved cases and reproducible commands.

## Seligmann and Seligmann's Vedda vocabulary (1911)

`data/other/forms/20260911-seligmann-vedda.csv` installs 502 lexical forms from all
185 numbered articles on pp. 424–450 of *The Veddas* (Cambridge, 1911), including
numbered subsenses and explicit supplementary attestations. The source package
`data/other/forms/raw_data/seligmann_vedda_1911/` pins the scan hash, preserves the
original page OCR, and stores an image-checked transcription. Reproduce a preview
with `python3 data/other/forms/raw_data/seligmann_vedda.py`; add `--install` to write
the canonical CSV and per-form JSONL audit, or `--sample 19110911` for the seeded
audit. This needs no network or PDF download once the checked package is present.

The dedicated sound profile preserves the source's broad field transcription,
including its occasional diacritics; there is no independent IPA or native-script
field. Vedda is registered separately (`vedd1240`, existing `Other` clade), with
11 source-locality dialects. Historical coordinates remain blank. The consultant
Wannaku of Uniche remains provenance. Forty-four forms carry an unresolved `T.`
label (the abbreviation key says Tamil, while locality lists also use `Tk.` for
Tamankaduwa); these are not silently remapped. One source-questioned spelling and
one disputed bee identification also carry typed uncertainty in the audit. The
comparative etymological commentary is preserved in the raw evidence, but this
lexical ingest makes no ancestry or borrowing assertions. Comparative-language
forms, narrative chapters, songs and the following Kaele-base appendix are outside
scope. See `source_checklists/20260911-seligmann-vedda-review.md` for validation.


### Western dialect survey vocabularies (2026-09-11)

Added the Davar/Dungar columns of *Linguistic Survey of India: Dadra and Nagar
Haveli* (2003), plus Ghatage's *Konkani of South Kanara* (1963) and *Kudali* (1965).
The three rich CSVs contain 816, 1,389, and 1,902 forms respectively, all unlinked
lexical attestations. Davar uses Bhili; Dungar introduces canonical Dungar Varli;
the Ghatage varieties use existing Konkani/Marathi with named dialect tags.
Source-defined homographs and every page/item locator survive through stable keys.

Varli was manually transcribed from eight comparative tables; comparison columns
and blank targets are explicitly audited. Ghatage uses pinned, deskewed 300-DPI
OCR with `ocr-review uncertain` on every unreviewed form; 50 Konkani and 133 Kudali
unresolved/nonlexical candidates remain audit-only. These are not fully proofread
transcriptions. Image audits and corrections are recorded separately from the
parser's zero-error structural sample. See
[`source_checklists/20260911-western-surveys-review.md`](source_checklists/20260911-western-surveys-review.md)
and [`ghatage_western/README.md`](data/other/forms/raw_data/ghatage_western/README.md)
for reproduction, reference/licence provenance, transcription decisions, and
outstanding OCR review. The browser database is refreshed only on request.

### Kharia Living Dictionary (2026-09-11)

`20260911-kharia-living.csv` imports all 452 upstream entries from the public
2026-08-20 Living Dictionaries SQLite snapshot, yielding 521 sense/variant rows.
`raw_data/kharia_living.py` pins the SQLite hash and emits a complete per-entry
audit, preserving upstream entry/sense IDs, native spelling, phonetic forms,
multilingual glosses and dialect evidence. Reuse permission was confirmed by the
user. This is distinct from Peterson's 2009 lexicon. Dhelki/Dudh labels map to
registered dialects beneath `kh`; unknown dialect coordinates remain blank.
The preservation profile retains source breves/glottalization rather than
inventing a phonological interpretation. Six entries lack pronunciation and
remain marked native-script attestations. Printed phonetic alternatives produce
68 variant links; no ancestry is inferred. See the source checklist review for
coverage, exclusions, audit decisions and validation status.

### Approved Malvi, Nimadi and Bagheli etymologies (2026-09-11)

All 757 reviewed proposals are saved in the etymology overlay: 3,777 links on
3,731 survey records. At the user's direction, 223 Perso-Arabic loan links
are nested under Persian/Arabic etymological heads with transmission left
unspecified. A nine-head dictionary supplement resolves the missing heads;
`data/other/params/raw_data/central_surveys_donors.py` reproduces its dated
CSV from the adjacent audit. Existing Hindi heads remain unchanged.
The isolated complete compilation preserves all approved records and links;
validation exceptions and the unaccepted held records are documented in
[the acceptance report](curation/etymology-lab/central-surveys-20260911/ACCEPTED.md)
and [source checklist](source_checklists/20260911-central-surveys-donors.md).
The subsequent user-requested shared CLDF/browser rebuild is staged locally as cache version 32. See [rebuild results](curation/etymology-lab/central-surveys-20260911/DB-REBUILD.md).

### SDML online lexical survey

The 2026-09-11 SDML snapshot adds 47,317 Marathi attestations across 269 populated
village sites (271 source rows) and 73 lexical columns. The importer is
`data/other/forms/raw_data/sdml.py`; its pinned CC BY-SA 4.0 export, audit, frequencies,
transcription policy and exclusions are in `data/other/forms/raw_data/sdml_2026/`.
All responses remain unetymologised. There are 33 excluded ambiguous tokens and 37
retained unusual-source-symbol forms flagged for review. See the source review for
build and test results.

### Bajjika and comparative Dravidian datasets (2026-09-11)

`data/other/forms/raw_data/ia_dravidian.py` imports the pinned Bajjika survey
(Regmi et al. 2014), Lindgren et al.'s Zenodo TSV (2023), and DravLex CLDF.
The installed files are `20260911-bajjika.csv` (1,208 alternatives from 1,050
site/concept cells), `20260911-lindgren.csv` (3,754 additional records), and
`20260911-dravlex.csv` (2,127 records). All 2,127 DravLex records republished by
Lindgren retain the second publication's IDs and citations without becoming
independent observations. Complete raw records, language mapping, hashes and
per-record audits are in `raw_data/ia_dravidian_2026/`.
Bajjika is extracted from positioned PDF tables without OCR; the Dravidian
sources retain original transcription and segmented IPA separately. Source
cognate sets and loan labels are preserved without guessed etymological parents.
See `source_checklists/20260911-ia-dravidian-review.md` for scope and validation.

### Census comparative lexicons, Danuwar and archived Tharu (2026-09-11)

`data/other/forms/raw_data/census_nepal.py --output /tmp/census-proposal --install`
reproduces six survey imports from the pinned `census_nepal_2026/` source cells.
With the original hashed downloads cached, `--extract` reconstructs PDF tables and
the Tharu XLSX using the bundled OOXML reader. The Census PDFs require structural
normalization; 148 image-containing cells have manually restored transcriptions.

The batch adds 20,403 records from Tamil Nadu, Uttar Pradesh, Bihar, three regional
Nepali lists, five Danuwar sites and eight Tharu lists. It reuses 761 exact existing
Kochila readings with supplemental citations. Source-local IDs, full per-cell
audits, exclusions, provisional dialect mappings and transcription uncertainties
are retained. The Census tables contain published gloss inconsistencies and some
unresolved word-wrap boundaries; affected records are explicitly review-flagged.
See `source_checklists/20260911-census-nepal-review.md` for scope and validation.

### Additional Census and Nepal surveys (2026-09-11)

`data/other/forms/raw_data/more_surveys.py` ingests the Indo-Aryan/Dravidian comparative
columns of LSI Jharkhand, Himachal Pradesh, Rajasthan I and West Bengal I, plus Mahato's
five-site Kisan survey: 18,318 raw installed readings from 17,092 target cells. Cached
positioned PDF cells, hashes, explicit readings, language maps and per-record audits live
in `data/other/forms/raw_data/more_surveys_2026/`. Mixed ASCII/IPA transcription uses
`conversion/more-ascii.txt` and `conversion/more-ipa.txt`; ambiguities are flagged.
The public Bote and Rajbanshi–Tajpuriya PDFs lack their wordlist appendices, and the Darai
PDF reproduces the already-ingested Danuwar appendix. These three are excluded with evidence,
not falsely represented as ingested. See `source_checklists/20260911-more-surveys-review.md`
for source coverage, omissions, editorial decisions and validation status.


### Angika, Majhi, Koraga and Orissa (2026-09-11)

The pinned `data/other/forms/raw_data/selected_surveys.py` importer installs four
selected sources with `--install`; without that option it stages a review proposal.
The `selected_surveys_2026/` package contains source URLs/hashes, cached extraction,
manual collation, per-record JSONL audits, explicit reading decisions, and repeatable
seeded/targeted crop renderers. Original PDFs and large images remain ignored inputs.

- Regmi (2017), Angika pp. 83–90: 1,050 cells, 1,076 forms, five Nepal sites;
  exclude two source missing-glyph cells and the English/Nepali controls.
- Chalise (2014), Majhi pp. 71–77: 1,050 cells, 1,055 forms, five sites;
  exclude one source missing-glyph cell and controls. Reuse canonical Majhi/Kunauri.
- Bhat (1971), Koraga pp. 88–118: 1,192 entries, 1,369 forms; reuse Onti, Tappu,
  Mudu, with four unspecified-lect forms. Exclude grammar examples and the Belari
  appendix. All 857 candidate short-i glyphs were inspected for plain i versus ɨ.
- LSI Orissa (2002), comparative vocabulary pp. 192–233: 5,065 cells, 5,713 forms,
  Standard Oriya/Sambalpuri/Bhatri/Desia/Relli; exclude 21 empty cells and one printed
  missing-glyph cell. The clear official 300 dpi scan replaces the compressed mirror
  as OCR input. Other chapters are outside this table ingest.

Installed files are `data/other/forms/20260911-selected-{angika,majhi,koraga,orissa}.csv`.
The four `conversion/selected-*.txt` profiles convert defensible correspondences;
ambiguous symbols remain visible in `Original` and are flagged. Comparisons remain
source prose, with no inferred ancestry or borrowing. Angika and Reli are new base
languages; survey sites are registered dialects, with no invented point coordinates.
Gondi and Kui are deliberately excluded from this batch.

All rows retain typed uncertainty. Angika/Majhi/Koraga passed 0/20 seeded checks.
Orissa remains a legacy OCR source: the latest independent sample found 3/20
character/noise errors (all corrected), with no row/lect/prompt misalignment. Its
unreviewed readings are not represented as clean manual transcription. See
`source_checklists/20260911-selected-surveys-review.md` for audit history, residual
limitations and full build/test evidence. Routine ingestion does not refresh the
browser database.

### KEED 2018 and Muduga centralized-vowel examples (2026-09-12)

The Kannada–English Etymological Dictionary (2018 second / first electronic edition, CC BY-NC 4.0) is imported from the canonical ILCAA PDF using glyph IDs and page/font structure, not OCR. Its 28,797 anchors yield 31,250 lexical units plus separately keyed printed variants, paradigms and donors. Native spelling, romanization and IPA remain distinct. Arsenault and Abraham’s Muduga author-preprint Tables 3–19 and glossed prose contribute 91 examples and 18 phonetic realizations (109 rows); the underlying 1,100-word field corpus is not included. The preprint was available as indexed text rather than a downloadable PDF.

Reproduce with `uv run python data/other/forms/raw_data/keed_2018.py --install` and `uv run python data/other/forms/raw_data/muduga_2022.py --install`. The sibling directories contain pinned evidence, complete coverage/record audits and reference/dialect maps. Installed files are `20260912-keed.csv` and `20260912-muduga.csv`. Only source-supported, validated DEDR/CDIAL links and explicit donor/variant/derivation relations are accepted; uncertain or unavailable references remain visible. See [the full review](source_checklists/20260912-keed-muduga-review.md) for transcription, exclusions, unresolved cases and validation.

## Ho, Bhumij and Dhurwa survey integration (14 September 2026)

`20260914-sil-ho.csv`, `20260914-sil-bhumij.csv`, and `20260914-sil-dhurwa.csv`
add 2,900 Ho, 2,100 Bhumij and 809 Dhurwa records to shared lexical inputs.
They reuse canonical parents `ho`, `mu` and `Parji`, register 28 source sites,
and supply explicit conversion profiles and complete references. Frozen manual
ledgers and all exclusions are preserved. Ho item 93 is corrected to *tail*;
Bhumij source qualifiers and Udala's uncertain language label remain explicit.

Read-only verification on 21 September confirms all 5,809 records in the existing
shared CLDF, with stable keys/IDs and 46 correct source-defined variant edges.
Current source profiles reproduce every compiled display form; source transcription,
locators and dialect tags survive unchanged. No database was rebuilt. Full-build,
repository-wide suite and global retrospective audit gates remain unclaimed. See the [integration review](source_checklists/20260914-manual-surveys-review.md).

## Sheth's Prakrit–Hindi dictionary (14 September 2026)

`data/other/forms/20260914-sheth.csv` installs 42,118 structurally resolved
headword/sense/alternate records from the pinned DDSA transcription of Sheth's
*Paia-sadda-mahannavo* (1923–1928). All 41,638 digital articles are audited;
31,501 articles are represented and 10,137 unresolved articles remain audit-only.
The import preserves native Devanagari and source romanization, scoped grammar,
See references and etymological prose, and explicit alternate-head relationships.
It reuses Prakrit, Apabhramsha and Ashokan Prakrit plus existing literary dialects.
The Sanskrit follow-up adds 26,191 source-local counterpart records and 27,484
source-attributed lexical comparison links; no automatic ancestry is asserted.
Verified quoted-source tags now cover 30,147 records, with full work names registered
for frontend filters and pills. Unknown codes and exact locators remain audited.

This is a partial dictionary integration. Embedded compounds/paradigms, damaged
heads, remaining auxiliary reference identities and printed-supplement reconciliation remain
open. The full CLDF build and full suite are deferred under the workspace resource
policy; no browser refresh is included. See the [source package](data/other/forms/raw_data/sheth_2026/README.md)
and [integration review](source_checklists/20260914-sheth-review.md).

## Yoshioka's Eastern Burushaski vocabulary (14 September 2026)

The corrected Yoshioka source has 4,886 rows from 3,233 entries/senses. Native
Gentium font decoding replaces OCR repair, restores indented entries, and preserves
grammatical labels, noun classes, full plural forms and suffix paradigms. Exact
cross-references become variant links. Follow-up image review resolves 56 of 62
forms in the 57 previously unresolved index entries; six ambiguous forms
retain their candidate senses without an accepted edge. Historical
entry keys and aliases preserve existing identities. Source-local validation passes;
the full CLDF build, full suite and browser refresh remain pending. See the
[source package](data/other/forms/raw_data/yoshioka_2026/README.md) and
[review](source_checklists/20260914-yoshioka-review.md).

## Dialect site coordinates de-duplicated (17 September 2026)

191 registered dialects that had been pinned to one shared display point — every survey site
of a base language at the language's own coordinate (Kok Borok, Garo, Koch, Palakkad/Idukki
tribal surveys, Nilgiri Irula, Mudhili Gadaba), Morgenstierne's Pashai and Prasun villages,
Bhadarwah villages, Bashir's Khowar sub-localities, and SDML villages whose printed
coordinates duplicate a neighbour's — now carry their own reviewed points.
`fix_shared_dialect_coordinates.py` holds the hand-reviewed table (OpenStreetMap/Nominatim
and GeoNames village points at quality `B`; the nearest source-named administrative unit or an
explicit map reading at quality `C`) and records every row in
`data/dialect-coordinate-decisions.csv`. Comparative-language labels in the Romani source now
sit on their own country's centroid, and Magadhi/Paiśācī on approximate historical points.
Hamlets absent from both gazetteers (several Attappady and Idukki settlements, Garo villages
in Haluaghat/Dhobaura/Kalmakanda, the West Garo Hills Koch villages, the Rudhari sites) share
their block/upazila point and are marked approximate rather than left on the language point.
The 51 Bhil-area survey sites (ESR 2018-011 Bareli/Bhilali/Bhili, Nimadi 2012 Bhilala, Malvi 2009,
Dhule 2013 Vasave/Noiri, Noira 2015) previously had no coordinates at all, so the browser drew
them on their parent language's point; they are resolved the same way from the tahsil/district
printed in each Location, and `backfill_language_coordinates.py` now derives centroids for
Vasavi, Noiri and Dungra Bhili from them.
A further 347 dialects with blank coordinates (SIL Malvi, Bagheli, Kullu, Pothwari, Ho, Korwa,
Kodaku, Koya, Korku, Bonda/Didayi, Desia, Karbi, Gujari, Haryanvi, Kurumba, Adi and Kurux
surveys; PLSI chapter localities; Angika/Majhi/Danuwar/Vedda sites; Zoller's toponymic variety
labels) received hand-reviewed points the same way. The 129 still blank are regional or
register labels with no locatable place ("Uttar Pradesh", "Standard Bangla list", Zoller
"Western"), plus a few field villages (Bhumij 1989/1996, some Seligmann Vedda sites) that no
gazetteer knows.

## `id.` glosses resolved; grammatical labels attached to their forms (17 September 2026)

DEDR and CDIAL write `id.` (*idem*) for "same meaning as the preceding form"; 6,946 compiled
forms carried that abbreviation verbatim (the DEDR parser had even stripped its period).
`resolve_idem()` in `data/dedr/parser_utils.py` now expands it from the previous glossed form
in both parsers, keeping additions and citations. Two source problems it exposed are handled
alongside: italic Latin species names set in bold (`<i><b>Linum usitatissimum</b></i>`) no longer
parse as forms (221 bogus DEDR rows removed, the names restored to their glosses), and the
fifteen DEDR entries whose digital text opens with `id.` because the printed first gloss was
lost take that gloss from the printed dictionary (`entry_initial_glosses` in
`data/dedr/abbrevs.py`). In CDIAL, a grammatical label printed before a form (`pl. dōnye̯`,
`imper. bēza`, `obl. hamā`) is now stored as that form's note instead of the preceding form's,
and only such a detached label opens a new comma-run for definition sharing.

CDIAL blank definitions were then filled by Turner's layout conventions: a borrower in a
`(→ …)` note keeps the lender's meaning; a short comma-run closed by one glossed form across a
language label shares it; `do.` (ditto) substitutes the preceding head sense; and a reflex
printed without a meaning keeps its headword's (or its numbered section lemma's) meaning —
with causative/valency runs, derivative, contamination (`X`) and borrowing groups excluded,
extension groups preferring their own first printed gloss, and gender-labelled nouns under a
verbal lemma left blank. Two 50-row manual audits of the blanks drove these rules (≈62% then
≈62% would have been filled correctly by the naive "preceding gloss" rule, versus 0 wrong of
31 applicable in the sample for the headword rule). CDIAL blank glosses fell from 15,353 to
about 4,000, of which 3,600 are OIA head-paragraph variants and `see X` heads left blank on
purpose.

### Bhattacharya 1957 Ollari vocabulary

`data/other/forms/20260921-bhattacharya-ollari.csv` contains 880 rows from all
657 physical vocabulary entries on printed pp. 48–77 of *Ollari: A Dravidian
Speech*. The accompanying YAML routes the source transcription and preserves
stable keys, homonyms and meaningful hyphens. Regenerate with
`make ingest SOURCE=20260921-bhattacharya-ollari`. The importer requires the
passing pinned source-to-preview audit before installation.

The image-only Tamil Digital Library scan was visually transcribed using a
retained OCR scaffold. Raw review records, acquisition details, source hashes,
failed and passing audits, and relationship decisions are under
`data/other/forms/raw_data/bhattacharya_ollari_1957/`. The 509 ordered prose
passages are installed in `data/other/entry_texts/20260921-bhattacharya-ollari.csv`.
Comparative-language forms remain source-attributed prose, not independently
ingested attestations or inferred ancestry. The source creates 124 variant
and 111 derivation/inflection relations through stable keys.

Ollari maps to existing `OllariGadaba`; collection villages remain provenance
because no per-entry dialect assignment is supplied. No entries are excluded.
One cross-reference has no printed gloss; two suffix scopes remain unexpanded;
three lexical rows and two prose entries retain documented reading uncertainty.
The profile maps ṅ to ŋ and printed affricate pronunciations to house notation;
13 rows preserve explicit pronunciation overrides separately from source spelling.
The fresh 20-entry audit has zero material errors. Canonical inputs are installed;
full compilation, graph/ID survival, formatted references, and the full test suite
remain required. No browser refresh was requested.


Das (1987), *The Yeravas of Kodagu*, adds 90 source rows from its complete
comparison table: 48 Panjiri Yerava (Ravula), 42 Pani Yerava (Paniya). Printed
popular spelling is preserved without inferred phonemic values, and the two
Kodagu varieties have distinct dialect records. The [source package](data/other/forms/raw_data/das_yerava_1987/README.md)
records the 0/20 visual acceptance audit, exclusions, reproduction and outstanding
compiled-integration gates. Source inputs are installed; full ingestion is pending.


### Complete Sansi ordinary and argot source stage (LSI XI,1922)

`data/other/forms/raw_data/grierson_sansi_1922/import_source.py` reproduces1,943 reviewed forms from2,562 source units: full lexical/grammatical prose49–53/60–63, ordinary specimens54–57, argot specimens64–70, and all241 prompts in both standard-list columns178–210. The source-local audit retains32foreign comparisons,80Kheri Hindostani controls,26bound endings, two attached pronunciation approximations and533 repeated specimen attestations with every locator. Argot is a usage register under Sansi, while explicit regions use registered dialect tags without invented coordinates. Literal accents, nasal marks and underdots remain source evidence;154 explicit argot derivations are structured,148 contextual base-attributions remain uncertain. All87 pilot keys survive. Independent20/20 audit plus edges and6focused tests passed; fullCLDF/database/browser gates remain explicitly user-deferred. See `source_checklists/20260926-sansi-full-source.md`.
