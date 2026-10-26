# Iranic family extension (2026-10-25)

Checklist: `data/SOURCE_INGESTION_CHECKLIST.md` is active for every source below.
Source PDFs (not redistributed): `/Users/aryamanarora/Documents/Linguistics/Indo-European/Iranic/`.

## Sources and installed names

| # | Source | Key (`cldf/sources.bib`) | Stem / package | append_order | Addenda |
|---|--------|--------------------------|----------------|--------------|---------|
| 1 | Cheung, *Etymological Dictionary of the Iranian Verb* (Brill 2007), 625 pp., born-digital text layer with custom-encoded glyphs | `cheung2007ediv` (exists) | `20261025-cheung-ediv`, `raw_data/cheung_ediv_2007/`, etymon layer `data/other/params/20261025-cheung-ediv.csv` | 112 | etymological/comparative, dictionary |
| 2 | Uesugi & Catt (eds.), *Old Avestan Dictionary* (ILCAA, Asian and African Lexicon 67, 2024), CC BY 4.0, 558 pp., text layer | `uesugi-catt2024oad` (new) | `20261025-uesugi-oad`, `raw_data/uesugi_oad_2024/` | 113 | dictionary |
| 3 | Kamioka & Yamada, *Lārestāni Studies 1: Lari Basic Vocabulary* (ILCAA 1979), 232 pp., image-only scan | `kamioka-yamada1979lari` (new) | `20261025-kamioka-lari`, `raw_data/kamioka_lari_1979/` | 114 | survey wordlist, OCR-heavy |
| 4 | Rastorgueva & Edel'man, *Ètimologičeskij slovar' iranskix jazykov* vols. 1–3 (a–h; 2000, 2003, 2007), 1324 pp., ABBYY Russian text layer | `rastorgueva2000esija` (exists) | `20261025-esija`, `raw_data/esija_2000/`, etymon layer `data/other/params/20261025-esija.csv` | 115 | etymological/comparative, OCR-heavy |

All four YAMLs use `defaults.identity.legacy_ids: stem` with the `append_order` above so no
legacy pre-ID is renumbered (see `source_files.appended_source_files`).

## Family model

- **Proto-Iranian** `PIr` (reconstructed; `cldf/historical-languages.csv` + `cldf/languages.csv`,
  clade `Old Iranian`). Cheung's roots and ESIJa's lemmata are etymon entries (params) in `PIr`.
  Attested reflexes cite their etymon in `Parameter_ID`. The two dictionaries' entries stay
  distinct records (source-defined); they are not merged.
- No PIr→PIIr or Iranian→CDIAL ancestry is inferred. Sanskrit/IE comparanda printed by the
  dictionaries are comparison prose (entry texts), not graph edges.
- **Clades** (`languages.csv` `Clade`, synced to `jambu-static/src/lib/clades.ts`/`cladeTree.ts`):
  `Old Iranian` (Av, OPers, PIr, Median), `Middle Iranian` (MPrs, Pahl, Parth, Sogd, Khot,
  Bactrian, Khwarezmian, Tumshuqese), `SW Iranian` (Persian, Tajik, Luri, Larestani, Kumzari,
  Bashkardi, …), `NW Iranian` (Kurdish, Balochi, Zazaki, Gurani, Gilaki, Mazandarani, Talysh,
  Tati, Semnani and the Central Plateau languages, …), `E. Iranian` (Ossetic, Yaghnobi, Pashto,
  Waneci, Ormuri, Parachi), `Pamir` (Shughni group, Yazghulami, Wakhi, Ishkashimi, Sanglechi,
  Munji, Yidgha, Vanji). Groups: Old Iranian, Middle Iranian, Western Iranian, Eastern Iranian;
  super-branch and family `Iranian`.
- **Base languages**: one per Glottolog language-level languoid (glottolog-cldf v5.2, cached at
  `tmp/iranic/glottolog/languages.csv`); named villages, corpus varieties and stages are
  language-qualified dialect tags (`dialect:<Language_ID>:<source label>:<name>`). Existing IDs
  are reused: `Av OPers MPrs Pahl Parth Sogd Khot Pers Kurd Bal Psht wne Orm Par Oss Yghn Shgh
  Rosh X Sar Yazgh Wkh Ishk Sang Mj Yid Ir`, plus `Chvar` (Khwarezmian =
  Cheung's Choresmian), `Wj` (Wanji/Vanji) and `Darw` (Darwazi). Bartangi stays the existing
  `dialect:Shgh:Brt:Bartangi`. Planned new base IDs (final list set by the registry pass):
  `PIr Bactr Tumsh Taj Lur Lar Kumz Bashk Zaza Gur Gil Maz Tal Tati Semn Sangs`, plus the
  Central Plateau languages Glottolog recognises (Natanzic, Soic, Khunsaric, Gazic, Nayinic,
  Zoroastrian Yazdi, Vafsi, Ashtiani, Sivandi, Lasgerdi, Sorkhei-Aftari, Shahmirzadi, …).
- Each importer maps source lect labels through its own package table `lects.csv`
  (`Source_Label, Language_ID, Dialect_Tag, Status, Note`). Labels not yet registered are
  proposed there (`Status=proposed`) and are resolved by the central registry pass, which owns
  `cldf/languages.csv`, `cldf/dialects.csv` and `cldf/historical-languages.csv`.

## Transcription decisions

- Old and Middle Iranian forms are cited in their standard philological transliteration
  (Avestan, Old Persian, Pahlavi/Manichaean MP, Parthian, Sogdian, Khotanese, Bactrian,
  Choresmian). Converting them to phonology would impose an interpretation, so `Form` keeps the
  transliteration (NFC; only mechanical repairs of the PDF/OCR encoding).
- Modern languages in Iranist transcription convert consonants to house symbols as
  `conversion/cdial.txt` does for Turner's Iranian citations: `š→ś ž→ź č→c ǰ→j δ→ð ϑ→θ γ→ɣ
  x̌→χ w→v`; Iranist dental affricates `c→ʦ`, `ʒ (dz)→ʣ`. Vowels keep their printed quality and
  length (`ā ē ō ə` etc.); `ʌ/ɪ/ʊ` follow `profile_policy.py`. The w→v merger loses the Wakhi
  w/v contrast exactly as the existing Wakhi profiles already do; this is flagged for review.
- Every new profile is registered in `profile_policy.py` and has full input-symbol coverage tests.

## Division of work

Per-source packages (importer, YAML, installed CSV, params CSV, entry-text sidecar, audit,
sample, manifest, conversion profile, tests, package README) are built independently. Shared
files — `cldf/*.csv`, `cldf/sources.bib`, `tags.py`, `profile_policy.py`, `source_files.py`,
`make_cldf.py`, the frontend — are changed only in the central integration pass. Proposed tags,
bib records and profile-policy entries are left in the package as `proposals.md`.
One `make all` runs after all four packages are installed.
