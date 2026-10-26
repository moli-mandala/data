# Cheung (2007) — proposals for the integration pass

Package: `data/other/forms/raw_data/cheung_ediv_2007/`. Nothing below has been applied to
shared files; each item names the file the integration pass owns.

## 1. `cldf/sources.bib` — update `cheung2007ediv`

Replace the indirect-citation fields of the existing record:

```bibtex
  included     = {Yes; dictionary body pp. 1-475 complete: 701 Proto-Iranian verbal roots
                  (etymon layer), 12,960 attested Iranian forms (Old, Middle and modern Iranian),
                  3,817 typed prose blocks (Sanskrit, PIE, IE cognates, Misc. comparanda,
                  commentary, references); inflected-form paradigms, non-Iranian comparanda and
                  preverb-only senses accounted for in the audit},
  provenance   = {data/other/forms/20261025-cheung-ediv.csv; data/other/params/20261025-cheung-ediv.csv;
                  data/other/entry_texts/20261025-cheung-ediv.csv;
                  data/other/forms/raw_data/cheung_ediv_2007/audit.csv (importer import_source.py);
                  also cited indirectly through data/other/params/20260827-wiktionary-piir.csv},
  ocr          = {No},
```

Keep `etymology_provenance = {source}` and `jambu_editor`. Add `isbn = {978-90-04-15496-7}` if
the shared style records ISBNs.

## 2. Languages (`cldf/languages.csv`, `cldf/historical-languages.csv`)

Existing IDs reused: `PIr Av OPers MPrs Parth Khot Sogd Chvar Pers Kurd Bal Psht wne Orm Par Oss
Yghn Shgh Rosh X Sar Yazgh Wkh Ishk Sang Mj Yid Wj`. Choresmian uses the existing `Chvar`
(registry correction). New base IDs used by the CSV (all `Status=proposed` in `lects.csv`;
coordinates are Glottolog points, quality `C`, several of which look displaced and should be
checked against the dialect localities):

| ID | Name | Glottocode | Lat, Lon (Glottolog) | Clade |
|----|------|-----------|-----------|-------|
| `Bactr` | Bactrian | bact1239 | 36.81, 66.85 | Middle Iranian (historical: attested) |
| `Tumsh` | Tumshuqese | tums1237 | 39.87, 79.07 | Middle Iranian (historical: attested) |
| `Taj` | Tajik | taji1245 | 38.82, 68.55 | SW Iranian |
| `Bakht` | Bakhtiari | bakh1245 | 32.02, 49.77 | SW Iranian |
| `Lur` | (Northern) Luri | nort2645 | 32.54, 47.92 | SW Iranian |
| `Kumz` | Kumzari | kumz1235 | 26.16, 56.42 | SW Iranian |
| `Bashk` | Bashkardi | bash1263 | 27.33, 56.87 | SW Iranian |
| `Zaza` | Zazaki | diml1238 (Dimli; Kirmanjki kirm1248 separate) | 38.61, 39.64 | NW Iranian |
| `Gur` | Gurani | gura1251 | 36.84, 44.50 | NW Iranian |
| `Gil` | Gilaki | gila1241 | 37.13, 49.84 | NW Iranian |
| `Maz` | Mazanderani | maza1291 | 36.24, 52.30 | NW Iranian |
| `Tal` | Talysh | taly1247 | 38.46, 48.63 | NW Iranian |
| `Tati` | Tati (Takestani group) | take1255 (or group tati1243) | 36.07, 49.69 | NW Iranian |
| `Natz` | Natanzic | nata1252 | 33.46, 51.95 | NW Iranian |
| `Soic` | Soic | soii1239 | 31.85, 54.60 (check) | NW Iranian |
| `Gazic` | Gazic | gazi1243 | 34.04, 49.46 (check) | NW Iranian |
| `Khuns` | Khunsaric | khun1255 | 36.70, 49.37 (check) | NW Iranian |
| `Nayin` | Nayinic | nayi1242 | 34.22, 53.52 | NW Iranian |
| `Kavir` | Kavir (Khur) | kavi1234 | 33.52, 55.12 | NW Iranian |
| `ZorYzd` | Zoroastrian Yazdi | zoro1242 | 31.82, 54.45 | NW Iranian |
| `JHam` | Judeo-Hamadani-Borujerdi | jude1268 | 34.79, 48.50 | NW Iranian |
| `Asht` | Ashtiani | asht1244 | 37.75, 48.68 (check) | NW Iranian |
| `Vafsi` | Vafsi | vafs1240 | 34.53, 50.00 | NW Iranian |
| `Siv` | Sivandi | siva1239 | 30.08, 52.92 | NW Iranian |
| `Lasg` | Lasgerdi | lasg1238 | 35.40, 53.00 | NW Iranian |
| `SorkhA` | Sorkhei-Aftari | sork1239 | 35.46, 53.22 | NW Iranian |
| `Sangs` | Sangisari | sang1315 | 35.72, 53.30 | NW Iranian |
| `Semn` | Semnani | semn1249 | 35.53, 53.29 | NW Iranian |
| `Shmz` | Shahmirzadi | shah1253 | 35.78, 53.32 | NW Iranian |

If the registry pass chose different IDs for the same Glottolog languoids (plan names
`Lur Lar Kumz Bashk Zaza Gur Gil Maz Tal Tati Semn Sangs`), change only the `Language_ID` column of
`lects.csv` and re-run `import_source.py --install` (Entry_Keys do not contain Language_IDs).

## 3. Dialects (`cldf/dialects.csv`)

Every dialect tag used is listed in `lects.csv` (column `Dialect_Tag`, `Status` says
`existing-dialect` or `proposed-dialect`; `Note` carries the display name and Glottocode).
Reused: `dialect:Shgh:Bj:Bajui`, `dialect:Shgh:Brt:Bartangi`, `dialect:Psht:waz:Waziri`.
Proposed: corpus/stage varieties (`YAv OAv El-OP MMP BMP IMP IPth OKh LKh BSogd MSogd CSogd
SSogd`), Ossetic `Iron`/`Digor`, Kurdish `Kurm Sor Sul Sina Mukri Abd`, `Bal EHB`, Gurani
`Kand Gahv Bajal Awrom`, `Gil Rsht`, Zazaki `Kor Siwer`, Tati villages, Central-Plateau villages
(Abyāneh, Farizand, Natanz, Yarand, Tār, Abuzeydābād, Delijān, Jowšaqān, Kešeh, Qohrūd, Soh,
Meime’, Kashan, Ārdestān, Gaz, Isfahan, Sedeh, Kafrōn, Varzeneh, Zefreh, Khunsār, Mahallāt,
Vōnišun, Anārak, Nāin, Khur, Yazd, Gabri, Parsi, Hamadan, Aftar, Sorxeh), Persian
`JudeoPers Sist Khor Herat`, Luri `Feili`, Bashkardi `NorthBashk SouthBashk`, Shughni `Orosh`,
Pashto `Afr`. Locality coordinates: none supplied by Cheung; use the dialect's Glottolog point
where one exists (quality C) or leave blank.

Decisions needing linguistic review (all marked `review` in `lects.csv`): village → Glottolog
language assignments not covered by Glottolog (Abuzeydābād, Tār → Natanzic; Meime’, Kashan →
Soic; Kafrōn, Varzeneh, Zefreh → Gazic; Vōnišun → Khunsaric; plain "Yzd." → Zoroastrian Yazdi);
Bajelani (language-level in Glottolog) kept as a Gurani dialect; Judeo-Persian kept under
`Pers`; BMP mapped to `MPrs` although the registry also has `Pahl` (same Glottocode); Tati
villages under one `Tati` base although Eshtehardi/Harzani are language-level in Glottolog;
undocumented abbreviations `Tt. (Ram.)` and `Kasa`.

## 4. `profile_policy.py` (applied by the coordinator; `tests/test_sound_profiles.py` wired)

Add `"cheung-ediv"` to `SCHWA` (ə is a distinct vowel in the modern Iranian data). It is not a
long-vowel-plain-letter source (length is printed with macrons), so not `LONG_A/I/U`.
`tests/test_sound_profiles.py`: map filename key `"cheung"` to `"cheung-ediv"` in the explicit
profile mapping, and add `"cheung-ediv": [("20261025-cheung-ediv.csv", {<modern IDs>})]` to
`CHECKED_PROFILE_FILES` (the modern IDs are the `languages:` list of the YAML).

Profile decisions to review: x̆ (breve) read as x̌ → χ and γ̆ read as γ̌ → ɣ̌ (the font prints both
caron and breve forms; 31 and 121 rows); ǧ kept (as in `cdial.txt`); ы → ɨ (Wakhi); ь/ъ
(Ishkashimi, 28 rows) kept as printed; Lecoq's Meime’/Jowšaqān IPA (tʃ dʒ ʃ V:) converted to
house letters; Ossetic/Pashto ’ → ʼ. Old/Middle Iranian and Bactrian (Greek script) are not
converted (plan: transliteration preserved).

## 5. `tags.py`

Done by the coordinator: `inchoative` is used for `inch.` (141 rows). Preverb formations are
not tagged (`preverb` is a word-class tag); they are in `Etymology` as `preverb *X-`
(5,000+ rows; audit column `Preverb`).

## 6. Entry texts / shared code

Entry texts target the etymon rows by `Form_ID` = params ID (as `20260827-wiktionary-piir`
does); `read_entry_text_sources` cannot resolve an `Entry_Key` to a params row (params carry no
Entry_Key), so no Entry_Key column is used. Nothing needs changing unless `assign_form_ids.py`
renames params IDs.

## 7. Unresolved / editorial notes

* Two unnumbered `*baǰ` headwords (pp. 1 and 3) → IDs `cheung-ediv-baj-p1`, `-p3`.
* Glosses `id.` take the preceding glossed item's meaning (Notes record ‘id.’).
* p. 321 `M. rwxšn-` in the Sogdian section is read as Manichaean Sogdian (not Munji).
* Source slips kept as printed: Yzgh. `šəníx̆ʷtšnī́x̌ʷ-` (no slash between principal parts in the
  PDF), Munji `dυf-` (Greek υ), the Parthian-section `MMP gy’n` label.
* Greek accents: 4 legacy-font accent glyphs re-read from words (θαῦμα, νεύω, αὖλαξ, σάρξ,
  Doric σᾶμα); 2 remain low-confidence in `glyphs.csv` (prose only).
* 427 Avestan preverb senses printed without a form (`(+ *ā-) ‘to apportion’`) stay in the
  audit (`preverb_sense_only`): installing them would require constructing unprinted forms.
* 165 non-Iranian comparanda in Misc. (Arab., Arm., Toch., …) and 3 unmapped items (Sarmatian
  names from Olbia, one prose sentence) are audit-only; Misc. sections containing them are
  kept as `comparison` prose.
* Borrowing statements (`(< Pth.)`, `borrowed into Bal.`, `(LW)`) are kept as Notes /
  `loanword` tags; no Borrowed_From_Key relations were created.
* Greek accents in IE-cognate prose come from a legacy 8-bit Greek font; a few precomposed
  accent glyphs are low-confidence (marked in `glyphs.csv`). Greek letters inside Sogdian,
  Choresmian and Bactrian forms are base letters only and are reliable.
