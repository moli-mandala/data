# Cheung (2007), *Etymological Dictionary of the Iranian Verb*

Johnny Cheung, *Etymological Dictionary of the Iranian Verb* (Leiden Indo-European
Etymological Dictionary Series 2). Leiden – Boston: Brill, 2007. Citation key
`cheung2007ediv`. Copyrighted; the 625-page born-digital PDF (SHA-256 in `manifest.json`) is
read from a local path and never copied into the repository.

## Scope

Dictionary body, printed pp. 1–475 (PDF pp. 26–500): 701 Proto-Iranian verbal roots.
Excluded: front matter, introduction and abbreviation lists (used for the language table),
the bibliography and indexes (PDF 502–625).

| Output | Rows |
|--------|------|
| `data/other/params/20261025-cheung-ediv.csv` (PIr etyma) | 701 |
| `data/other/forms/20261025-cheung-ediv.csv` (Iranian attestations) | 12,960 |
| `data/other/entry_texts/20261025-cheung-ediv.csv` (prose on the etyma) | 3,817 |
| `audit.csv` (one row per raw unit) | 18,408 |

Audit statuses: 701 `installed_etymon`, 12,960 `installed_form`, 1,200
`audit_only_paradigm` (indented inflected-form lines with text passages), 2,882 `entry_text`
(Sanskrit, PIE, IE cognates, Misc. comparanda, ◊ commentary, References), 429
`preverb_sense_only` (Avestan `(+ *ā-) ‘to apportion’` with no printed form), 165 `non_iranian`
(Misc. comparanda in Arabic, Armenian, Tocharian, …), 66 `no_form_segment` (annotations
printed between commas without a form), 3 `unmapped_label` (Sarmatian names, one prose
sentence), 2 `not_a_form` (italic words cited inside prose).

## Pipeline

1. `pdf_layer.py` — pdfminer/pdfplumber with `ToUnicode` bypassed: every glyph of every
   `/Differences`-encoded font is recorded by glyph name. Each of the 23 fonts is a single PDF
   object, so glyph numbering is document-wide. Lines are rebuilt from baselines; overprinted
   duplicates (the bold headword font draws `*` twice) are removed; 5–7.9 pt raised/lowered
   glyphs (homonym indices, `H₂`, `bʰ`) attach to their line as scripts. Cached as
   `tmp/iranic/cheung/extract.json.gz` (outside the repo; ~35 s to rebuild).
2. `glyph_table.py` → `glyphs.csv` (488 rows): glyph name → Unicode with method and evidence.
   Standard Adobe names (213), descriptive Dutch names decoded compositionally and checked on
   outlines (89: `onderstip`, `keepje`, `onderboogje`, `golfje`, `sleufje`, …), opaque numeric
   and PUA names identified from the embedded CFF outlines plus printed context (138, e.g.
   `tt260`=ā, `tt928`=ž, `gg6`=γ, `tt1264`=β, `uniF16B`=ū; the 8-bit Greek font by key
   position), and 48 non-spacing marks (drawn left of their origin, so postfix combining
   characters; the generator cross-checks this geometry against every decision). No glyph is
   unresolved; Arabic-script words (on 23 printed lines, prose only) become `[Arabic script]`.
3. `decode.py` — Unicode lines with style markers (italic = citation form, bold = headword /
   cross-reference, scripts). Greek-font letters join the italic form (Sogdian/Choresmian β δ γ
   θ; Bactrian is printed in Greek script).
4. `structure.py` — entries (bold `*root` after a blank line or at a page top), `•LABEL:`
   sections, 7.9 pt paradigm blocks, the right-hand `→` reference column (including wrapped
   continuations), running heads dropped, line-break hyphenation repaired.
5. `tokens.py`, `items.py` — citation items: `||` preverb groups `(+ *fra-)`, comma-separated
   items `[?] [grammar] Label [(Sub)] form [/phonemic/] [(notes)] [‘gloss’]`. Unlabelled items
   inherit the previous label or the section language; a gloss scopes back over preceding
   gloss-less items of its group (`MMP b’m, BMP b’m ‘splendour’`), never across `||` or `;`.
   Gloss boundaries tolerate the source's quotation slips: a closing ’ followed by `=`/`(`,
   a straight `'`, a missing closer (the gloss then ends before `||`, `(+`, `◊` or
   `, Label form`), a missing opener (`š’m to swallow’`), nested quotes; `id.` repeats the
   preceding gloss. Label-only segments share the next form (`Abyan., Abz. yöšta/yǖz-`); a
   second label without a comma starts a new item; `(+ *X-)` after a gloss opens a preverb
   group even without `||`.
6. `import_source.py` — rows, entry texts, audit, seeded sample, manifest. Dry run by default;
   `--install` writes. `build_profile.py` regenerates `conversion/cheung-ediv.txt`.

## Modelling

* Etyma: `Language_ID=PIr`, `Form` = printed root with `*`, gloss = headword gloss, ID from the
  printed root (ASCII transliteration) + homonym index (`cheung-ediv-baud2`). The two unnumbered
  `*baǰ` headwords are `cheung-ediv-baj-p1`/`-p3`. Alternative reconstructions and `?` on the
  root go to an `etymology` entry text.
* Forms: base language + dialect tag through `lects.csv`; `Entry_Key` =
  `cheung2007ediv:<root id>:<printed label>:<printed form>[:n]`. `Original` keeps the printed
  form; principal parts stay together (`bastan/band-`); comma-listed alternates inside one
  italic run become build-time variants. Grammar labels → canonical tags (`m f caus pass intr
  tr denom ppp pret stem impv subjunctive inchoative loanword proper-noun …`; queried labels
  `(denomin. ?)` add `uncertain`). Cheung's formation and derivation analysis goes to
  `Etymology`: the preverb group (`preverb *fra-`), `< *…` derivations, cf./rather/orig./
  sec./old comments and supporting citations. Passage loci (`Yt 10.134`, `DB 4.84`, Khotanese
  and Bactrian document loci) go only to the audit column `Passage_Locators`. Notes keep
  residual source information: alternate stems `(baxš-)`, suppletive stems, Old Persian
  graphemic spellings `<b-r-š-n-a>`, usage remarks. `?` → `uncertain` with the typed reason in
  the audit (`etymology: source query`, `transcription: tentative/emended reading`,
  `transcription: starred stem`, `grammar: queried label`).
* BMP `/…/` transcriptions → `Phonemic`. Pahlavi glosses of Avestan words (`‘[BMP] škpt’`) →
  Notes.
* No graph relations are inferred: Sanskrit/PIE/IE material is prose; borrowing statements
  stay as Notes/`loanword` tags.

## Transcription

Old and Middle Iranian (Av, OPers, MPrs, Parth, Khot, Tumsh, Sogd, Chvar, Bactr) keep the
printed philological transliteration (not routed to any profile). Modern languages use
`conversion/cheung-ediv.txt` (š→ś ž→ź č→c ǰ→j c→ʦ ʒ→ʣ δ→ð ϑ→θ γ→ɣ x̌→χ w→v; Lecoq IPA tʃ dʒ ʃ
V: → c j ś macron; ы→ɨ; ’→ʼ; vowels as printed). Review items are in `proposals.md`.

## Audit

`sample.csv` is a seeded (`cheung2007ediv-audit-3`) 50-row raw-vs-output comparison stratified
over OPers, Bactr, Chvar, Sogd, Khot and modern languages (≥5 each): 0/50 material errors.
History: seed 1 (20 rows) found a citation form broken after its hyphen at a line end
(`haɫ-|āwasīn`); a coordinator review of the whole CSV then found gloss over-capture (source
quotation slips), Bactrian ϸ (set as text-font þ) and hyphens split off Greek-script forms,
analysis and passage loci in Notes, and Old Persian graphemic spellings yielding a stray form.
All are fixed and covered by whole-corpus regression tests. Remaining empty glosses (95) are
unglossed in the source: names (16), Ormuri `=` dialect equivalents (13), forms cited only
inside a phrase (12), suppletive/inflected/affix entries (11), list items without a printed
gloss (40), loan lists (2), one with both quotes missing. Tests: `tests/test_cheung_ediv.py`.

## Reproduce

```bash
cd data
.venv/bin/python -I data/other/forms/raw_data/cheung_ediv_2007/glyph_table.py      # only if decisions change
.venv/bin/python -I data/other/forms/raw_data/cheung_ediv_2007/import_source.py --install
.venv/bin/python -I data/other/forms/raw_data/cheung_ediv_2007/build_profile.py
.venv/bin/python -m pytest -q tests/test_cheung_ediv.py
```
