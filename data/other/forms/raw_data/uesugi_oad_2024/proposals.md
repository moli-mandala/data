# Integration proposals — Uesugi & Catt, *Old Avestan Dictionary* (2024)

Package: `data/other/forms/raw_data/uesugi_oad_2024/`. Installed: `data/other/forms/20261025-uesugi-oad.csv`
(+ `.yaml`, `append_order: 113`), `data/other/entry_texts/20261025-uesugi-oad.csv`,
`conversion/uesugi-oad.txt`, `tests/test_uesugi_oad.py`. Nothing below has been applied to shared files.

## 1. `cldf/sources.bib` (new key)

```bibtex
@book{uesugi-catt2024oad,
  title        = {Old {A}vestan Dictionary},
  editor       = {Uesugi, Heindio and Catt, Adam Alvah},
  note         = {Edited by the Old Avestan Dictionary editorial team; Heindio Uesugi (editor), Adam Alvah Catt (editorial supervisor). Part I: Dictionary; Part II: Text and Translation},
  year         = {2024},
  series       = {Asian and African Lexicon},
  number       = {67},
  address      = {Fuchu, Tokyo},
  publisher    = {Research Institute for Languages and Cultures of Asia and Africa (ILCAA), Tokyo University of Foreign Studies},
  isbn         = {978-4-86337-542-0},
  license      = {CC BY 4.0 (Creative Commons Attribution 4.0 International), stated on the copyright page},
  included     = {Part I Dictionary, printed pp. 1--404: all 1199 printed lemmata (997 non-verb, 202 verb-root articles); 1198 installed as Old Avestan (Av) rows plus 164 printed alternant headwords as variants; inflected attestations, passage citations, F cross-references and verb-stem sub-headings are kept in the audit; Part II (Yasna 27.13--15, Gāthās Y28--53, 54.1 text and translation) is excluded},
  provenance   = {Publisher PDF B542_AAL67_OldAvestan.pdf (558 pp., SHA-256 7bb39fa3f87a5bc1400bc7243ad6da4d203ea8bad10765d13fe8a8045851b5f9), born-digital text layer read with pdfplumber; importer and audit: data/other/forms/raw_data/uesugi_oad_2024/ (import_source.py, audit.csv, editorial.csv, sample.csv, manifest.json); installed data/other/forms/20261025-uesugi-oad.csv and data/other/entry_texts/20261025-uesugi-oad.csv},
  ocr          = {No},
  etymology_provenance = {source},
  jambu_editor = {Aryaman Arora and Anthropic}
}
```

Add the ILCAA repository URL/handle if one is confirmed (not invented here).

## 2. Dialect registration (`cldf/dialects.csv`) — see `lects.csv`

| ID | Tag | Language_ID | Source_Language_ID | Name | Glottocode | Lat | Long | Clade | Location | Quality |
|---|---|---|---|---|---|---|---|---|---|---|
| av_old | `dialect:Av:OAv:Old%20Avestan` | Av | OAv | Old Avestan | (blank) | 31.54 | 53.63 | Old Iranian | Old Avestan corpus (Gāthās, Yasna Haptaŋhāiti, Y27.13–14, Y54.1); point = Glottolog aves1237, not a historical locality | C |

Leaving the coordinates blank is equally defensible; the source gives no locality. 1297 rows carry this tag.
The 65 rows from lemmata printed in square brackets (non-Old-Avestan, e.g. Y27.15, Y42, chapter names)
carry base `Av` only. Optional: register `dialect:Av:YAv:Young%20Avestan` if the Cheung/ESIJa
packages also need it; this package does not emit it because the OAD only says "non-Old Avestan".

## 3. Tags (`tags.py`, `../jambu-static/src/lib/tags.ts`)

- New: **`preverb`** for OAD `pvb.` (15 lemmata: e.g. `hə̄m`, `²pairī`, `ā`, `fra`, `¹auuā`). Not emitted
  until registered. After adding it, set `model.EMIT_PROPOSED_TAGS = True` and rerun
  `import_source.py --install` (the audit already records each withheld tag).
- All other emitted tags already exist: `adj m n f mn adv pron personal demonstrative relative poss
  indef interr prep postp part clitic emph conj num neg interj indecl prefix suffix participle denom
  pres perfect desiderative proper-noun verb alternate uncertain 1sg 2sg 3sg 1pl 2pl pl nom acc dat
  abl instr loc`.
- Deliberately untagged: `act./med.` (voice scopes over senses of a root), `superl./comp.`, and the
  lemma label `(ind.)` on `adə̄` (indicative vs indeclinable is ambiguous) — recorded in the audit.

## 4. `profile_policy.py`

```python
SCHWA |= {"uesugi-oad"}          # Avestan ə/ə̄ are transliteration letters, not house a/ā
KEEP |= {("uesugi-oad", "š"), ("uesugi-oad", "š́"), ("uesugi-oad", "ž")}
```

`uesugi-oad` is a preservation profile (batch decision: Avestan transliteration is preserved; NFC;
only mechanical PDF repairs). Without these entries `profile_policy.py check` would rewrite
`ə→a, ə̄→ā, š→ś, š́→ś́, ž→ź`. `tests/test_uesugi_oad.py` simulates the check with exactly these
additions and passes; it switches to the real tables once they contain them.

## 5. Build-time follow-ups

- Run `make all` once after all four Iranic packages are installed, then un-skip/confirm
  `test_compiled_rows_*` in `tests/test_uesugi_oad.py` (they skip until the dialect is registered and
  `cldf/forms.csv` is newer than the installed CSV).
- `cldf/edges.csv`: expect 171 `derived` rank-1 edges from lemmata to OAD `√root` entries (the
  dictionary prints `(√root)`), 164 `variant` edges, and no reflex/borrowed edges (Parameter_ID blank).
- `cldf/entry-texts.csv`: 1177 blocks resolve by `Entry_Key` (1132 `comparison`, 45 `source-note`
  blocks for `{…}` syllabifications and `v.l.`/`mss.` readings).

## 6. Linguistic review items (not blocking)

- `āh-/ə̄əāh-` (p. 54): second alternant printed `ə̄əāh-` (confirmed on the page image); kept as printed.
- `√²par/frā` (p. 246): `frā` not installed as an alternant (may denote the preverb; cf. "(+ fra)").
- Mechanically expanded shorthand: `pərəϑa/ā-` → `pərəϑa-|pərəϑā-`; `frō.gu/gau-` → `frō.gu-|frō.gau-`.
- 14 printed roots stay unlinked as homograph-ambiguous (e.g. `(√vid)` with √¹/²/³vid, `(√var)`,
  `(√man)`, `(√zan)`), 3 as unresolved (`√hand/had`, `√rā` ×2); 10 abbreviated `°` alternants
  (e.g. `fərašaō°`) are audit-only.
- 7 headers print an unclosed parenthesis (`afšman-`, `aodar-`, `arədra-`, `ārəzuua-`, `paitii-āstar-`,
  `srauuah-`, `zərəd-`); flagged in the audit; only `arədra-` lost a sense (fixed editorially).
- `²auuā` (p. 48) is printed as a tentative analysis of one passage; installed with `uncertain`,
  no gloss.
