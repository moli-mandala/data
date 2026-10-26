# Uesugi & Catt, *Old Avestan Dictionary* (ILCAA, AAL 67, 2024)

Heindio Uesugi (ed.) & Adam Alvah Catt (editorial supervisor), *Old Avestan Dictionary* (Asian and
African Lexicon 67; Research Institute for Languages and Cultures of Asia and Africa, Tokyo
University of Foreign Studies, 2024; ISBN 978-4-86337-542-0). The copyright page states
**CC BY 4.0**. Citation key `uesugi-catt2024oad` (BibTeX record proposed in `proposals.md`).

## Source and scope

- PDF `B542_AAL67_OldAvestan.pdf`, 558 pages, SHA-256
  `7bb39fa3f87a5bc1400bc7243ad6da4d203ea8bad10765d13fe8a8045851b5f9` (Adobe Acrobat Pro 24.5,
  born-digital text layer). Not redistributed; the importer fails clearly without it
  (`--pdf` / `$OAD_PDF`).
- **Included:** Part I Dictionary, printed pp. 1–404 = PDF pp. 30–433 (asserted: every running head
  except p. 1 prints page = PDF − 29).
- **Excluded:** front matter (PDF 1–29), Part II Text and Translation (PDF 434–558), and inside
  entries: inflected attestations, Gāthā passages and translations, Vedic/Sanskrit and other
  comparanda (kept as audit/entry-text prose only).

## Extraction (`extract.py`, `segment.py`)

pdfplumber characters, columns split at the page midpoint, lines clustered by `top` (±5 pt).
Combining marks are positioned glyphs, sometimes detached (`aš ̣a`, `ā ̊`) or out of stream order;
each is re-attached to its stream predecessor (or geometric base), then NFC (`aṣ̌a`, `ā̊`). Fonts
give structure: `◊` (Verdana) non-verb lemma, `♣` verb root, Wingdings 3 `9` attestation,
Wingdings `F` cross-reference, bold headword, bold-italic F-line forms, < 8 pt superscripts
(homograph numbers `¹²³⁴`, corrected-reading `⁺`). A cache keyed by extractor version + PDF hash
lives in `../tmp/iranic/oad/` (outside the repo).

## Units → audit → rows

| raw unit | count | audit status | installed |
|---|---:|---|---:|
| lemma `◊` + verb `♣` | 997 + 202 | `installed_lemma` 1154, `installed_lemma_editorial` 44, `audit_subsection` 1 | 1198 |
| printed alternants (`X/Y`, `X, Y`, `X or Y`, `(or X)`, `X (m.), Y (f.)`) | 175 | `installed_variant` 164, `audit_variant_not_installed` 11 | 164 |
| attestation `9` | 2274 | `audit_attestation` | 0 |
| cross-reference `F` | 331 | `audit_cross_reference` | 0 |
| verb-stem sub-heading | 243 | `audit_stem_heading` | 0 |
| section heading | 29 | `audit_section_heading` | 0 |

1362 installed rows; `audit.csv` has 4251 rows (one per unit plus one per alternant), each with
the raw text, page/column, parsed fields, review notes and a SHA-256. No orphan units: every
attestation and stem heading carries its lemma key.

## Modelling

- `Language_ID=Av`; lemmata outside square brackets carry the proposed dialect tag
  `dialect:Av:OAv:Old%20Avestan`; bracketed (non-Old-Avestan) lemmata carry no dialect tag.
- `Form`/`Original` = printed headword (NFC), including homograph superscripts (`√¹ah`) and stem
  hyphens (stripped from display by the build). `Entry_Key = uesugi-catt2024oad:<headword>`;
  unnumbered printed homographs add `:p<page>[:<order on page>]`; variants add
  `:alt:<form>` / `:or:<form>`.
- `Gloss` = depth-0 quoted senses of the header joined with `; `. Collection stops at `→`, at the
  first comment/attestation, and at prose that cites another (bold) word. 44 entries with
  unquoted definitions, names of texts, or source typography problems use `editorial.csv`
  (each row states its reason). 18 lemma rows (plus 1 alternant) keep an intentional blank gloss (particles, pronouns,
  affix elements with only functional descriptions).
- `Tags` from printed labels only (`LABELS` in `parse.py`); `♣` lemmata get `verb`. `preverb` is
  proposed and withheld (15 lemmata).
- `Derivation_Parent_Keys`: only when the header prints a top-level `(√root)` / `(√root ‘gloss’)`
  that resolves to exactly one OAD root headword (homograph number respected; printed slash
  alternants must agree): 171 linked, 14 ambiguous, 3 unresolved (audit `Review`). Compound
  analyses (`aibī + √gar`), nested and Vedic roots are not linked. `Parameter_ID` is blank (no
  printed etymological IDs).
- Entry texts: header residue (comparanda, reconstructions, printed roots) as `comparison`;
  `{…}` syllabifications and `v.l.`/`mss.` readings as `source-note`.

## Transcription

`conversion/uesugi-oad.txt` is a preservation profile: identity mappings for Hoffmann's
transliteration as printed (`a ā ā̊ å ą ą̄ ą̇ b β c d δ e ē ə ə̄ … š š́ ṣ̌ t t̰ ϑ … x x́ xᵛ y ẏ z ž ḿ`, `ŋ ŋ́
ŋᵛ`), capitals for names, `√`, homograph superscripts and notation marks. Every installed form
tokenizes without `�`. `profile_policy.py` needs `SCHWA`/`KEEP` entries (`proposals.md`).

## Seeded audit

`sample.csv` (seed `uesugi-oad-2026-10-26-r4`, 20 lemmata) with verdicts in `sample-review.csv`:
**0/20 material errors**. Earlier rounds found 1/20 each (unquoted primary definition; unclosed
printed parenthesis; quotes of cited words taken as senses); each class was fixed before the next
fresh draw (`manifest.json` → `seeded_sample_history`). Edge cases checked: pp. 1 and 403–404,
40 headers crossing column/page breaks, homographs (`ā` ×5, `mat̰`, `ϑβā`, …), multi-line headers.

## Commands

```bash
.venv/bin/python -I data/other/forms/raw_data/uesugi_oad_2024/import_source.py            # dry run
.venv/bin/python -I data/other/forms/raw_data/uesugi_oad_2024/import_source.py --install
.venv/bin/python -m pytest -q tests/test_uesugi_oad.py
```

Deferred to the integration pass: registry/tag/profile-policy/bib changes (`proposals.md`), the
single `make all`, compiled-CLDF checks and the full test suite.
