# Kamioka & Yamada 1979 — proposals for the central integration pass

This package does not edit shared files. Everything below is for the integration pass
(`source_checklists/20261025-iranic-extension.md`, source #3).

## 1. Bibliography (`cldf/sources.bib`)

```bibtex
@book{kamioka-yamada1979lari,
  title        = {L{\=a}rest{\=a}ni Studies 1: Lari Basic Vocabulary},
  author       = {Kamioka, K{\=o}ji and Yamada, Minoru},
  year         = {1979},
  address      = {Tokyo},
  publisher    = {Institute for the Study of Languages and Cultures of Asia and Africa (ILCAA), Tokyo University of Foreign Studies},
  pagetotal    = {xvii + 227},
  included     = {Vocabulary part, printed pp. 1--177: all 1000 numbered questionnaire entries (0001--1000). Head forms and sub-forms with their own printed glosses are installed (1946 rows, 71 of them printed ~ or / alternates); 632 Lari example sentences, the Persian medium-language prompts (left-hand pages), cross-references and the authors' bracketed notes are kept in the audit only; the Lari, English and Persian indexes (pp. 179ff.) and the introduction are excluded},
  provenance   = {Image-only ILCAA scan (232 PDF pages, each a two-page spread; SHA-256 9196621ac485cbbe60cd302712a0d5289e7b2c3ad118446d74361cde0710c569; not redistributed). Reproducible Tesseract OCR with geometric re-attachment of the typewriter overstrike diacritics and an independent second reading, 2026-10-25/26: data/other/forms/raw_data/kamioka_lari_1979/ (ocr_runner.py, import_source.py, audit.csv, sample_review.csv, corrections.csv, manifest.json); installed data/other/forms/20261025-kamioka-lari.csv},
  ocr          = {Yes},
  etymology_provenance = {none},
  jambu_editor = {Aryaman Arora and Anthropic},
  license      = {Copyrighted scan not redistributed; extracted linguistic facts used for scholarship and research}
}
```

Romaskevič, "Lar i ego dialekt" and Eghtedari's *Farhang-e Larestani* are mentioned in the
introduction only (not cited by entries); no auxiliary keys are needed.

## 2. Language and dialect registry

* **New base language** `Lar` — Name `Larestani`, Glottocode `lari1253`, clade `SW Iranian`.
  Coordinates: Glottolog point 27.6436, 54.63449 as quality `C` (the planning doc's family model;
  alternatively derive a `C` centroid from the dialect points once other Larestani sources
  exist). Not historical (no `cldf/historical-languages.csv` row).
* **New dialect** (see `lects.csv`): ID `kamioka-lar`, Tag
  `dialect:Lar:kamioka-lar:L%C4%81r` (= `dialects.dialect_tag("Lar", "kamioka-lar", "Lār")`),
  Language_ID `Lar`, Source_Language_ID `kamioka-lar`, Name `Lār`, Glottocode `nucl1274`
  (Glottolog's dialect "Lari" under lari1253, the town variety), 27.667, 54.333 (town of Lār,
  27°40′N 54°20′E, the elicitation locality), clade `SW Iranian`, Location
  "Lār (old town), Lārestān, Fars province, Iran; elicitation locality", Quality `A`.
  Informants (Mohammad-Bagher Eftekhari, Yusof E'temadi, Yusof Ghoreisi) are provenance only.
* Do not confuse with the existing Sindhi dialect `dialect:S:la:Lari` (lari1254).

## 3. Sound profile (`profile_policy.py`)

`conversion/kamioka-lari.txt` is new. `profile_policy.py check` reports nothing for it. The
profile needs **no** entry in `LONG_A`/`LONG_I`/`LONG_U`/`SCHWA`: the source writes length
itself (`ā` is Persian â; plain `a i u` are printed as such). Optional `KEEP` entries for
documentation: `("kamioka-lari", "ü")` (the front rounded vowel, Introduction 2.3.1 (1)).

## 4. Tags

No new tags. Rows use `uncertain` (typed in the audit as `Review=ocr:unreviewed`),
`alternate` (printed `/`), `sound-variant` (printed `~`), `verb`/`tr`/`intr` (from printed
`(vt.)`/`(vi.)`), and the dialect tag.

## 5. Transcription decisions needing linguistic review

1. **Velar vs uvular g.** The inventory (Introduction 2.1, p. x) lists a voiced uvular /g/
   ([ɢ] initially, [ʁ] otherwise) beside velar /g/, but in this scan both are printed as plain
   `g` (any distinguishing mark is not visible). Forms keep `g`; Persian-cognate ġ/q words
   (domāg "nose") are therefore not distinguished from velar g. Not converted to ɣ.
2. **c, j** are the palatal stops (č, ǰ; Introduction 2.1) → house `c`, `j` (not the Iranist
   dental affricate `c → ʦ` of the planning doc).
3. **Glottal stop** `'` (phonemic, initial and intervocalic; Introduction 2.2 (3)) → `ʔ`.
4. **w → v** per the planning doc; in Lari `w` occurs only as the second element of the
   diphthongs aw, āw, ow (Introduction 2.2 (5)), so `xāw` displays as `xāv`. Review whether the
   house rule should be relaxed for Iranian diphthongs.
5. **Vowels as printed**: `a ā e i o u ü`; `ü` (≈ Persian u) is kept, house transcription has no
   ü of its own. `ā̃` occurs once (0492 "yes", Introduction 2.3.1 (2)).
6. **Stress**: the acute marks fixed initial/irregular stress (Introduction 2.5.1 (2)) and is
   kept on the vowel (`'ána`, `váli`, `cáqad`); 27 forms.
7. **Non-phonemic barred vowels** ɨ, ʉ, ʉ̈ (Introduction 2.4.2) are overstruck bars that the
   OCR cannot see; affected sandhi forms surface as plain `i`/`u`/`ü`. Mostly in example
   sentences (audit only).
8. **š ž** → `ś ź` (house), as in `cdial.txt` for Iranian citations.
9. Unverified oddity: 0318 `jaheda` "jump" contains `h`, which the introduction (2.2 (6))
   restricts to two Persian loans (0579 haram, 0727 hoseyniya); the image was not checked.

## 6. What the integration pass must do

* Add the BibTeX record, the `Lar` language and the `kamioka-lar` dialect above.
* Run `make all` once with the other Iranian packages; confirm `errors.txt` has no
  `20261025-kamioka-lari` conversions, 1946 forms compiled, 1946 `form-source-keys`.
* Run `tests/test_kamioka_lari.py` (the registry test un-skips once `Lar` exists),
  `tests/test_sound_profiles.py`, `tests/test_dialects.py`, then the full suite.
* Optional later work: image review of the remaining `uncertain` rows (OCR estimates below).
