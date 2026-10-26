# ESIJa (`20261025-esija`) — proposals for the central integration pass

Package: `data/other/forms/raw_data/esija_2000/`. Nothing below has been applied to shared files.

## 1. `cldf/sources.bib` — update `rastorgueva2000esija`

The record exists (currently `included = {No; cited indirectly ...}`). Proposed fields:

```bibtex
@book{rastorgueva2000esija,
  title        = {\`{E}timologi\v{c}eskij slovar' iranskix jazykov},
  author       = {Rastorgueva, V. S. and Edel'man, D. I.},
  publisher    = {Vosto\v{c}naja literatura},
  address      = {Moscow},
  year         = {2000--2015},
  volume       = {1--3 used (a--\={a}, 2000; b--d, 2003; f--h, 2007)},
  included     = {Partial: reconstructed Proto-Iranian lemma layer of vols. 1--3 (see manifest for counts);
                  English glosses are an editorial translation of the Russian; reflex citations not installed},
  provenance   = {data/other/params/20261025-esija.csv; data/other/forms/raw_data/esija_2000/20261025-esija-audit.csv;
                  also cited through data/other/params/20260827-wiktionary-piir.csv},
  ocr          = {Yes},
  etymology_provenance = {source},
  jambu_editor = {Aryaman Arora and Anthropic Claude (OCR repair, English translation of glosses)}
}
```

Locators are `rastorgueva2000esija[vol. N, p. P]` (printed page of the article's headword).

## 2. Language registry

Only `PIr` (already in `cldf/languages.csv`) is used by installed rows. `lects.csv` lists the
ESIJa language abbreviations with proposed mappings for a future reflex stage; rows marked
`proposed` need the registry pass (Bactr, Tumsh, Taj, Tal, Gil, Maz, Semn, Gur, Zaza, Lar, Kumz,
Bashk, Lur, Tati and the dialect tags shown). No dialect tag is emitted now.

## 3. Tags / profile policy

- No new tags. Review state is kept in the audit `Review` column (`unreviewed-ocr`, plus
  `variants-unverified`, `gloss-blank`, `source-query` where applicable); the params schema has no
  tag column, so the integration pass may want an etymon-level `uncertain` flag source.
- No conversion profile: `data/other/params/20261025-esija.csv` is not in `make_cldf.convertors`, so
  `Form` is compiled as printed. ESIJa's PIr notation is kept: `ā ī ū`, `č ǰ š ž ś ź`, `ϑ`,
  glides `i̯ u̯` (U+032F), syllabic `r̥` (U+0325), `xᵛ hᵛ` (U+1D5B), accent `á í ú` where printed,
  `(a)`-type optional segments, final `-` for roots/stems. **Linguistic review:** whether PIr
  etyma from ESIJa and from Cheung's EDIV should share one house notation (e.g. `i̯/u̯` vs `y/w`,
  `ϑ` vs `θ`, `ź` vs `ʒ`). This package does not impose one.

## 4. Lemma decisions needing review

- `Form` is the **first printed lemma** of the headword line. Ablaut/variant lemmata (`*das- : dās-`,
  `*X, *Y`, `*X / *Y`, `*X > *Y`) are kept in the audit (`Fused_Headword`, `Raw_Headword_*`) but not
  installed as separate etyma; a slash *inside* a word (`*ap/fšah-`) is notation and stays in the form.
- Homonym superscripts become an ID suffix (`esija-1-79-ad-2`), not part of `Form`.
- Printed see-reference entries (`*X см. *Y`) are audit-only (`cross_reference`).
- Lemmas the OCR pass disputed or could not read were read manually from headword crops
  (`manual-lemmas.csv`, 241 installed); two remain excluded as ambiguous and three detected
  "articles" were printer's signature lines (see README). Printed breves (`ă ĭ ŭ ā̆`) are kept.
- The article's Aryan/PIE comparison prose ("из арийск. *…, ср. др.-инд. …") is Latin-script text in
  the ABBYY Russian layer and is as garbled as the reflexes; it is kept raw in `Article_Head_Raw` and
  no entry-text sidecar is emitted.

## 5. Reflex stage (deferred) — blocking error classes

Every article is in the audit with `Reflex_Stage=deferred` and its printed page span. Reflex
installation was not attempted because the only text source for the Latin-script citations is the
ABBYY Russian-mode layer, which shows these systematic error classes (examples from the layer):

1. Latin letters replaced by Cyrillic look-alikes or digits: `cp.-nepc.` for `ср.-перс.`, `аца-`
   for `aṷa-`, `5`/`8`/`§`/`$` for `š`/`s`, `£`/`{`/`б` for `č`, `]` for `ǰ`.
2. Diacritics lost or substituted: macrons as `ä ü ï` or dropped; `ś/š/s`, `ź/ž/z` collapse;
   `δ ϑ γ β` as `ô ö û & 5 6 ß`; subscript `i̯ u̯ r̥` lost entirely.
3. Superscript homonym numbers, `ᵛ` and Manichaean/Pahlavi transliteration braces split into
   separate tokens or merged into neighbours.
4. Language-abbreviation boundaries and reflex/gloss/reference scoping depend on Russian prose that
   the parser would have to interpret (e.g. "то же", "в сочет.", "из перс.").

A reflex extraction would need its own re-OCR of every italic Latin span (Tesseract
`script/Latin` plus the diacritic geometry used here for lemmata) with a seeded 0/20 audit; on the
lemma layer that pipeline needed five audit rounds to reach an acceptable error rate on bold,
isolated headwords, and italic running text is harder. Deferred.

## 6. `data/other/params/20261025-esija.yaml` (created; validated with `source_meta.py`)

Content:

```yaml
file: 20261025-esija.csv
sources:
  rastorgueva2000esija:
    reference:
      editor: "Aryaman Arora; Anthropic Claude"
      ocr: true
      etymology_provenance: source
```

Check first that no other YAML already declares `rastorgueva2000esija` (the Wiktionary PIIr
params cite it inside citation strings only).

## 7. Integration checklist for this package

1. Apply §1 (bib); the params YAML (§6) exists. No `make_cldf.py`, `tags.py` or `profile_policy.py` change is needed.
2. One `make all` after all Iranic packages; confirm `cldf/parameters.csv` has 587 `esija-*` rows
   with `Language_ID=PIr`, Source `rastorgueva2000esija[vol. N, p. P]`, and that `errors.txt` has no
   `20261025-esija` line (the file is not routed to a converter).
3. Confirm the Cheung EDIV layer and this layer stay distinct records (no merge on identical `Form`).
4. Run `tests/test_esija.py` plus the full suite.
