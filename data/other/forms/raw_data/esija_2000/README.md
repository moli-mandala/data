# ESIJa lemma layer (`20261025-esija`)

Source: V. S. Rastorgueva & D. I. Edel'man, *Ètimologičeskij slovar' iranskix jazykov*,
vols. 1 (a–ā, 2000), 2 (b–d, 2003), 3 (f–h, 2007). Moscow: Vostočnaja literatura.
Key `rastorgueva2000esija`; locators `rastorgueva2000esija[vol. N, p. P]` (printed page of the
headword). Scan: 1324-page PDF with an ABBYY FineReader Russian text layer (sha256 in the
manifest); copyrighted, not redistributed.

## What is installed

`data/other/params/20261025-esija.csv` — one Proto-Iranian (`PIr`) etymon row per accepted
article: `ID, PIr, Form, Gloss, Source`.

- **Form** = the first printed lemma of the headword line, in ESIJa's own notation (see
  `proposals.md` §3), prefixed with `*`. Homonym superscripts go into the ID, not the form.
- **Gloss** = English translation of the article's opening Russian gloss(es). The translation is an
  editorial step by the Jambu editors (table `glosses_en.csv`, Russian original kept in the audit).
- **ID** = `esija-<vol>-<printed page>-<ascii lemma>[-<homonym>]`, minted once and frozen in
  `ids.csv` (keyed by `Entry_Key = rastorgueva2000esija:v<vol>:p<page>:h<n>`, n = n-th headword
  on the printed page), so later OCR or gloss corrections never re-mint IDs.

Lemma provenance (audit `Lemma_Source`, `Review`):

- `ocr` (346 rows, `unreviewed-ocr`): the fused Tesseract+geometry reading agreed with the ABBYY
  letter skeleton (plus `variants-unverified`, `gloss-blank`, `source-query` where applicable).
- `manual` (241 rows, `manual-collation`): every headword the OCR pass disputed or could not read
  was read by eye from the cached headword-line crop (wider re-crops for 6 lines whose segment was
  truncated) and recorded in `manual-lemmas.csv` (`Entry_Key, Decision, Form, Homonym, Gloss_RU,
  Note, Reviewer`). The importer applies it reproducibly; IDs stay frozen in `ids.csv`. Printed
  breves (`ă ĭ ŭ`, `ā̆`) and word-internal `/` are kept as printed. For 16 rows the ABBYY gloss parse
  failed and the gloss was read from the article opening (`Gloss_RU` override).

## Counts (760 articles detected in the dictionary body)

| Volume | articles | installed (ocr + manual) | cross-reference | layout false positive | ambiguous |
|---|---|---|---|---|---|
| 1 (pp. 63–327) | 240 | 190 | 49 | 1 | 0 |
| 2 (pp. 40–502) | 266 | 205 | 59 | 1 | 1 |
| 3 (pp. 44–493) | 254 | 192 | 60 | 1 | 1 |
| total | 760 | 587 (346 + 241) | 168 | 3 | 2 |

Not installed: `cross_reference` (printed `*X см. *Y` see-entries; one found manually),
`layout_false_positive` (three "articles" that were printer's signature lines, not headwords),
`excluded_ambiguous` (`v2:p261:h1` *²či(?)am-: glide-mark placement not resolvable in the crop;
`v3:p401:h1` *hāriši-/*hārišī-: final vowel length not resolvable). Raw readings and locators stay
in the audit.

## Extraction

1. Page mapping: printed = PDF+1 (vol. 1, PDF 1–327), PDF−326 (vol. 2, 328–830), PDF−829
   (vol. 3, 831–1324); verified by OCR of every printed folio (no missing/duplicate leaves).
2. Article boundaries: articles open with a hanging (outdented) bold lemma line; derivative
   paragraphs are indented. Lines come from the ABBYY layer via pdfplumber; the left margin is fitted
   per page (skew-corrected); split layout lines of one headword open one article.
3. Lemma reading: each headword segment (up to the gloss) is cropped at native scan resolution and
   read with Tesseract `script/Latin` (hOCR character boxes); a connected-component pass adds
   macrons, háček/acute, `i̯ u̯ r̥` under-marks, homonym superscripts and `ᵛ`. ABBYY supplies the
   letter skeleton check, superscript tokens (by font size) and hyphenation.
4. Acceptance: only lemmas whose first form agrees with the ABBYY skeleton and passes the
   alphabet/diacritic checks are installed.
5. Glosses: the quoted Russian glosses that open the paragraph and follow each other directly;
   later quotes (comparanda) are excluded. A dash-delimited description is used when no quote exists.

Commands (from the data git root; cache in `../tmp/iranic/esija`, override with `ESIJA_CACHE`):

```
.venv/bin/python -I data/other/forms/raw_data/esija_2000/import_source.py --extract   # needs the PDF
.venv/bin/python -I data/other/forms/raw_data/esija_2000/import_source.py             # dry run
.venv/bin/python -I data/other/forms/raw_data/esija_2000/import_source.py --install
```

## Audit history (seeded samples of installed rows, 20 each, checked against headword crops)

1. Consensus readings: 1/20 (lost homonym when ABBYY dropped the superscript) → leading digits = homonym.
2. 5/20: acute accent read as macron; geometry missed ź; gloss+`см.` entries; gloss from a
   split layout line; homonym lost → accent kept, sibilant geometry, cross-reference rule, anchoring.
3. 4/20: ź/ž threshold; word-internal `/`; false `r̥` from a `g` descender; hyphen dropped as
   "superscript" → thresholds 0.16, slash rule, neighbour-descender guard, hyphen fix.
4. 2/20: subscript arc cut by crop trimming; `ū̆` → wider crop padding, projection baseline,
   implausible geometry → disputed.
5. Final OCR round (seed 20261029): 0/20 material errors after fixing the last class found while
   redrawing (ABBYY `J`/`6` superscript and `č` look-alikes).
6. Manual round (seed 20261030, 20 rows drawn only from `manual` rows; crops re-read
   independently; glosses checked against the article opening): 0/20 material errors.
   Both rounds are in `20261025-esija-sample.csv` with verdicts from `sample-review.csv`.

Known residual risks: in the OCR rows, `ū̆`-type double marks are not modelled and variant lemmata
after the first are unverified; eight installed rows have an intentionally blank gloss.

## Not installed

Reflex citations (deferred; `proposals.md` §5), the Aryan/PIE comparison prose (Latin spans are
garbled in the ABBYY layer; raw text kept in `Article_Head_Raw`), the front matter, abbreviation
lists and bibliographies.

Files: `import_source.py`, `esija_ocr.py`, `20261025-esija-audit.csv`, `20261025-esija-sample.csv`,
`sample-review.csv`, `manual-lemmas.csv`, `20261025-esija-manifest.json`, `ids.csv`,
`glosses_en.csv`, `lects.csv`, `proposals.md`; params settings `data/other/params/20261025-esija.yaml`;
tests `tests/test_esija.py`.
