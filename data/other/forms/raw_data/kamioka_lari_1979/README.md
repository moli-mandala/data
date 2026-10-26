# Kamioka & Yamada 1979, *Lari Basic Vocabulary*

Kōji Kamioka & Minoru Yamada, *Lārestāni Studies 1: Lari Basic Vocabulary*. Tokyo: ILCAA, 1979.
Citation key `kamioka-yamada1979lari`; installed as `data/other/forms/20261025-kamioka-lari.csv`
(`append_order: 114`, profile `conversion/kamioka-lari.txt`). Language `Lar` (Larestani),
dialect `dialect:Lar:kamioka-lar:L%C4%81r` (the town of Lār, where the data were recorded in
1977–78). Shared decisions: `source_checklists/20261025-iranic-extension.md`; registry, bib and
profile-policy proposals: `proposals.md`, `lects.csv`.

## Source and scope

* Scan: image-only, CCITT 300 dpi, 232 PDF pages, `/Rotate 270`; SHA-256 and page count are
  asserted by `ocr_runner.py`. Not redistributed; the runner fails clearly when it is absent.
* Each PDF page is a **two-page spread**: left = Persian medium-language prompts (Persian
  digits), right = numbered English concept, underlined typewritten Lari, sub-forms with quoted
  glosses, example sentences. Printed page = PDF page − 9 (PDF 10–186 = pp. 1–177; 144 folios
  read by OCR, 143 agree, PDF 121 misread).
* Front matter PDF 1–9 (Foreword, Introduction with the phonemic inventory, symbols); indexes
  PDF 187–232 (excluded). The questionnaire is ILCAA's 1000-item list; entries 0001–1000 are
  all present exactly once, in sequence.

## Extraction (`ocr_runner.py`, pinned `PARAMS`, cache in `../tmp/iranic/lari/ocr/`)

The typewriter diacritics are overstrikes that Tesseract misreads, so the runner:

1. renders the spread at 300 dpi and splits it;
2. detects underlines (long thin runs; underlined text is Lari) and erases them;
3. detects overstrike marks geometrically (small components directly above a letter), records
   them with their shape bitmap, and erases all but i/j dots;
4. reads the cleaned right page with `tesseract -l eng --psm 6` (TSV word boxes), the raw right
   page again with `--psm 4` (independent second reading) and the left page with `-l fas`;
5. records each word's glyph boxes and descender centroid.

`layout.py` re-attaches the marks by monospace pitch (classes: macron, diaeresis, caron,
acute/grave; a caron-sized mark with ink only at its top right is the stress acute) and repairs
known OCR classes from page geometry: the apostrophe (glottal stop) read as t/i/l/j or followed
by a spurious t (a glyph whose ink lies above the x-height is `'`), q/g (the q descender is a
right-hand stem: centroid 77–83 % vs 38–49 % for g), j/l (no descender). `import_source.py`
additionally takes from the raw-image reading the apostrophe where the cleaned reading lost it,
q where the two readings disagree on q/g, and the lower-case reading of an upper-case Lari word.
Every repair is listed per unit in `audit.csv` (`Apostrophe_Repairs`); mechanical repairs inside
Lari words (digit→letter, doubled initial `sSina`, `_`) are in `clean_form`. Image-verified hand
corrections (14 fields in 12 units) are in `corrections.csv` with evidence.

## Parsing and modelling

* Head line: number → English concept (gloss; `/`-equivalents such as `fur/fourrure` go to
  Notes, `(vt.)`/`(vi.)` to tags) → Lari items split at top-level commas; `;` pairs Lari groups
  with `;`-separated senses; a quoted parenthesis is the item's own gloss, a short unquoted one a
  qualifier, a longer one a note; `cf.`/`see` cross-references go to the audit.
* Sub-line: Lari up to the opening quotation mark, then the quoted gloss. A sub-line is an
  **example sentence** (audit only) when the Lari ends in `. ? !`, contains a sentence break, or
  translates as `idem.` or as a capitalised phrase of ≥2 Lari words. Unglossed sub-forms are
  installed only when wholly underlined and confidently read (Notes name the questionnaire entry).
* Printed `~` → `sound-variant` row, printed `/` → `alternate` row (word-level substitution:
  `goš kerda/dada` = `goš kerda`, `goš dada`), both with `Variant_Of_Key`.
* `Entry_Key` = `kamioka-yamada1979lari:<entry>:<n>` (head items), `:<entry>:sub:<n>[:<m>]`
  (sub-forms), `:v<k>` (variants); locator `kamioka-yamada1979lari[p. N, entry NNNN]` with the
  printed page of the line. No etymological claims (`Parameter_ID` blank). Persian prompts are
  questionnaire medium, kept in the audit (`Persian_OCR`), not in Notes; `Native` is blank.

## Counts (manifest.json)

2446 audit units → 1527 installed units → **1946 rows** (1360 head-item rows, 586 sub-form rows;
71 variant rows). Not installed: 632 example sentences, 23 heads without a Lari form (blank in
the source, `(nonexistent)`, or forms only in sentences), 3 cross-reference-only heads, 2 authors'
notes, 115 OCR noise lines (drawings, margins, folio fragments), 144 folio lines.

## Review state and residual risk

* 9 seeded rounds of 20 units each were checked against small page crops
  (`sample_review.csv`; the published `sample.csv` is round 9). Every finding was traced to a
  class and fixed in the code (glottal stop ×4, English OCR, digit, gloss structure, item split,
  upper-case word, j/l, repair scope, q/g, hyphenation) or, for isolated misreadings, in
  `corrections.csv`; all 180 checked units now match the image. The last round still found 2
  isolated letter misreadings (n→h, a spurious hyphen), so roughly 5–10 % of unreviewed units
  may contain a letter-level OCR error.
* Rows from image-checked or corrected units (221) carry no review tag; all others carry
  `uncertain` with `Review=ocr:unreviewed` in the audit.
* Not recoverable from this scan: velar vs uvular g; the non-phonemic barred vowels; diaeresis
  or macron lost where the overstrike is faint (OCR has no second source for them).

## Reproduce

```bash
cd data
nice -n 10 env OMP_THREAD_LIMIT=1 .venv/bin/python -I \
    data/other/forms/raw_data/kamioka_lari_1979/ocr_runner.py --pages 10-186   # cached
.venv/bin/python -I data/other/forms/raw_data/kamioka_lari_1979/import_source.py  # dry run
.venv/bin/python -I data/other/forms/raw_data/kamioka_lari_1979/import_source.py --install
.venv/bin/python -m pytest -q tests/test_kamioka_lari.py
```
