# Yoshioka vocabulary recovery (2026-09-14)

Replaces the July 2026 OCR import with deterministic extraction from the native
PDF's embedded Gentium character map. The 626-page TUFS dissertation is identified
by SHA-256 in `manifest.json`; the PDF itself is not redistributed.

## Reproduce

From the data repository:

```sh
.venv/bin/python data/other/forms/raw_data/yoshioka.py --output-dir /tmp/yoshioka-review
.venv/bin/python data/other/forms/raw_data/yoshioka.py --install
```

Both commands use the committed `source-lines.jsonl.gz`; they require no OCR or
network access. `--extract --pdf /path/to/pinned.pdf` rebuilds that snapshot using
pdfplumber/pdfminer and the font's own cmap. It checks the file hash, 626-page
count, all 114 vocabulary pages, and the 3,212 historical entry anchors.

```sh
.venv/bin/python data/other/forms/raw_data/yoshioka_2026/review.py \
  --seed 20260918 --output /tmp/yoshioka-images
```

Image review additionally requires the pinned PDF, pypdfium2 and Pillow. Images
are temporary review aids. `review-samples.json` records the samples inspected,
errors found during development, and final decisions.

The complete follow-up cross-reference review can be rendered with:

```sh
.venv/bin/python data/other/forms/raw_data/yoshioka_2026/review-crossreferences.py \
  --output /tmp/yoshioka-crossreferences
```

`crossreference-decisions.json` records all 57 reviewed index entries (62 forms),
source/target text and page evidence, candidate senses, and hashes of the 28
reviewed image groups. The importer checks source and target row hashes before
applying the decisions; changed extraction or interpretation requires renewed
review. `crossreference-audit.csv` records each emitted form's outcome.

## Representation

- Vocabulary: PDF 505–618, printed CLXXIX–CCXCII. Narrative chapters, example
  sentences elsewhere in the thesis, and the appendix introductions are excluded.
- Every old anchor retains its `yoshioka-entry-N` identity. New indented entries
  and class-separated senses use physical page/line keys; children use stable
  `:variant:N` or `:inflection:N` keys. A source-specific alias table rejoins old
  false fragments without discarding their public IDs.
- `audit.csv` accounts for all 3,267 recovered units: 3,233 accepted source units,
  nine bare headings, five overlapping fragments and twenty continuation lines.
  Accepted units emit 4,886 rows, including full printed inflections/alternants.
- Full grammatical strings remain in `Source grammar:` notes on every emitted
  row from the article. Canonical tags identify the applicable POS, number,
  noun class and aspect. Class-specific suffixes remain verbatim; the importer
  never constructs unprinted words by concatenating a stem and suffix.
- A plural paradigm does not make its headword plural. A standalone `PL` does.
  `SG PL` marks invariant forms; `DOUBLE PL` is a second plural formation, not
  dual number. Explicit argument labels such as `Y.PL.OBJ` describe the verbal
  argument, not the number/class of the verb itself.
- Separate class-specific meanings stay separate. The first `čhu` sense and its
  following Y-class sense share the single reference printed after both.
- Dialect tags are registered, source-specific `dialect:Bur:Yoshioka-*` tags.
  Their location metadata explicitly identifies inherited approximate registry
  points, rather than attributing those coordinates to Yoshioka's fieldwork.
- `B.` citations are indirect references to Berger 1998 **Teil III**; `AA.#`
  citations are ILCAA 1967 questionnaire item numbers. Both remain identifiable
  as “cited by Yoshioka”. `¶` source commentary remains in `Etymology`.
- The original 163 exact, unique `see` resolutions are unchanged. Image review
  of the remaining 57 index entries resolves 56 of their 62 forms to matching
  printed stems/subentries. Overall, 214 index entries are fully resolved,
  two partially resolved and four unresolved. Six ambiguous forms retain the
  literal target, candidate meanings, an empty lexical gloss, `uncertain`, and
  no accepted variant edge. Multi-form index lists are resolved per form.
- Resolved forms retain both index and target citations. Target grammatical
  strings are copied as `Referenced entry grammar:` notes, with the applicable
  POS, number, noun-class and dialect tags. A plural suffix in that grammar does
  not make its headword plural. Only `duqhúlan` requires a reviewed terminal
  stem-hyphen equivalence; its index spelling stays unchanged.
- Donor mentions are not automatically converted to ancestry edges; ten
  source-questioned donor statements remain flagged.
- `conversion/yoshioka.txt` preserves c / č / c̣ as distinct house-transcription
  affricates. Accent, underdot, nasalization, and personal-prefix-slot symbols
  survive. The slot symbols are source notation, not inferred phonemes.

Raw-text and emitted-row hashes in the audit make the interpretation reviewable.
Row hashes use SHA-256 of compact UTF-8 JSON arrays (`ensure_ascii=False`, separators
`,:`). The compressed snapshot is deterministic (`mtime=0`). Original misspellings
in English definitions are retained unless the PDF extraction caused them.

See `source_checklists/20260914-yoshioka-review.md` for verification and deferred
full-build gates. The installed source files have changed; the global CLDF/browser
database is not rebuilt by this importer.
