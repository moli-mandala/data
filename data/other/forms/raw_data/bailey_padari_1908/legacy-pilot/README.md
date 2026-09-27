# Bailey 1908 Pādari, bounded glossary column

Thomas Grahame Bailey, *The Languages of the Northern Himalayas* (London:
Royal Asiatic Society, 1908), part III, “Pādari,” printed p. 82 (PDF page
196), right column of the headed “List of Common Nouns, Adjectives and Verbs.”
The original edition is public domain. The
[Internet Archive scan](https://archive.org/details/languagesofnorth00bailrich)
is an ignored local cache at `tmp/pdfs/bailey-sainji/bailey1908.pdf`, 358 PDF
pages, SHA256 `953f5da5ee9bc341cdf7eb558920e824dc6f15f7473a8c0d5c8134c401ad60b5`.
The scan is not installed in Git. The source page was rendered at 300 dpi and
the 28 cells checked visually; OCR is not installed as source evidence.

The bounded scope is **all 28 directly glossed right-column lines**, pig
through river. The left-column list on the same page, the continuation on
p. 83, preceding numeral section, paradigms, and subsequent sentences are
outside scope. `transcription.tsv` and `audit.jsonl` inventory every target
line, keyed by page, column, and local item number; none is inferred from
running narrative. Diacritic, italic and underdot uncertainty is recorded per
cell. Provisional readings marked `(?)` are audit descriptions, not installed
assertions.

Bailey's part III introduction places Padar just north of Pangi and east of
Bhales, and p. 82 labels the lect Pādari. Jambu already has the canonical
`Padri` for Zoller's `Pāḍ.` label; this source is mapped to that record without
inventing a more precise collection site or coordinates. The canonical record
has no Glottocode and retains that classification caveat. Of the 25 existing
source-stage `Padri` rows, Zoller's `rŏṭṭh` “hair” explicitly cites Bailey
1908 p. 82, and his `sugaili` “fox” matches Bailey's p. 82 print lexeme while
citing an LSI comparison. Both are **excluded as same-print evidence**, not
counted as independent Bailey field attestations. The LSI wording/pagination
is a derivative comparator, not a new collection event. No other installed
Padri row duplicates a secure form plus gloss in this bounded column.

Thirteen plain or clearly marked lines are installed; 13 typography-uncertain
lines are held, in addition to the two same-print exclusions. The literal
profile retains secure Roman spelling and macrons. No POS, etymology,
borrowing or variant-direction analysis is inferred. A fresh seeded 20-cell
review against the page image found zero material accepted-reading errors and
no known systematic accepted-glyph error class.

From the Jambu data root, regenerate with
`.venv/bin/python data/other/forms/raw_data/bailey_padari_1908/import_source.py --check-pdf --install`.
Focused tests cover regeneration, full audit/locator counts, local parsing,
metadata, and profile coverage. The full CLDF, graph and reference builds,
full test suite, and browser QA are deferred by the user's explicit no-build
instruction.
