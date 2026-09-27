# Bailey 1908 Bhalesi comparison list

Thomas Grahame Bailey, *The Languages of the Northern Himalayas* (London:
Royal Asiatic Society, 1908), part III, “Bhalesi,” printed pp. 73–74 (PDF
pages 187–188). The original edition is public domain. The
[Internet Archive scan](https://archive.org/details/languagesofnorth00bailrich)
is an ignored local cache at `tmp/pdfs/bailey-sainji/bailey1908.pdf`, 358 PDF
pages, SHA256 `953f5da5ee9bc341cdf7eb558920e824dc6f15f7473a8c0d5c8134c401ad60b5`.
The scan itself is not installed in Git. The pages were rendered at 300 dpi
and all list entries checked visually; OCR is not installed as source evidence.

The bounded scope is the **complete 34-line directly glossed Bhalesi list**:
printed p. 73, bottom, left and right columns (father through husband), and
p. 74, top, left and right columns (wife through ass). The subsequent numbered
sentences, preceding grammar, and adjacent Bhadrawahi/Padari/Pangwali data are
excluded. `transcription.tsv` and `audit.jsonl` have one unit for every printed
line, with exact page, column, and local item locator. Father has two printed
answers on one line and yields two answer keys; other repeated English glosses
are separate printed lines, not silently merged. The horse/mare stem and
suffix are held because the free forms are not directly printed.

Bailey identifies Bhalesi separately from Bhadrawahi and Padari and locates
it in a valley east of Bhadrawah. These forms map to existing canonical `bhal`
(Bhalesi); no more precise site or coordinates are asserted. Of the existing
52 source-stage `bhal` rows, Zoller's `kuǖś` “woman” explicitly cites LSI
IX(IV) p. 886. That comparative table reproduces the Bhalesi list and is
derivative of Bailey's print evidence, so Bailey's corresponding woman cell
is **excluded as same-print evidence** as well as typographically unresolved.
Other LSI p. 886 Bhalesi forms are comparators only, not independent field
records installed here. No other current `bhal` source row matches a secure
Bailey p. 73–74 form and gloss. No POS, etymology, or variant-direction
analysis is inferred.

The 34 lines yield **15 secure cells and 16 rows**, with 16 typography holds,
one same-print exclusion, one complex stem hold, and one incomplete suffix
hold. Secure macrons are retained. Tentative forms marked `(?)` in the TSV
are audit descriptions, not installed readings. A fresh seeded 20-line review
was checked against the page images; it found zero material errors in accepted
readings and no known systematic accepted-glyph error class.

From the Jambu data root, regenerate with
`.venv/bin/python data/other/forms/raw_data/bailey_bhalesi_1908/import_source.py --check-pdf --install`.
Focused tests cover regeneration, audit coverage, locators, local parsing,
metadata, and the literal sound profile. The full CLDF, graph and reference
builds, full test suite, and browser QA are deferred under the user's explicit
no-build instruction.
