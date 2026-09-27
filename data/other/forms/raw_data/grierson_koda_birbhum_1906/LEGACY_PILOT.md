# LSI IV Birbhum Koda, pp. 109–110

This is a five-row source-stage pilot from the original 1906 *Linguistic Survey of India*, volume IV. Grierson credits Rev. P. O. Bodding's 1903 Birbhum specimen. The [Wikimedia Commons scan](https://commons.wikimedia.org/wiki/File:Linguistic_Survey_of_India_Vol_4.djvu) is public domain. The original scan has 701 DjVu pages (SHA256 in `manifest.json`); printed pp. 109–110 are DjVu pages 128–129. The page images here are review evidence; the full scan remains outside the repository.

`audit.jsonl` enumerates all 20 directly glossed candidate examples in the editorial prose. Five visually clear simplex forms are installed. Fifteen are held for uncertain diacritics or because they are grammatical/derived forms. Longer paradigm and sentence illustrations on these pages are excluded as non-simplex examples. The Birbhum interlinear narrative (pp. 111–113), Dhaṅgār comparison (p. 114), and the Bankura material (pp. 114–115) are outside this pilot. Grierson specifically warns that the Bankura specimen is corrupt and partly reconstructed.

The source uses Roman transcription; selected diacritics are preserved in `Original` and mapped conservatively by `grierson-koda-birbhum-1906`. No graph relation is inferred from Grierson's general Aryan-loanword observation about numerals. The source-qualified Birbhum dialect has no invented point coordinate. Existing Koda survey forms remain separate independent attestations unless source dependence is established.

`overlap-review.json` compares the five selected forms against 620 already compiled Koda rows: zero exact form-plus-gloss matches. Related numeral forms in the later Bangladesh survey do not prove dependence on Bodding's Birbhum specimen.

Run `python3 data/other/forms/raw_data/grierson_koda_birbhum_1906/import_source.py --install` from the data repository to regenerate the installed CSV. This does not build CLDF or the browser database. Full pipeline, database and browser gates remain deferred under the user's explicit no-build instruction.
