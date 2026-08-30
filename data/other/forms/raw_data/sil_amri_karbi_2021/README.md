# SIL JLSR 2021-050 Amri Karbi survey

Source package for Binny Abraham and Pronay Daimary's *A Sociolinguistic Study
of Amri Karbi [ajz] in Northeast India*. The survey/comparative-table addendum
of `SOURCE_INGESTION_CHECKLIST.md` is active. The OCR-heavy addendum is not
applicable: Appendix B.3 has a structured Unicode text layer, but that text was
used only as an extraction scaffold and every source response was visually
checked against the rendered PDF.

## Canonical source and scope

- SIL archive: <https://www.sil.org/resources/archives/91601>
- Publisher PDF: `JLSR2021_050.pdf`, 4,428,369 bytes, 165 pages
- PDF SHA-256: `cd121ad102e96b43bf68a1cc5b44f1559c764bc4ae8d71988c6b292a1896ccb1`
- Included: Appendix B.3, physical PDF pp. 37--115, printed pp. 27--105
- Appendix matrix: 307 prompts, 17 printed lists, 5,219 conceptual cells

The PDF is © 2021 SIL International. Its printed fair-use statement permits
scholarly research and instruction but not republication or commercial use.
The PDF is therefore cached outside the data repository and is not checked in.
It was accessed on 2026-08-28.

The report describes 21 collected wordlists, but Appendix B.3 publishes only
17: twelve Karbi lists (`A H M a b c d h k l m s`), three Amri Karbi lists
(`K P S`), one Khasi control (`C`), and one Assamese control (`Z`). Four Amri
Karbi collections reported elsewhere in the report (`e` Tapatuli, `f` Isam
Horo, `g` Guriaghuli Hindu, `w` Guriaghuli Christian) are absent from B.3 and
are not reconstructed. No existing Jambu language or dialect row overlaps the
two proposed base languages.

## Extraction and manual visual review

`extract_amri.py` verifies the exact PDF hash and extent, parses only Appendix
B.3's Unicode text layer, and deterministically reproduces
`extraction_scaffold.tsv`. That scaffold is not itself an import authority.
Every one of the 79 rendered appendix pages and every response on those pages
was manually inspected at 180 dpi. `reviewed_transcription.tsv` records the
page, line, item, site, extracted and verified form, review status, confidence,
and note for all 5,966 printed records. `finalize_review.py` records the already
completed page-by-page decisions; it performs no OCR or image recognition.

The review accounts for all 5,219 conceptual cells and all 5,960 printed
response occurrences. Six cells explicitly print “no entry”: target cells item
36/S, item 37/b, item 50/P, plus control cells item 41/Z, item 53/Z, and item
127/C. No transcription is unresolved or illegible. One excluded Assamese
control response (PDF p. 59, printed p. 49, item 91/Z) visibly prints
`soʌ̆ĭ??`; the two literal question marks are retained as source-marked
uncertainty, while the transcription of what is printed is high-confidence.
The question-like glyph in other forms is the source's glottal stop `ʔ`, not an
uncertainty marker.

## Installation policy and results

The three Amri Karbi lists attach to proposed base language `amri_karbi`
(Glottocode `amri1238`, ISO 639-3 `ajz`); the twelve Karbi lists attach to
proposed `karbi` (`karb1241`, ISO 639-3 `mjw`). Khasi and Assamese are controls
and remain audit-only. The report's lexical-similarity group numbers are kept in
notes but never installed as cognacy, etymology, borrowing, or graph claims.

Some site/item cells print the identical form more than once under different
similarity-group numbers. All 237 such target occurrences remain in the audit,
but only the first exact item/site/form occurrence is installed. The importer
therefore emits 5,092 forms (993 Amri Karbi and 4,099 Karbi), 5,966 audit rows,
and a pinned manifest. All exact source IPA is preserved in NFC in both `Form`
and `Phonemic`. `conversion/sil-amri-karbi.txt` covers every installed grapheme.

## Reproduction

With the canonical PDF cached at
`../tmp/pdfs/amri-karbi/JLSR2021_050.pdf` relative to the outer workspace:

```sh
python data/other/forms/raw_data/sil_amri_karbi_2021/extract_amri.py \
  ../tmp/pdfs/amri-karbi/JLSR2021_050.pdf \
  --output data/other/forms/raw_data/sil_amri_karbi_2021/extraction_scaffold.tsv \
  --review-template data/other/forms/raw_data/sil_amri_karbi_2021/reviewed_transcription.tsv
python data/other/forms/raw_data/sil_amri_karbi_2021/finalize_review.py
python data/other/forms/raw_data/sil_amri_karbi_2021/import_amri.py --install
UV_CACHE_DIR=/tmp/uv-cache uv run --with pytest --with segments pytest -q tests/test_sil_amri_karbi_2021.py
```

The source package intentionally defers shared bibliography, language/dialect
registry, profile routing, consolidated build, full pytest, and browser rebuild
to the coordinating task. Exact proposed integration is in `INTEGRATION.md`.

