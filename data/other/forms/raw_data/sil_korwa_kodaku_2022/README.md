# SIL Korwa and Kodaku survey (JLSR 2022-014)

This package installs the Korwa and Kodaku target lists in Gangadhar Behera's
*A Sociolinguistic Profile of Korwa and Kodaku Tribes in Chhattisgarh and
Jharkhand, India*. The report is SIL's *Journal of Language Survey Reports*
2022-014. Fieldwork took place in 2004–2005; the report was published in 2022.

## Canonical source and representations

- SIL publication record: <https://www.sil.org/resources/publications/entry/94564>
- canonical PDF: <https://www.sil.org/system/files/reapdata/13/03/86/13038659512317049919318473837327540493/JLSR2022_014.pdf>
- pinned archived retrieval: <https://web.archive.org/web/20240617131527id_/https://www.sil.org/system/files/reapdata/13/03/86/13038659512317049919318473837327540493/JLSR2022_014.pdf>
- SHA-256: `a8efbe88405e27024a7a6ec786cd6fde3e382f0eaf0d0081197d3880ed97eb0c`
- 2,198,621 bytes; 115 pages

The checked source remains at
`tmp/pdfs/korwa-kodaku-2022/source.pdf`; the PDF is not redistributed here.
Appendix B.5 occupies physical PDF pages 66–90 (printed pages 56–80).

The appendix is typeset, not handwritten or raster-only, and has an embedded
Unicode text layer. `text_layer_scaffold.txt` and `tesseract_scaffold.txt` are
independent navigation/comparison aids. Neither is parsed by the installer.
`manual_review.tsv` is authoritative after direct visual comparison with every
rendered page; `page_review.tsv` records completion for all 25 pages.

## Scope and manual review

The source compresses 25 lists × 210 prompts into response rows whose brackets
name all sites sharing that spelling and similarity group. The review expanded
and audited all 5,250 conceptual site/item cells:

- 18 target lists: nine Korwa and nine Kodaku;
- seven excluded comparison lists: Asuri, Birjia, Mundari, Tanmai, and three
  Sadri lists;
- 2,900 printed response lines visually checked;
- 3,780 target cells audited: 3,730 attested and 50 blank/unlisted;
- 1,470 control cells audited: 1,453 attested and 17 blank/unlisted;
- 4,458 target rows installed after deduplication within a source cell and
  expansion of the one slash alternative.

Items 23 `urine` and 24 `feces` say `NO ENTRY` for all lists. Item 4 assigns
`NO ENTRY` specifically to C, D, H, and L. Thirteen further known site/item
slots have no response printed and remain explicit unlisted blanks.

There are no clipped, illegible, or ambiguous installed readings. Two source
brackets contain unidentified lowercase codes not defined in Appendix B.4 or
the 25-list matrix: `u` at PDF p.73/item 83 and `n` at PDF p.84/item 173. Their
forms are transcribed in `unresolved_source_codes.tsv`, excluded, and never
silently reassigned to a known site.

Similarity-group labels are retained only in Notes and audit metadata. They are
survey similarity judgments, not etymologies or cognate sets; every installed
row therefore leaves `Parameter_ID`, `Cognateset`, and `Etymology` blank.

## Reproduction and checks

```sh
sh data/other/forms/raw_data/sil_korwa_kodaku_2022/render_wordlist.sh
python data/other/forms/raw_data/sil_korwa_kodaku_2022/extract_scaffolds.py
# Do not run build_review_scaffold.py without repeating the complete visual review.
python data/other/forms/raw_data/sil_korwa_kodaku_2022/import_korwa_kodaku.py --verify-pdf --install
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_sil_korwa_kodaku_2022.py
```

Shared bibliography, language/dialect registration, profile routing, full
build, full test suite, and browser QA are intentionally deferred and specified
exactly in `INTEGRATION.md`.
