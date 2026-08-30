# Bonda further survey 2022 — installed, consolidated build pending

This package contains the exhaustive manual source-local extraction of Chacko Mathew's *The Bonda: Further Sociolinguistic Survey* (JLSR 2022-005). Its 644 target rows are installed at `data/other/forms/20260829-sil-bonda-further.csv`; shared bibliography, dialect rows, profile routing, and the checked-Dumripada replacement policy are applied. Generated CLDF, graph validation, and browser QA remain deferred to the requested consolidated build.

## Source and corrected topology

- Official archive: SIL 92609.
- Pinned canonical PDF: `tmp/pdfs/bonda_further_2022/JLSR2022_005.pdf`, 1,247,227 bytes, 74 pages, SHA-256 `9c4457aa6e73906b34e8c69e790e9d205a9b95cfc2a94ccae054bcb1537dfcfa`.
- Appendix A: physical pp. 15–47 / printed pp. 10–42.
- Correct denominator: 210 prompts × 11 lists = 2,310 cells, not the earlier 15-list/3,150-cell estimate. Three locality/language labels wrap over two lines but each has one response.
- Target scope: Podeiguda, Bondapada, and the checked Dumripada Upper Bonda lists, 630 cells. The report calls Podeiguda and Bondapada new and says checked Dumripada replaces the prior report's list.
- Comparison scope: eight same-list comparanda from JLSR 2022-004, 1,680 cells, audit-only.

## Complete source-local census

Physical pp. 15–47 / printed pp. 10–42 / items 1–210 are complete: 2,310 manually reviewed cells, comprising 2,259 attested cells, seven explicit `no entry` source blanks, and 44 explicit `DISQUALIFIED` exclusions at items 11, 23, 24, and 70, with zero ambiguities or illegibles and 2,394 expanded responses. The three target lists contribute 630 conceptual cells and 644 staged forms; the eight comparison lists contribute 1,680 conceptual cells and 1,750 audit-only responses.

All cells were hand-keyed from rendered images at 600 dpi and rechecked in targeted 1200-dpi crops. PDF text extraction was used only for nonlexical report prose and locating item/page ranges; it did not supply or verify forms. Every ledger row carries `hand-keyed-from-rendered-source; OCR-not-copied`.

## Reproduce

From the workspace root:

```sh
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_001_005.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_006_010.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_011_015.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_016_020.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_021_025.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_026_030.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_031_035.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_036_040.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_041_045.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_046_050.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_051_055.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_056_060.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_061_065.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_066_070.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_071_075.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_076_080.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_081_085.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_086_090.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_091_095.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_096_100.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_101_105.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_106_110.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_111_115.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_116_120.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_121_125.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_126_130.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_131_135.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_136_140.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_141_145.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_146_150.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_151_155.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_156_160.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_161_165.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_166_170.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_171_175.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_176_180.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_181_185.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_186_190.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_191_195.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_196_200.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_201_205.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/manual_chunks/hand_keyed_items_206_210.py
python3 data/data/other/forms/raw_data/sil_bonda_further_2022/import_bonda_further_2022.py \
  --write --pdf tmp/pdfs/bonda_further_2022/JLSR2022_005.pdf
UV_CACHE_DIR=/tmp/uv-cache uv run --project data python -m pytest -q \
  data/data/other/forms/raw_data/sil_bonda_further_2022/test_bonda_further_2022_checkpoint.py
```

The guarded importer rejects OCR-bearing schemas, missing conceptual keys, non-NFC text, an incorrect reviewer declaration, and PDF checksum drift. It stages targets only and builds a separate after-the-fact comparison audit against the reviewed JLSR 2022-004 ledger.

The complete comparison audit and standalone 210-cell checked-Dumripada replacement audit, complete-source profile coverage, shared bibliography/dialect/profile routing, and final target installation are finished. Consolidated builds, compiled-CLDF checks, graph validation, full-suite validation, and browser QA remain deferred.
