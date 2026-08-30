# SIL Mavilan Tulu survey (JLSR 2025-005)

This source-local package guards the complete manual ingestion of Appendix A in
Canvin, Joseph, and Manoj (2025). The mandatory survey-table and OCR/manual-review
addenda are active.

## Source and corrected topology

The pinned 61-page PDF is `data/tmp/pdfs/JLSR2025-005.pdf` (1,232,467 bytes;
SHA-256 `d7675b86c9f083eb2389d268078325643979db4a520da776246cf8ecb5fdc629`).
On 2026-08-28, a fresh download from the attachment on SIL archive record 111012
matched those bytes exactly.

Table 10 on physical p.27 names six lists. Appendix A.2 occupies physical
pp.28–38 / printed pp.22–32. Direct visual inspection corrects the preliminary
210-prompt census: physical p.37 has items 187–207 and physical p.38 has only
item 208, Dust, followed by blank page body. The source therefore contains
**208 prompts × 6 lists = 1,248 conceptual cells**; items 209–210 do not exist.

MTP, MTV, and MTE are target Mavilan Tulu lists (624 cells). MAL, TUL, and KOD
are audit-only controls (624 cells). Table 10 prints `KOD`; physical p.38 prints
`IKOD` (a capital I, not a seventh list), preserved as alias evidence in
`list_registry.tsv`.

## Complete manual review and accounting

All 1,248 cells were inspected by eye against 400-dpi page renders, with
900/1200-dpi crops for dense glyphs. Every retained form was hand-keyed from the
rendered source under the declaration
`hand-keyed-from-rendered-source; OCR-not-copied`. PDF text, OCR, and legacy data
were locator/post-entry comparison aids only and never supplied or verified a
reading.

- 1,230 attested cells: 615 targets and 615 controls.
- 18 literal source blanks: 9 targets and 9 controls.
- 0 ambiguous and 0 illegible cells; `unresolved_readings.tsv` is header-only.
- 615 target forms are staged; 615 attested controls and all 18 blanks are
  excluded from form staging but retained in the 1,248-row audit.

Only after each block was closed was the target ledger compared against the
legacy 615-row install. All 615 target coordinates are present: 556 strings
agree exactly and 59 preserve literal source distinctions rechecked against
tight source crops. Per-block comparisons are recorded in the seven audit
files. Similarity percentages are notes, not cognacy claims, and staged
`Parameter_ID` values are deliberately blank because the source makes no
etymological assignments.

The existing source key `canvin2025`, base language `markodi`, three registered
dialect IDs, and `conversion/markodi.txt` are reused. The profile covers every
staged source form without additions; `profile_inventory.tsv` records 59 source
characters and `profile_additions.tsv` is header-only.

## Reproducibility

```bash
python3 data/data/other/forms/raw_data/sil_mavilan_tulu_2025/manual_chunks/hand_keyed_items_001_018.py
python3 data/data/other/forms/raw_data/sil_mavilan_tulu_2025/manual_chunks/hand_keyed_items_019_039.py
python3 data/data/other/forms/raw_data/sil_mavilan_tulu_2025/manual_chunks/hand_keyed_items_040_079.py
python3 data/data/other/forms/raw_data/sil_mavilan_tulu_2025/manual_chunks/hand_keyed_items_080_119.py
python3 data/data/other/forms/raw_data/sil_mavilan_tulu_2025/manual_chunks/hand_keyed_items_120_159.py
python3 data/data/other/forms/raw_data/sil_mavilan_tulu_2025/manual_chunks/hand_keyed_items_160_189.py
python3 data/data/other/forms/raw_data/sil_mavilan_tulu_2025/manual_chunks/hand_keyed_items_190_208.py
python3 data/data/other/forms/raw_data/sil_mavilan_tulu_2025/import_mavilan_tulu_2025.py --stage
python3 -m pytest -q data/data/other/forms/raw_data/sil_mavilan_tulu_2025/test_mavilan_tulu_2025.py
```

Shared registry edits, replacement of the legacy generated form file, the full
CLDF build/test suite, graph review, and browser QA are intentionally deferred
to consolidated integration.
