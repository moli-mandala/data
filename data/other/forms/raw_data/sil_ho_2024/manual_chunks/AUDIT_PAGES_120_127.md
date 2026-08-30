# Ho pp120--127 manual chunk audit

- Scope: physical PDF pp120--127; printed pp111--118; items 145--168.
- Topology: 24 items × 27 sites = 648 explicit `Item` + `Site_Code` cells.
- Review method: every cell hand-keyed from 400-dpi rendered source images.
- OCR policy: no OCR columns, no OCR input, no scaffold lookup, and no fallback
  or generated transcription defaults. Every row declares
  `hand-keyed-from-rendered-source; OCR-not-copied`.
- Accounting: 619 attested, 28 source blanks, 1 ambiguous, 0 illegible.
- Confidence: 315 high, 332 medium, 1 low.

Unresolved reading:

- Physical PDF p127; printed p118; item 167 `where?`; site HKE; left
  column: overwritten/struck source cell. Tentative diplomatic reading
  `1 okonṯe`; status `ambiguous`, confidence `low`; independent re-review
  required.

Focused validation:

```sh
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  data/other/forms/raw_data/sil_ho_2024/manual_chunks/test_pages_120_127_hand_keyed.py
```
