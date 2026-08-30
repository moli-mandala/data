# Dhurwa 2021 guarded source package — exhaustive source-local review

This package records an exhaustive, guarded manual review and source-local staging of Joseph and Joseph's *A Sociolinguistic Survey among the Dhurwa of Madhya Pradesh and Orissa* (JLSR 2021-034). It deliberately does not alter shared registries, shared staging, or generated CLDF.

## Source and current accounting

- Canonical official archive: SIL archive 89899.
- Pinned PDF: `tmp/pdfs/dhurwa_2021/JLSR2021_034.pdf`, 649,838 bytes, 24 pages, SHA-256 `92965cbf77b88685a3f46e59053ce027b4a600037c8043d518c477ac7eac341e`.
- Appendix B: physical pp. 17–21 / printed pp. 12–16.
- Exact topology: 200 prompts × 5 response columns = 1,000 cells.
- Complete review: physical pp. 17–21 / printed pp. 12–16, items 1–200 = 1,000 cells.
- Exact accounting: 995 attested cells, five explicit `--` blanks, 1,008 expanded responses, zero transcription ambiguities or illegibles.
- Known Dhurwa targets: four columns, 800 conceptual cells and 809 source-local staged forms.
- Unresolved fifth column: 200 cells, 199 expanded responses, two explicit blanks; audit-only because its printed header is empty.

Every cell was manually read from 600-dpi rendered images and visually rechecked, with difficult glyphs checked at 1200 dpi. OCR/PDF text did not seed, supply, or verify lexical readings. Every ledger row declares `hand-keyed-from-rendered-source; OCR-not-copied` and contains exact physical/printed page, item, and column coordinates.

## Reproduce the exhaustive source-local package

From the outer workspace root:

```sh
python3 data/data/other/forms/raw_data/sil_dhurwa_2021/manual_chunks/hand_keyed_items_001_041.py
python3 data/data/other/forms/raw_data/sil_dhurwa_2021/manual_chunks/hand_keyed_items_042_082.py
python3 data/data/other/forms/raw_data/sil_dhurwa_2021/manual_chunks/hand_keyed_items_083_124.py
python3 data/data/other/forms/raw_data/sil_dhurwa_2021/manual_chunks/hand_keyed_items_125_167.py
python3 data/data/other/forms/raw_data/sil_dhurwa_2021/manual_chunks/hand_keyed_items_168_200.py
python3 data/data/other/forms/raw_data/sil_dhurwa_2021/import_dhurwa_2021.py \
  --write --pdf tmp/pdfs/dhurwa_2021/JLSR2021_034.pdf
UV_CACHE_DIR=/tmp/uv-cache uv run --project data python -m pytest -q \
  data/data/other/forms/raw_data/sil_dhurwa_2021/test_dhurwa_2021_checkpoint.py \
  data/data/other/forms/raw_data/sil_dhurwa_2021/test_dhurwa_2021_items_042_082.py \
  data/data/other/forms/raw_data/sil_dhurwa_2021/test_dhurwa_2021_items_083_124.py \
  data/data/other/forms/raw_data/sil_dhurwa_2021/test_dhurwa_2021_items_125_167.py \
  data/data/other/forms/raw_data/sil_dhurwa_2021/test_dhurwa_2021_items_168_200.py
```

The importer rejects any ledger with an OCR-bearing schema, incomplete keys, non-NFC text, a missing declaration, or a mismatched PDF checksum. It writes only the exhaustive source-local `checkpoint_forms.csv`, `checkpoint_audit.tsv`, and manifest hashes.

## Editorial decisions

- `--` is recorded as a source-explicit blank, never as a lexical form.
- A printed slash is retained in the ledger and expands deterministically for staging; ordinary spaces remain within one source response.
- The source colon is interpreted as phonetic length only in the proposed source-local conversion profile; the diplomatic form remains unchanged in the ledger and `Phonemic` field.
- The four named columns map provisionally to Duruwa `[pci]` dialects. The fifth column remains identity-unresolved and excluded.
- The complete-source profile covers every staged form. Shared bibliography/language/dialect/profile routing, consolidated build, full tests, graph validation, and browser QA remain deferred to integration.
