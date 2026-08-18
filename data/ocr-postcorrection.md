# OCR post-correction overlays

The local Jambu site exposes an OCR review workbench at `/dev/ocr`. It reads
source audits and writes small, checked-in correction overlays. Audits remain
reproducible extractor output; compiled CLDF is never edited by the workbench.

## Automatic adapters

An audit under `data/other/forms/raw_data/*-audit.csv` is discovered when it
has an `Entry_Key` column and either OCR-specific columns or OCR review states.
The workbench infers common `Form`, `POS`, `Gloss`, raw-evidence, page, and
confidence fields. The adjacent overlay name is `*-corrections.csv`.

This generic adapter supports text correction immediately. Add a source entry
to `data/ocr-postcorrection.json` when the source needs page images, unusual
field mappings, extra model-candidate files, or crop geometry.

Cached `.cache/ocr/*/output/*_entries.csv` datasets are also discovered when
they expose a case-insensitive `Entry_Key` field; their overlays are written
under `data/ocr-corrections/`. OCR datasets without an immutable source key
are deliberately not exposed until their extractor supplies one. Row order,
OCR coordinates, and normalized form text are not safe correction identities.

Source PDFs are intentionally not committed. Each configured document is
resolved in this order:

1. its source-specific environment variable;
2. `JAMBU_OCR_SOURCE_DIR/<filename>`;
3. `~/Downloads/<filename>`;
4. `.cache/ocr/sources/<filename>` in the data repository.

The image endpoint only accepts manifest-listed source/document IDs. Pages are
rendered with Poppler into a temporary cache. Per-document page offsets,
vertical-coordinate fields, and column bounds handle duplicate editions and
different page layouts.

## Durable overlay

Every corrections file has these columns:

```text
Entry_Key,Status,Form,POS,Gloss,Notes,Audit_Fingerprint,Updated_At
```

`Status` is one of `accepted`, `corrected`, `illegible`, or `skipped`.
`Audit_Fingerprint` hashes the exact source audit row. The API uses optimistic
concurrency when saving, and the shared `ocr_corrections.py` importer helper
refuses stale decisions after an audit is regenerated.

Importers must opt in explicitly: load the overlay with
`load_corrections(corrections_path, audit_path)` and emit only `accepted` or
`corrected` records. `illegible` and `skipped` remain accounted for without
entering lexical data. Thari is the reference integration.

The workbench and its write/image APIs are available only in a Vite development
build requested through `localhost`, `127.0.0.1`, or `::1`.
