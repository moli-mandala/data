# Berger audit evidence — 14 September 2026

This is an audit package, not an installation or a replacement for the existing
20260828 source artifacts. See [the report](../../20260914-berger-audit.md).

- `summary.json`: counts, unchanged source-file hashes, scope and results.
- `catalog-key-review.csv`: 157 normalized-form differences at keys referenced
  by the cognate catalog. These are review candidates, not all confirmed mistakes.
- `confirmed-cognate-errors.json`: four image-verified identity mismatches and
  their actual compiled parent relationships.
- `missing-relations.csv`: all 78 relations to excluded source entries, with
  current compiled IDs and any other accepted parents.
- `sample.csv` / `sample.json`: seed 20260914, 20 distinct primary articles,
  installed data, source audit evidence, image hashes and manual findings.
- `targeted.json`: four identity examples and three grammatical examples checked
  against the pinned PDF. These seven are additional to the random sample.

The existing pinned PDF and `.cache/ocr/berger/pages` are required. No OCR service,
translation model or network access is needed. Diagnostic scripts write only to
`../tmp/berger-audit-20260914` from the data repository:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python source_checklists/audits/20260914-berger/audit_berger.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python source_checklists/audits/20260914-berger/compiled_check.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python source_checklists/audits/20260914-berger/render_samples.py
```

For the targeted images, copy this package's `targeted.json` into that temporary
directory, then run `render_targeted.py` by the same route. Image rendering requires
the already-used pypdfium2 and Pillow packages. Sample verdicts are human-readable
review records and are not inferred by the scripts. Rendering hashes can depend
on library versions.

Do not use the normal `berger_cleanup.py` CLI for a read-only audit: even without
`--install`, it rewrites the main source audit, sample and manifest. These diagnostic
scripts instead call the existing read/parse functions in memory and check that
the installed inputs and source artifacts retain their initial hashes.
