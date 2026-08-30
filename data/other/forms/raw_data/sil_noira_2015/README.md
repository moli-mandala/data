# Noira 2015 source-local ingestion package

This is the complete guarded source package for SIL ESR 2015-012, *Noira Bhils
and a Few Other Groups: A Sociolinguistic Study*. Shared registries and
generated CLDF outputs are deliberately untouched.

## Contents and accounting

- Canonical PDF: `tmp/pdfs/noira_2015/silesr2015_012.pdf`, 1,676,716 bytes,
  96 pages, SHA-256
  `cb93db089a21e55e878f436632d8282c64c98fca85afe18179f8f3383db35280`.
- Appendix A3: physical pp. 33–78 / printed pp. 27–72.
- Topology: 210 prompts × 17 lists = 3,570 conceptual cells.
- Manual review: 3,526 attested cells, 44 source-explicit blanks, 4,385
  expanded responses, zero ambiguous/illegible/unresolved cells.
- New targets: 11 lists, 2,310 conceptual cells, 2,271 attested cells, 39
  blanks, and 2,714 staged forms with 2,714 unique immutable entry keys.
- Exclusions retained in `exhaustive_audit.tsv`: 630 cells / 834 responses
  republished from the Dhule team's ESR 2013-004 lists, plus 630 cells / 837
  responses from Gujarati, Marati, and Hindi controls.

Every cell was visually inspected and hand-keyed from 400-dpi rendered pages;
selected difficult glyphs were checked at 800 or 900 dpi. OCR/PDF text was not
used to seed, supply, or verify any lexical reading. Each ledger row has the
exact declaration `hand-keyed-from-rendered-source; OCR-not-copied`, and the
guard refuses OCR-bearing schemas.

## Reproduce source-local artifacts

From the outer workspace root:

```sh
for range in 001_027 028_054 055_081 082_108 109_135 136_162 163_189 190_210; do
  python3 data/data/other/forms/raw_data/sil_noira_2015/manual_chunks/hand_keyed_items_${range}.py
done
python3 data/data/other/forms/raw_data/sil_noira_2015/import_noira_2015.py \
  --all --write --pdf tmp/pdfs/noira_2015/silesr2015_012.pdf
UV_CACHE_DIR=/tmp/uv-cache uv run --project data python -m pytest -q \
  data/data/other/forms/raw_data/sil_noira_2015/manual_chunks/test_items_*_hand_keyed.py
```

The importer writes only source-local `staged_forms.csv`,
`exhaustive_audit.tsv`, `dhule_republication_reconciliation.tsv`, and manifest
checksums. `staged_forms.csv` uses the raw-form column order documented in the
importer and intentionally has no header, matching Jambu raw-source files.

## Editorial conventions

- Multiple responses remain in printed order and expand deterministically.
- Printed cognate/similarity numbers are parallel audit evidence, not part of
  a phonetic form.
- Printed `0 no entry` is an explicit `source_blank`, never an inferred blank.
- IPA and unusual source typography are diplomatic and NFC-normalized;
  apparent compounds, spaces, hyphens, aspiration spellings, and diacritics
  are not silently regularized.
- Astambha, Mundalwad/Mutalwad, and Toranmal use the earlier Dhule elicitation
  and are excluded here. `dhule_republication_reconciliation.tsv` accounts for
  all 630 source cells against ESR 2013-004. Its 3 literal-ledger-exact and 627
  representation-different labels reflect different storage of printed
  similarity labels in the two frozen packages, not lexical disagreements;
  report-identified source/list identity determines the exclusion.
- The two Kotli lists are provisionally routed as distinct survey-site
  dialects under canonical Noiri. The source labels `Kotli` and `Adivasi
  Bhil-Taradi` are preserved, dialect Glottocode/coordinates remain blank, and
  neither list is equated with historical Kotali/Khandesi. This is a
  source-supported integration route, not a genealogical determination.
- `conversion_profile.tsv` covers every grapheme in all 2,714 staged forms.
- `unresolved_readings.tsv` is header-only because visual review resolved every
  cell.

## Frozen pre-integration contract

`preintegration_audit.py` independently rechecks the PDF, all frozen manual
ledgers/generators, staging/audit crosswalk, immutable keys and citations,
classification, profile coverage, and republication exclusions. Its checked
artifacts are:

- `preintegration_manifest.json`: exact integration-ready counts, identities,
  hashes, exclusions, and deferred shared gates;
- `render_hashes.tsv`: 46 fresh 144-dpi topology/renderability audit images for
  physical pp. 33–78 (these fresh images did not supply or verify readings);
- `profile_inventory.tsv`: greedy profile-token coverage of every staged form.

The frozen manual ledger/generator bundle SHA-256 is
`7532ec17e39058a0d49920b4462af242744b7d2ab8aeb92739e70d5f2d8ac566`;
the staged CSV SHA-256 is
`c82983a319d6d6fbf5c07063f0655ae3e4e8e3890d625e1bfc2a38f95c811746`;
the fresh render-tree SHA-256 is
`4a89c9c3125d79cebd96c40ff5d225230de37c33600bc817fbc41c2f17ee9d10`.

Run the source-local freeze gate with:

```sh
python3 data/data/other/forms/raw_data/sil_noira_2015/preintegration_audit.py
UV_CACHE_DIR=/tmp/uv-cache uv run --project data python -m pytest -q \
  data/data/other/forms/raw_data/sil_noira_2015/test_preintegration_contract.py \
  data/data/other/forms/raw_data/sil_noira_2015/manual_chunks/test_items_*_hand_keyed.py
```

Shared source-specific integration is complete: the 2,714 target rows, exact
profile, bibliography/reference record, Dungra Bhili parent, eleven dialect
rows, and source-key profile route are installed. Exact scope and hashes are in
`shared_integration_manifest.json`. The consolidated build, global source-audit
regeneration, full tests/graph validation, browser refresh/QA, and commit remain
explicitly deferred.
