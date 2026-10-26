# Bailey 1920 Rohru — whole-source inputs installed

Thomas Grahame Bailey, *Linguistic Studies from the Himalayas* (Royal Asiatic Society, 1920), public domain. The original scan is retained at `tmp/pdfs/bailey1920/bailey1920.pdf`, SHA-256 `7a1577a2d2a24eca8b95e9b1ed700ef01f716d1069e4468f36a673611d093b39`. Printed page +26 gives the one-based PDF page. Source-attributed Rohru maps to existing `roh` (Rohrui); no new dialect or invented coordinates are introduced.

The installed source covers Rohru-attributed material in the shared introduction pp.113–118 and the whole main section pp.118–130: grammar, all22 translated responses, lexical notes and all255 glossary cells. **671 audited units yield723 rows, preserving all309 historical glossary rows and adding414.** Seventeen rows carry explicit transcription uncertainty. Three English cross-reference-only glossary cells remain excluded. Other-lect and unattributed shared examples remain separately attributed audit evidence.

The historical glossary snapshots and `generate_glossary()` preserve the earlier309-row stage. Whole-source regeneration uses immutable first readings plus `whole-second-reading-20260927.json`; `import_source.py` requires equality with the independently audited proposal before installation. Stable entry keys, canonical filename, citation key and identity settings are preserved. The two formerly held glossary cells now contribute three legible candidates with explicit glyph-identity uncertainty, not asserted phonological interpretations.

The profile preserves historical notation, with lowercase display and house w→v; exact source spelling remains Original. NFC and NFD inputs were checked. Printed paradigms, suffixes, full sentences and source qualifications remain distinct. No new cognacy or borrowing is inferred. The audit accounts for source-local variants and excludes controls explicitly. `whole-source-text-inventory.jsonl` retains the locator OCR; original images govern corrections.

The first independent whole-output sample found a missing macron in two related phrases. A complete nasal-mark class review repaired it; a fresh disjoint20-unit sample passed0/20 material errors. Historical glossary audits are retained separately. Canonical parser checks preserve all723 ordered keys, Originals, Notes, citations and tags with zero conversion errors; all12 focused historical/whole-source tests pass. See `whole-installation-result-20260927.json` and `root-whole-output-audit-20260927-pass2.json`.

Reproduce from `data/`:

```sh
.venv/bin/python data/other/forms/raw_data/bailey_rohru_1920/import_source.py --install --check-pdf
.venv/bin/python -m pytest -q tests/test_bailey_rohru_1920.py tests/test_bailey_rohru_whole_recovery.py
```

Full CLDF build, compiled graph/durable-ID checks, reference regeneration, full-suite and browser database/app QA remain deferred. This is source-stage completion, not a claim that those gates passed. Existing Zoller/Patyal attestations remain untouched; later LSI quotations are derivative comparators.

## Screened alternative: Doda Siraji

Bailey 1908, Part IV, printed p. 43 has 36 direct-gloss left-column items headed Siraji; its preface explicitly calls this chapter “Siraji (Doda Siraji)” and printed p. 36 locates it near Doda. All 36 gloss prompts in that column occur in the already installed `LSI-DODASIRAJI` vocabulary, with conspicuously matching source readings such as `ikk` one, `satt` seven, `nau` nine, `das` ten, `zanan` woman/wife, and `mattho` child. Jambu's earlier LSI audit manually mapped `DODASIRAJI` to canonical `dod` Dodi (registry Glottocode `kash1277`), while that audit, its registered LSI dialect, and current Glottolog identify the source lect with `sira1264` “Siraji of Doda”; canonical `sir` Sarazi already carries that code. This is a genuine historical classification/registry conflict, not grounds for silently remapping forms by name. The substantially overlapping Bailey page was **not** installed or registered as another source. It remains a review lead if a later linguistic registry reconciliation is authorized.
