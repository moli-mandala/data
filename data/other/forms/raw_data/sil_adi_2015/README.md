# SIL Adi 2015 manual ingestion

`data/SOURCE_INGESTION_CHECKLIST.md` is active. The applicable source type is
the survey-wordlist/comparative-table addendum, with guarded rendered-cell
review for phonetic tables.

The pinned source and complete 307 x 9 topology are documented in
`DISCOVERY.md`. Fifteen authoritative chunks cover all items 1--307 on
physical PDF pp.17--38 / printed pp.13--34: 2,763 cells, 2,670 attested, 93
explicit `no entry` cells, 0 ambiguous, and 0 illegible. Separately labelled
responses produce 2,770 staged form rows.

The pre-integration audit freezes all evidence before shared registry work.
The canonical PDF SHA-256 is
`8e1500383a02445252a3eb6973a1b011fabea71eb25ad79fc43ba5b78bd1135c`;
the fifteen-chunk manual bundle SHA-256 is
`a9a1aac22c77c4cf66230c2fa014a7b151cd676db44810744b06748070bd92f0`;
the staged forms SHA-256 is
`edb29a8f65fea0600e3d54bfcf2adef81fd833c47b619de5cd701bd61df4031c`;
and the exhaustive audit SHA-256 is
`6fb69a145419fff42c6b48d8e965acf2dbd9dc06bd297edf2e19f62e4f88877b`.
A reproducible 400-dpi render set covers all 22 lexical pages (physical
pp.17--38); `render_hashes.tsv` freezes tree SHA-256
`0746c68daf48349570eb0d37e2d69afb79c22571d69a410484b8437e1efd794c`.

Every cell was keyed while viewing the rendered page and visually matched
before acceptance. The born-digital text layer was only a character-input
scaffold; OCR supplied no accepted reading. The ledger carries exact physical
and printed page, item, list code, and column coordinates.

The source-local review is complete. `import_adi_2015.py --stage` validates the
pinned checksum, every ledger declaration/method stamp, NFC, response/label
cardinality, unique cell and entry keys, and the nine-list routing registry. It
writes a lossless 2,770-row `staged_forms.csv`, an exhaustive 2,763-row
`staged_audit.tsv`, an empty-header-only `unresolved_readings.tsv`, and a
complete `symbol_inventory.tsv`.

The staged rows map to four canonical parents: 1,249 Bokar-Ramo, 613
Mising-Padam-Miri-Minyong, 610 Bori-Karko, and 298 Milang. All nine elicitation
sites have immutable, language-qualified dialect tags and exact item/page/list
citations. The 42-symbol inventory contains no replacement character. The
dedicated source profile preserves diplomatic IPA in `Original` and normalizes
only display `Form`; it explicitly retains spaces, commas, length,
nasalization, and two source-printed question marks.

Shared source-specific integration installs the frozen file byte-for-byte as
`data/other/forms/20260829-sil-adi.csv`, registers the four parents and nine
sites, adds the canonical reference, and routes only `padung-sako2015adi`
through `conversion/sil-adi.txt`. `shared_integration_manifest.json` records
the exact contract. The consolidated build, global audit, browser refresh and
commit remain deferred.

Run the frozen source gates from `data/` with:

```sh
python3 data/other/forms/raw_data/sil_adi_2015/preintegration_audit.py
uv run --with pytest --with segments pytest -q tests/test_sil_adi_2015.py \
  data/other/forms/raw_data/sil_adi_2015/manual_chunks/test_*_hand_keyed.py \
  tests/test_sound_profiles.py
```
