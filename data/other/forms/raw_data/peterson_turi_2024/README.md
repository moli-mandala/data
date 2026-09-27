# Peterson et al. (2024): Odisha Turi

Source-local installation, 2026-09-21. Consolidated CLDF build, full-suite and
browser gates remain open; this is not a completed ingestion.

Canonical source: John Peterson, Abhay Sagar Minz, Prabhat Linda, Ariba Khan,
Francis Xavier Kachhap and Manish Gari, *A Brief Introduction to the Turi
Language of Eastern India*, Bhasha 3(2), 261–302.
DOI: https://doi.org/10.30687/bhasha/2785-5953/2024/02/005. CC BY 4.0, confirmed
on the DOI landing page and the PDF. The landing page currently links the
un-suffixed PDF; a separately indexed suffixed PDF has pagination 263–304.
`manifest.json` records both hashes and the edition decision. Search-engine
text is not transcription evidence. No OCR was used.

## Coverage and evidence

Appendix 1, printed pp.291–297 / PDF pp.31–37: 275 records, including extra
52.1 and the second printed 234 (why, in the expected slot 254). 224 attested
responses expand to 246 forms; 51 explicit unelicited cells remain audit-only.
106 source responses carry an IA annotation. No ancestry, donor, or variant
edges are inferred. Grammar examples and Appendix 2 texts are outside this
wordlist ingest. There are no control-language columns or etymon IDs.

Turi had 23 rows in the current compiled forms at selection time; this source
addresses sparse Munda coverage. It reuses canonical `Turi` / `turi1246`, adding
the named Odisha variety. No village coordinates are given for this wordlist.
The workshop in Ranchi is elicitation provenance, not the dialect's location.
The Odisha dialect therefore has no invented coordinates or separate glottocode.

`records.json` retains text-layer strings, source baselines and every extracted
glyph's text/font/position. `page-*.txt` preserve ordinary page extraction as an
alternative representation. The reproducible extractor uses glyph baselines:
bounding-box crops had incorrectly pulled accents from the next row and a
subscript from the preceding row. The corrected extraction is tested explicitly.
`visual-review.json` records a 0/20 seeded visual audit and additional edge cases.

## Transcription and interpretation

Read phonology pp.266–267 before conversion. Appendix IPA distinguishes vowel
quality but has no phonemic vowel-length contrast. The profile maps ɑ → a,
ʊ → u, preserves ɛ/ɔ/ə, maps IPA j → y, and uses house consonants and ʰ for
aspiration/breathy release. Palatal stops c/ɟ and affricates tʃ/dʒ have the
conventional house c/j displays; the source IPA distinction survives in
Original and Phonemic. Glottalization and voiceless r̥ are preserved.

Item 58 has redundant nasal marking in the PDF text layer. Original and
Phonemic retain it; only display collapses the repeated mark. The row remains
`uncertain` with a typed transcription reason in the audit. This is the one
unresolved source transcription detail, not a silently repaired source reading.

IA annotations are preserved as the **complete source response** in Etymology,
so the scope of labels in mixed alternative lists is not reassigned to an
individual alternative. The source explicitly says these labels indicate likely
borrowing/similarity and do not establish ultimate Indo-Aryan origin. No donor
endpoint is identifiable. Item 206's explicit “likely IA” is also tagged
uncertain with a typed borrowing reason. Compounds are retained; the paper's
exclusion of compounds and likely loans was for its COG analysis, not a denial
of lexical attestation. No internal component/derivation edges are guessed.

The English jar/plain subscript distinctions are retained as numbered glosses.
`you (pl.)` receives `pl`; male/female **referents** remain gloss information,
not grammatical gender tags (the paper says Turi has no grammatical gender).
Repeated prompts remain distinct by source key. Keys use printed page/item,
with stable numbered children for alternatives; the duplicated printed 234
therefore does not collide.

## Reproduce

From the data repository, download the URL in `manifest.json` to a local PDF.
The extractor fails if its SHA-256 differs; the snapshot is usable offline.

```sh
.venv/bin/python data/other/forms/raw_data/peterson_turi_2024/extract.py PATH_TO_PDF --output /tmp/turi-records.json
.venv/bin/python data/other/forms/raw_data/peterson_turi_2024/import_source.py --output-dir /tmp/turi-proposal
make ingest SOURCE=20260921-peterson-turi
.venv/bin/python -m pytest -q tests/test_peterson_turi.py tests/test_dialects.py -k 'not compiled'
```

Installed rows: `data/other/forms/20260921-peterson-turi.csv` with sibling YAML.
Complete per-record audit: `20260921-peterson-turi-audit.json` in this directory.

## Validation and remaining gates

- 12 focused importer/profile/layer/dialect tests passed; one compiled test
  explicitly deselected until the consolidated build, and separately checked
  against the current stale CLDF to establish that integration is outstanding.
- `source_meta.py`: passes (195 files, 192 citation keys at this check).
- Source-specific profile-policy audit: no violations. Global profile-policy
  check is not green: existing `sil-pahari-pothwari` rule `čʰˑ` is missing.
- Full `make all`, full test suite, generated reference formatting, stable-ID
  survival/deduplication, compiled graph/concept checks, and app/browser QA are
  **deferred**, not passed. No compiled files or durable identity registry were
  regenerated. `test_compiled_source_survival_and_no_edges` is the source's
  explicit integration assertion and must run after the full build.
- Configured `.github/workflows/python-app.yml` only runs on push/PR and only
  tests; it is not an available dispatcher for the uncommitted full pipeline.
  No new remote runner, publication, or heavy local build was authorized merely
  to satisfy this gate. Workspace resource instructions require deferral here.
- Representative future app checks: Turi language, Odisha dialect, the new
  reference; item 1 head, item 52.1 flour, item 58 flavour, item 117 money's three
  alternatives, and item 206 son. Persistent entry URLs are unavailable until
  compilation; these are QA targets, not claims of browser verification.
