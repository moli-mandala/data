# Approved Shinaic donor supplement — 2026-09-10

The source-ingestion checklist is active: dictionary and comparative-source addenda.
Selected scope is 406 curated donor etyma for 416 approved Sauji, Ushojo and Palula
analyses. These are parameter heads, following the Kalkoti donor supplements; no new
Shinaic attestation corpus is imported. All other source contents are excluded.

## Evidence and editorial decisions

- 97 OPED dictionary heads: pinned 2025-10-30 XML archive, DOI 10.5281/zenodo.17487678.
  Exact entry locators, native script, XML evidence and reviewed transcription are retained
  in `data/other/params/raw_data/20260910-shinaic-donors-audit.json`.
- 307 Liljegren donor comparisons: Palula dictionary v1.2, DOI 10.5281/zenodo.5526477,
  exact LX entry IDs and printed Origin statements. These remain source-attributed
  comparisons; installation does not claim independently verified donor senses or routes.
- Two Gawarbati comparisons from Knobloch's Sauji thesis: exact pages retained in the audit.
  DiVA public text, lexical facts only; no PDF redistribution or new OCR.
- The OPED tandoor cross-reference 14872 resolves to lexical entry 14945; both cited.
  Grandson 34798 is distinguished from tweezers 34799, potato 6953 from plum 6951.
- Approved semantic, dialect and morphological qualifications remain in assignment Notes
  and the audit. Native spelling and alternatives remain in the audit because the established
  curated parameter format has five columns. Comma-separated alternatives use the first
  reviewed form as the head; the complete string remains in Original.

All displays follow the explicit preservation profile: NFC only, no guessed phonemic
conversion or reconstructed sound changes. Deterministic emitter checks exact Unicode and
byte-for-byte output. No uncertain OCR output is installed. Canonical H (302), Psht (99),
Kho (3) and Gaw (2) are reused; no new language, dialect, coordinate or tag classification.
Exact language/form/source/gloss deduplication yields 406 heads for 416 uses; unrelated
homonyms are not merged. Source-local keys are resolved by the actual identity pipeline.

Twenty evenly spaced donor records were checked against retained source evidence (0/20
material extraction errors); selected rows are recorded in the adjacent sample JSON.
All selected heads also undergo programmatic provenance, Unicode and compiled-node checks.
The source comparisons' residual scholarly uncertainty is explicitly retained, not counted
as an extraction error. No unresolved extraction record is silently dropped.

Bibliography inclusion/provenance fields for all three existing source keys are extended;
existing metadata is preserved. This supplement installs selected lexical facts, not a new
full-source snapshot. Existing source licensing applies; no complete protected work is added.
No new source-wide README inventory is needed for this curated supplement.

## Validation

Validation results are recorded in `20260910-shinaic-donors-validation.json` after the run.
The isolated build is `/tmp/shinaic-donor-build/data`. The first build reached persistent
ID assignment, then exhausted disk space in concepts.py; recovery resumes that stage.
Shared compiled CLDF and browser DB are not replaced. Browser refresh/QA, release, push
and deployment are inapplicable to the requested editorial-input stage and remain deferred.
This supplement must not be described as a globally clean ingestion if the repository-wide
suite or any full-build stage remains failing.

Final result: all compilation stages completed after recovery, including regenerated final
concept links, alignments and references. All 406 donor identities and 435 final assignment
rows are saved; all 741 overnight proposals are saved as 957 links across 899 records.
All old registry bytes and overlay rows were independently verified as preserved.
Exact intended graph edges/statuses and zero-change replay verified. Four focused tests pass.
The full isolated suite completed: 1,633 passed, 67 failed, eight errors, 17 skipped, four
warnings (433.35 seconds); none of the donor-specific tests failed. The separate manual-survey
check reproduces the two previously documented failures, with two other tests passing.
This is not a globally clean repository validation; the complete failure list is in the
validation JSON. Unreviewed and held lemmata remain excluded. No browser refresh was performed.
