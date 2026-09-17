# Sheth Sanskrit counterpart review — 14 September 2026

The dictionary, website-snapshot and comparative-source ingestion checklist is active.
This follows the user's request to parse Sheth's Sanskrit etyma after the quoted-work
tag integration. The scholarly distinction matters: DDSA calls the bracketed items
**Sanskrit equivalents**. A lexical equivalent is not by itself a demonstrated
historical ancestor. Frontmatter physical PDF page 2 explicitly defines `[दे]` as
देश्य-शब्द, so that marker must never become a Sanskrit headword.

## Installed scope

- All 41,638 frozen articles receive an etymology audit; 10,137 excluded articles
  remain excluded under the existing structural review.
- 26,191 Sanskrit counterpart records, one per accepted source sense, retaining
  source-local homographs. Explicit alternate Prakrit heads share their sense's
  counterpart.
- 27,484 source-attributed `related` comparisons with `undetermined` historical
  direction. High confidence describes the explicit printed correspondence, not
  a historical derivation.
- 6,866 installed rows have no bracketed etymology; 6,583 contain only the deśya
  label. These produce no counterpart.
- 140 compound analyses, 1,032 other unresolved expressions (including abbreviated
  alternants and uncertain readings), and 13 transcription failures remain audited.
- No inferred ancestry, borrowing or automatic CDIAL merges. No explicit historical
  direction was established for this batch of bare equivalents. Those scholarly
  decisions remain separate from extracting the printed forms.

The original 42,118 Prakrit/Apabhramsha/Ashokan records, their keys, source tags,
variants and prose are unchanged. Sanskrit uses the existing `Sk` language; no
language, dialect, clade or coordinate is added. Native contains the printed
Devanagari. Roman forms use the existing explicit Devanagari transliterator and
Sheth's ē/ō convention; all output symbols are covered by the existing preservation
profile. Phonemic is blank. The original native spelling is retained even where
unusual (e.g. द्बन्द्व); no silent spelling correction is attempted. Sanskrit Gloss
is deliberately blank: the Prakrit definition is labeled as context in Etymology,
not asserted to be an independently printed Sanskrit definition.

## Reproduction and evidence

From the data repository:

```sh
.venv/bin/python data/other/forms/raw_data/sheth_etyma.py \
  --output tmp/sheth-sanskrit --seed 20260915
# After reviewing the proposal, rerun with --install.
```

Inputs are the pinned `sheth_2026/audit.jsonl.gz` article snapshot. Outputs are
`data/other/forms/20260914-sheth-sanskrit.csv`,
`data/other/comparisons/20260914-sheth-sanskrit.csv`, and the package's
`etymology-{audit.jsonl.gz,report.json,sample.json}`. The compiler resolves both
comparison endpoints from rich source keys and rejects missing keys. Persistent-ID
assignment rewrites these endpoints along with the rest of the graph. The new
form file is appended after existing inputs to protect legacy positions.

The 20-article sample (seed 20260915) was reviewed against raw markup: **0 material
extraction errors**. Deliberate regressions cover deśya labels, compounds, uncertain
expressions, vowel length/virāma, numbered senses, alternate heads, missing graph
keys and full-corpus sound-profile coverage. **60 focused tests pass.**
The evidence manifest is
[audits/20260914-sheth-sanskrit-review.json](audits/20260914-sheth-sanskrit-review.json).

This is fact extraction from the existing DDSA transcription, with no new OCR or
new licence claim. Primary Sheth bibliography now lists the Sanskrit input and
comparison scope. Remaining auxiliary bibliography and missing printed reference
pages are unchanged from the earlier review. The full suite remains deferred under
the local resource policy; the two previously reported broader manual-survey
count/overlay failures remain unresolved. This does not claim complete ingestion
of the printed dictionary or complete historical etymologisation.

## Compiled and local browser validation

The full seven-stage data build and browser transform completed sequentially.
All 42,118 previous Sheth IDs were preserved. All 26,191 new Sanskrit records and
27,484 comparison links survived compilation with correct language endpoints;
the ancestry graph was unchanged. Post-build focused checks passed (46 tests).
SQLite quick_check and the browser decoder SHA256 round trip passed. The resulting
SQLite contains 791,794 nodes and 28,933 comparisons including the 1,449 pre-existing
comparisons. The local artifact is 73,146,701 bytes; maximum-compression production
size gates remain unmet and publication was not requested. Frontend check passes
with 0 errors and 6 existing warnings.

The local app loaded the updated database and showed both directions of the
*abhijjhā* / *abhidhyā* comparison, the precise native evidence, and the DDSA page 63,
article 23 citation. The source page shows all 26,191 Sanskrit counterparts and
68,306 total browser forms (three prior identical-record collapses). The comparison
panel was renamed from “Cross-family comparisons” to “Source comparisons” because
this batch stays within Indo-Aryan.

Representative local entries:
- http://127.0.0.1:5173/entries/f_lfykdfuys43jq — Prakrit abhijjhā.
- http://127.0.0.1:5173/entries/f_7iqg325pt3ig6 — Sanskrit abhidhyā.

Evidence: [build audit](audits/20260914-sheth-sanskrit-build.json).
The existing exclusions, auxiliary bibliography gaps, uncertain expressions and
full-suite limitation above remain open; this is not complete historical
etymologisation or a production release.
