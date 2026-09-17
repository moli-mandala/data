# Sheth DDSA integration — 14 September 2026

This installs the structurally resolved subset of the pinned 11 September 2026
DDSA transcription of Hargovind Das T. Sheth's *Paia-sadda-mahannavo* (1923–1928).
It is **not a complete ingestion of the dictionary**. The earlier snapshot and
parser reports are retained as historical records.

All 41,638 articles from 952 digital pages are preserved in `audit.jsonl.gz` with
raw markup, exact page/article identity, candidate senses, source references,
accepted rows or typed exclusion reasons. The compressed audit also acts as the
reproducible article snapshot; the earlier acquisition manifest pins the original
HTML page hashes. The original printed scans and web pagination are not equated.

31,501 articles produce 42,118 installed sense/alternate rows. 10,137 articles
remain audit-only because their structure, transcription, parentheticals or
reference scope needs review. The selector deliberately also holds out some
valid but structurally complex entries; these are not declared illegible or lost.
Unexpanded material remains in the raw article and candidate audit.

- `report.json`: reconciled counts, overlapping exclusion reasons and symbol inventory.
- `integration-manifest.json`: input/audit/profile hashes.
- `sample.json`: fresh raw-versus-output sample, seed 20260916.
- [Full review](../../../../../source_checklists/20260914-sheth-review.md): gates,
  metadata, transcription decisions, tests, unresolved catalogue and build work.

From the data repository, reproduce a proposal from this frozen article snapshot:

```sh
.venv/bin/python data/other/forms/raw_data/sheth_integrate.py \
  --output tmp/sheth-review --seed 20260916
```

Add `--install` only to install its generated rows/audit in the shared inputs.
The command processes articles sequentially. A newly acquired original page cache
can be supplied with `--cache tmp/sheth-ddsa-20260911`; all page hashes and counts
are checked before parsing. `sheth.py` acquires that cache, and `sheth_parse.py`
retains the broader development proposal with unresolved classes visible.

The `sheth-ddsa` sound profile preserves DDSA's romanization, including ē/ō, nasal
notation and bound-form °. Devanagari stays in Native. No IPA analysis is invented.
Printed Sanskrit counterparts stay in Etymology; no ancestry or borrowing links
are inferred. Printed alternate heads have source-local variant links. See
references remain scoped source notes until their targets can be reviewed.

The dictionary/glossary, website snapshot and source-comparison addenda apply.
No new OCR, new base languages, new dialects, new clades, publication or browser
refresh occurs. Existing Shauraseni, Magadhi and Paishachi dialect tags are reused.

Quoted-source follow-up: 30,147 installed rows now carry 219 distinct work tags
from a 195-entry verified frontmatter catalogue (`sheth_sources.tsv`). Full names
are registered in frontend labels. Exact locators, commentary/variant-reading
modifiers and unknown codes remain in each article's `source_tag_audit`; references
outside sense scope are retained separately. Missing printed catalogue pages 10–11
and edition-level bibliography remain unresolved. The tag-only audit is
`source_checklists/audits/20260914-sheth-source-tags.json` (20 reviewed, 0 errors).
All lexical columns and keys are unchanged; 45 focused tests pass. This updates
source inputs only, pending the separately requested full database rebuild.

Sanskrit counterpart extraction now lives in `../sheth_etyma.py`. Run it from the
data repository with `--output tmp/sheth-sanskrit` and optional `--install`.
It consumes this frozen article audit and emits a separate rich Sanskrit input
file plus source-keyed comparisons. `etymology-audit.jsonl.gz` accounts for all
41,638 articles, including excluded records; `etymology-report.json` reconciles
outcomes and `etymology-sample.json` is the reviewed 20-article sample.
26,191 Sanskrit records support 27,484 explicit lexical correspondences. These
are `related` comparisons with undetermined historical direction, not automatic
ancestry. Bare दे is a deśya label. Compound/alternate/uncertain expressions stay
in the audit. Sanskrit native spelling is retained; Roman transliteration uses
the existing explicit Devanagari converter and Sheth's long ē/ō display convention.
Prakrit meanings are labeled context, not copied into Sanskrit Gloss.
