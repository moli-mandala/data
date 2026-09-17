# Berger source repair, 14–15 September 2026

This package supersedes the August parser outputs. The frozen August audit and identity map remain historical evidence. See [the repair review](../../../../../source_checklists/20260914-berger-repair.md) for counts, decisions, limitations and validation.

## Reproduce

From the `data` repository:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python data/other/forms/raw_data/berger_cleanup.py --output-dir /tmp/berger-proposal
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python data/other/forms/raw_data/berger_cleanup.py --install
```

Without either output option, the importer is read-only. `make berger` uses this importer. Installation refuses unresolved retired source keys. Neither invocation builds CLDF or the browser database. The published identity crosswalk cannot be rebuilt with fuzzy matching.

`manifest.json` pins the scan and every editorial input. The importer verifies those hashes before use. A deliberate editorial change requires reviewing the change and updating its manifest hash. Do not regenerate the manifest merely to bypass a mismatch.

## Files

- `ocr-pages.jsonl.gz`: the frozen 241-page OCR cache, including the excluded duplicate for comparison. Ordinary reproduction does not need the original PDF or machine translation runtime.
- `layout.jsonl.gz`: fonts and positions aligned with the OCR. Spelling comes from OCR, not the damaged hidden text layer. `extract_layout.py` reproduces it from the pinned local PDF with pdfplumber, sequentially.
- `repair.py`: layout, grammar scope, attested full paradigms, references, relations and identity protection.
- `reviewed.json`: source-image corrections, preserving literal grammar alongside English editorial definitions. A reviewed headword does not imply review of every example or etymology in its article.
- `gold-alignments.json`: explicit source alignments correcting neighbouring-article contamination of the historical hand-entered tranche.
- `translations.json`: saved German/English pairs bound to German SHA-256. Machine translations remain unreviewed. `translate.py` takes a saved digest-to-German request JSON and a local pinned Argos model; it uses one CPU thread and checkpoints batches. Older translations are reused only by exact source hash.
- `installed-before.csv.gz`, `gold-before.csv`, `legacy-auto.csv.gz`: immutable baselines for public-key continuity and original catalog evidence. They are inputs, never overwritten during installation.
- `catalog-protection.json`: 157 changed-form candidates, the affected 149 accepted sets, and the 327 Berger keys separated from reparsed articles. Candidates are not all proven errors.
- `audit.jsonl.gz`: parsed records, exclusions, gold rows, compatibility records and added paradigm forms; actual emitted keys, full source context, typed review reasons and per-record hashes.
- `aliases.csv`: retired source keys and their surviving article or protected-evidence target. Installed again at `data/other/form_aliases/20260914-berger.csv`.
- `relations.json`: withheld relation endpoints with reasons; source claims remain in Notes.
- `crossreferences.json`: literal targets, candidate keys, exact resolutions and unresolved states.
- `summary.json`: reproducible counts and outstanding integration gates.
- `fresh-sample-before.json`, `fresh-sample-review.json`: frozen fresh sample (seed 20260915), per-entry decisions and hashes of inspected local crops. Full page images and the PDF are not redistributed.
- `validation.json`: focused test and scoped integration evidence for this repair.

## Editorial limits

Full source grammar is retained in Notes, including suffix-only paradigms. Canonical class/number/POS labels are scoped to the current form; full attested plural forms can become rows. Suffixes are never used to fabricate unattested words. Hunza, Nager and Yasin map to canonical Burushaski with source-qualified dialect tags. `NH` stays a literal source label with an explicit uncertainty note; its expansion is not independently verified. Registry coordinates are inherited quality-C approximations.

Cross-references use unique exact complete forms; uncertain homonym readings, relative references, missing targets and cycles are not guessed. Blank English definitions for unresolved index entries are intentional. German text and literal targets remain reviewable.

The 14–15 September corrections do not certify the whole OCR corpus. The fresh sample had 17/20 material errors before its recorded corrections. Legacy g/ġ loss, remaining OCR and translation mistakes, unreviewed class scope and source etymologies still require scholarly review. The frozen legacy identity map is not a linguistically validated mapping. Accepted graph sets with flagged Berger evidence now retain separate original evidence records, preventing current articles from silently inheriting their meaning.
