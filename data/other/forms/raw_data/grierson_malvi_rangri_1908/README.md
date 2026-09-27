# Grierson 1908 Mālvī (Rāngrī): complete source stage

George A. Grierson, *Linguistic Survey of India*, IX, part II, *Specimens of the
Rājasthānī and Gujarātī* (Calcutta, 1908). The original is public domain.
The original Commons DjVu and alternate Internet Archive PDF are retained with
provenance/hashes. Original DSAL grayscale images supply the definitive readings:
binary scans lost dots and short horizontal strokes. Manual transcription and
multiple complete readings were used; no OCR contributes to installed forms.
The secondary Nordic HTML is comparison evidence, never substituted for originals.

## Complete included scope

- Shared grammar, printed pp. 54–59: 280 physical units. The explicit p. 54 standing
  rule includes unqualified Rangri/Malvi examples; explicitly Malvi-only and
  other-language comparisons remain excluded controls.
- Rangri standard-list column, pp.305–321: all 241 prompts, yielding 317 expanded
  forms before reuse. Prompts172,174,201 are blank. Other lect columns are controls.
- Roman specimens I and II, pp. 249–251 and 254–256: 823 editorial aligned units from
  827 physical fragments after four explicit line-continuation joins. A pair with
  transposed interlinear English is emitted as one attested phrase, with both raw
  units and locators retained, giving 822 expressions before exact reuse.
- Native witnesses, pp.248,252–253: all 65 lines and 823 atom-level alignments.
  Two p. 55 written alternatives, बापे/बापए, are retained unpaired: the source does
  not assign either spelling to a specific Roman alternative.
- Free English translation on pp.256–257 is context, not another set of lexical
  attestations. Boundary/control pages and source-wide commentary are inventoried
  in `whole-source-scope-20260926.json` and the contextual review reports.

The installed CSV contains **1,170 rows** from 1,393 rows before 223 exact-analysis
reuses. `audit.jsonl` contains **1,346 units**: 1,024 ingested, 223 reused exact
attestations, one grouped-expression continuation, 78 excluded controls, 15 isolated
bound-morphology units, three blanks and two unpaired native alternatives. The
65 native lines remain in `native-witness-reviewed.jsonl`; individual alignments
are embedded in the main audit. Count categories are not all physical tokens:
grammatical function expansion and editorial line joins are explicitly recorded.

## Representation and decisions

Raw `Form` preserves original Roman notation, including vowel length, nasal marks,
underdots, superscript a, meaningful hyphens and distinct v/w spellings. The
registered profile lowercases capitals and converts conventional ṅ to ŋ; its
explicit w/W policy exception preserves the original v/w distinction. The parser
retains raw spelling as `Original`. `Phonemic` stays blank because no independent
IPA layer is supplied. Native script has its own field and secure alignments.
One place-name native glyph uncertainty remains typed in audit and Notes with an
`uncertain` tag; no target Roman glyph remains unresolved.

Source grammar is structured where explicit. Three stated functions of each
simple-present cell have separate function rows but share a physical unit.
Historical passive and cross-language descriptions remain source-attributed;
Gujarati/Marwari/Bundeli/Awadhi parallels do not create unsupported donor or
etymological links. Isolated grammatical affixes remain audit-only. Other-language
Carey/Khas/Kanauji comparisons are excluded and preserve their cited-authority
context; the unspecified Carey edition remains a typed bibliographic hold.
`grammar-source-commentary-review-20260926.json` records source-wide claims;
`grammar-independent-commentary-reconciliation-20260926.json` verifies all 25
independently reviewed commentary classes, including 85 emitted Notes occurrences.

Exact repeats merge only when language, literal form, gloss, Native, Phonemic,
residual Notes, complete tags and graph fields agree. Every citation and source
occurrence key remains in audit mappings. Homonymous analyses and divergent native
spellings remain separate. All five previously installed Entry_Keys survive
transcription corrections. `legacy-pilot/` preserves the earlier importer, CSV,
audit, metadata, test and readings; its old 16 glyph holds were superseded by full
recovery. Durable `data/form-identities.csv` was not modified by installation.

The base remains `Malw`, Glottocode `malv1243`, despite its legacy display name
“Malwai”; the existing language-qualified Rangri dialect tag is retained. Both
specimen headers say **State Dewas, Junior Branch**. Coordinates remain blank
because no exact elicitation site is supplied. Narrative Court Udaipur is retained
as source commentary, not recast as the collection locality.

## Reproduction and validation

From the data repository:

```sh
.venv/bin/python data/other/forms/raw_data/grierson_malvi_rangri_1908/import_source.py
.venv/bin/python data/other/forms/raw_data/grierson_malvi_rangri_1908/import_source.py --check-scan --install
.venv/bin/python -m pytest tests/test_grierson_malvi_rangri_full_stage.py tests/test_grierson_malvi_rangri_1908.py -q
```

The importer verifies the independent passing audit and exact frozen hashes,
regenerates rows/audit, then copies only approved source-stage outputs. Missing
required reviewed inputs fail clearly. `sample_full.py` produces a reproducible
four-stratum sample with frozen-hash verification and explicit prior-sample/key
exclusions; its CLI documents the seed/exclusion arguments.

Independent native review passed 20/20. Whole-output pass1 found a citation
integration defect: internal semicolons in native locators were interpreted as
citation separators. The source-local exporter now uses “and” within these
locators, preserving original punctuation in the audit. Exactly 36 citation fields
changed; all other fields and audit remained byte-identical. Fresh disjoint pass2
(seed 20260926142) passed 20/20. Seven staging tests plus three installed-source tests
pass, including both actual reference parsers, full profile coverage and scoped
parsing of all 1,170 rows with exact Native/Original/Phonemic preservation. Source
BibTeX formats in memory; registered routing, tags and source-local key targets
resolve. `postinstall-verification-20260926.json` records final checks and hashes.

This is **complete at the source stage**, not a completed full-pipeline ingestion.
Full CLDF build, compiled durable ID/alias/graph/reference verification and full
suite remain deferred under the user's instructions. Browser database refresh
and app inspection were not requested; no representative app entries are claimed.
No remote jobs, publication or database build occurred.
