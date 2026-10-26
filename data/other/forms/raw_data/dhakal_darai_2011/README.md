# Dhakal 2011: The Darai Language

The complete 486-page thesis is represented at the **source stage**. All chapters,
paradigms, quoted morphemes, interlinear examples, five narrative texts and
front-matter lexical mentions are accounted for. Questionnaire/respondent pages,
references, abstract symbols, other-language comparisons and source omissions have
explicit audit dispositions. Full compiled integration and browser QA remain deferred.

## Installed output

- `../../20260921-dhakal-darai.csv`: 8,361 author-transcription entries.
- `../../20260927-dhakal-darai-literal.csv`: 12 entries preserving literal notation,
  with conversion disabled and blank Phonemic where interpretation is not supported.
- `whole-installed-audit.jsonl`: all 11,304 expanded source units: 8,373 emitted
  entries, 2,431 citation-preserving occurrence reuses and 500 explicit controls.
- All 485 earlier installed keys remain distinct. Their Original, Phonemic and
  pre-existing graph fields are unchanged. Durable identity state was not modified.

The audit preserves exact source keys, printed-page locators, raw evidence pointers,
source claims, individual occurrence rows, output routes and reuse targets. Reuse
requires the same complete analysis, Notes, sound route, attribution and uncertainty.
Eighteen separately recovered Chapter 3 expressions align with the same previously
installed physical attestations; their full translations remain available in Notes.

## Source, evidence and transcription

Dubi Nanda Dhakal, *The Darai Language*, Tribhuvan University PhD thesis, 2011.
University catalogue: https://elibrary.tucl.edu.np/items/4c2a3db9-8d61-4dc4-8906-60db09180a1e
The 486-page PDF SHA-256 is
`6ca12ee82a22a16b0394b08e319afd5a94f075111fb8a994ee10d59da234bebf`.
No open redistribution licence is asserted; the original book is not committed.
The package records attributed linguistic facts and exact review evidence.

Native PDF font decoding, rather than OCR, supplies the source text; original-page
visual review resolves glyphs and layout. Source Original is preserved. The explicit
profile uses author transcription for display and retains existing Phonemic values;
structural removals are recorded per occurrence, with NFC/NFD coverage tested.
Thirty-seven printed Appendix 3 stem-boundary hyphens are preserved.

Four compressed notation groups remain literal because alternate scope is not
recoverable. Eleven literal entries retain uncertain marks or notation; one further
entry preserves Roman spelling without a claimed phonemic value. Across both routes,
117 entries have an uncertainty tag with source-specific explanations. Conflicting
printed analyses remain visible; only unsupported grammatical claims are withheld.
Two unresolved multiword interlinear groups retain typed alignment uncertainty.

Source grammar is structured from exact labels, heading scopes and reviewed claims.
Conditional particles do not acquire verb POS. Agreement indexing is distinguished
from case inflection; complete derived words are distinguished from derivational
markers. All 100 residual uppercase/person-number gloss cases were explicitly
classified. Calendar abbreviations remain dates, and text IDs remain provenance.

## Language, graph and bibliography

All entries use canonical Darai (`dara1250`, Bihari). Existing Chitwan and Pidrahani
registrations are retained; explicit Tanahun and Nawalparasi source districts are
registered separately, with no invented coordinates. Historical witnesses and
individual narrative speakers remain attributed provenance, not inferred dialects.

Twelve variant links and eighteen derivation links are supported by explicit source
claims or preserved earlier accepted evidence. Co-listed synonyms receive no invented
variant edges. All source-local targets resolve. Two earlier numerical etymology
fields remain unchanged; no new cross-source etymologies are invented.
The `dhakal2011darai` bibliography block records complete scope and provenance;
all other bibliography blocks were preserved.

## Reproduction and validation

From the data repository, `make ingest SOURCE=20260921-dhakal-darai` invokes
`import_source.py --install`. Without `--install`, the importer only regenerates
source-local proposals and verifies the independent approval hashes. Immutable
`legacy-before-whole-recovery.csv` and its profile snapshot prevent feedback from
the installed output. Missing or altered reviewed inputs fail clearly.

Independent original-page sample 206 passed **0/20 material errors**, after the
retained failed audits led to systematic repairs. The dependency-only baseline
patch reproduced both CSVs, audit and profile byte-for-byte. **66 focused tests
passed**, covering actual canonical parsing, metadata, formatted bibliography,
NFC/NFD profiles, exact literals, grammar roles, citations, graph targets and legacy
preservation. One compiled-source survival check was deliberately not run.

See `whole-installation-report-20260927.json`, the independent audit, and
`../../../../../source_checklists/20260927-dhakal-darai-whole.md` for the handoff.
Full CLDF/database build, compiled identity/graph/reference verification, full test
suite and browser QA remain deferred under the user's instructions. No database,
commit, push or release was performed. Historical Chapter 3 acquisition and review
remain in `CHAPTER3_HISTORICAL_NOTES.md` and their original artifacts.
