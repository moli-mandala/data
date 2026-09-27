# Kumari et al. (2026), *A Grammar of Gaddi*, lexical and numeral appendices

The [UCL Press open-access grammar](https://doi.org/10.14324/111.9781800089938)
is licensed CC BY-NC 4.0, excluding third-party material. The pinned 152-page
PDF is cached outside this source package, and its SHA256 is asserted by
`import_source.py`. The PDF and page renders are not redistributed here.

The source scope includes the complete Appendix II Table B.1, printed pp.119–124 (PDF pp.136–141): 221 prompts / 254 individually numbered attestations, plus the entire Appendix III numeral inventory on printed pp.125–128 (PDF pp.142–145). Appendix III has 105 cardinal cells, 22 ordinals, seven fractionals and five distributives. All 139 numeral cells are accounted: 138 forms and one source dash for million. Total installation is 392 forms with all 254 prior B1 keys and rows unchanged. Source table + editorial row number gives stable `C1`–`C4` keys; no forms are reconstructed for blank cells.

Appendix III explicitly credits **Kumari Mamta's doctoral research** (p.125 footnote1); this provenance is present in every numeral audit record. It is not presented as the same fieldwork as Table B.1. Prose examples, inventories, paradigms, references and index remain outside the lexical/numeral appendix scope. No row-specific dialect or site is invented.

Selectable PDF text preserves the three columns reliably. `pypdf` supplies
each row in reading order. Six spaces or tabs before superscript aspiration,
five wrapped glyphs and one spaced combining tilde inside brackets are PDF
extraction artifacts. One detached final `t` in the printed gloss *heart* is
rejoined after a source-layout assertion. The raw
bracketed content and full source line remain in `audit.jsonl`. No OCR is used.
The source's IPA drives house transcription; the source-local profile keeps
the /ɑ/ versus /ə/ distinction, aspirated affricates, nasal vowels, and the
two printed superscript schwas. It retains a printed colon at item 132 as
uncertain rather than interpreting it as the IPA length mark.

Items 44, 88, 132, 155.3 and 207.2 retain typed uncertainty in the audit and
an `uncertain` discovery tag. Item 155.3 is the word *split* grouped with
*spit* alternatives in the printed numbering; item 207.2's form duplicates
the printed *rain* form. No correction is inferred. Source parenthetical
gender and noun/verb labels become tags; contextual definitions remain in
the gloss. The source provides no etymological claims or row-specific dialect
sites. Gaddi uses existing canonical language `ga` without a new dialect.

From the data repository, regenerate with
`.venv/bin/python data/other/forms/raw_data/gaddi_grammar_2026/import_source.py --install`.
The required PDF location is `../tmp/pdfs/gaddi-2026/grammar.pdf`. Focused
tests run with `.venv/bin/python -m pytest -q tests/test_kumari_gaddi_2026.py`.
The seeded 20-row audit and all six page images were checked against the
source; see `sample-audit-2026092501.json`.

This is source-input integration only. Full CLDF build, compiled identity,
reference, graph and concept checks, full suite, and browser inspection remain
deferred under the user's no-build instruction. The browser database is never
refreshed as part of this package.

Appendix III extraction uses `pdfplumber` coordinate ordering because ordinary reading order scrambles groups on p.126. Every appendix page was visually checked with a single sequentially overwritten small render. Superscript aspiration spacing is joined. Coordinate-sorted tilde placement is repaired only in forty-five `pɛ̃tɑli` and sixty-five `pɛ̃ʈʰ`, verified on the printed page. Raw extraction lines and repairs remain in the audit. No OCR is used.

The printed fractional quarter `pɔ:ni` retains its colon; three-quarters `trijɑ coutʰɑ` retains Latin c rather than inserting tʃ. The source's tenth `əkʰ bəʈɑ solɑ` and sixteenth `əkʰ bəʈɑ dəs` are preserved with their printed glosses, not swapped. All four receive typed uncertainty and an `uncertain` tag. Cardinal/distributive/fractional rows use `num`; ordinals use `num ord`. The source profile adds ẽ coverage and retains real phrase spaces. No borrowing or etymological link is inferred.

Independent Appendix III pass 1 found 3/20 sampled errors: the parser incorrectly moved English distributive prompt suffixes into Form. All five C.4 rows were fixed and covered by exact form/gloss regressions. Fresh pass 2 (seed 2026092662) passed 0/20, excluding the initial sample; all five C.4 fixes and source anomalies were additionally rechecked. Nine focused tests pass, including both audit hash pins, full PDF regeneration, all 254 preserved prior rows, complete profile coverage and 392/392 scoped parse without errors. Source metadata validates. The source stage is complete; full CLDF, compiled IDs/graph, generated references, full suite and browser checks remain deferred.
