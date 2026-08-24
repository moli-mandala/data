# Emeneau 1997 Brahui ingestion review

## Scope and rights

- Source: Murray B. Emeneau, “Brahui etymologies and phonetic developments: new items,”
  *Bulletin of the School of Oriental and African Studies* 60(3), 1997, pp. 440–447.
- Stable record: <https://www.jstor.org/stable/619537>; DOI:
  <https://doi.org/10.1017/S0041977X00032481>.
- Input: nine-page publisher PDF, SHA-256
  `e2aa0c7a0063b83509cf402cb880de97a902b1195c7d5f69bb75d8775fb30dde`.
- Rights: copyright SOAS 1997. The publisher PDF is not redistributed; only extracted
  linguistic facts, page-level evidence, and audit metadata are installed.
- Included: all lexical, etymological, dictionary-correction, and phonological claims on
  printed pp. 440–447. The introduction, repeated supporting examples, cross-page repeats,
  and the reference list do not become duplicate lexical rows but remain accounted for in
  the audit.

## Extraction and reconciliation

Each rendered article page was independently passed to `gpt-5.6-luna` under the checked-in
reflex-extraction contract. The eight article pages yielded 76 page records
(`0, 16, 11, 17, 16, 9, 7, 0` for printed pp. 440–447). Page JSON is raw evidence, not the
editorial output.

Image-checked reconciliation records 18 corrections. These include restoring `bāšt`,
`taṛifing`, and `cīkap-`; separating target IDs from adjacent reconstructions; removing
spurious previous-target claims; deduplicating repeated claims; correcting the horse gloss;
and downgrading tentative comparisons from direct links to ranked hypotheses. The seeded
20-record final audit has 0 material errors and no known systematic residual error class.

## Installed result

- 76 audited source records -> 19 installed lexical rows -> 19 compiled source attestations.
- 35 ordered entry-text blocks preserve the article's comparison and phonological prose.
- Languages: 18 Brahui rows and one Gadaba row, all using existing canonical languages; no
  new dialect or coordinate claim is made.
- Direct/reflex analyses are installed for `bēɣ-`, `bēl`, `hōɣ-`, `mux`, `taf-`, `bāšt`,
  `šurūfing`, `kirrefing`, slaughter `taṛifing`, `tāring`, and `allāī`.
- Four source-supported rank-1 overlays update existing data: Brahui `(h)ullī`/`ullī` ->
  DEDR 701, Brahui `dū` as borrowed from CDIAL 6586, and Gadaba `cīkap-` as borrowed from
  Telugu `cīk-`.
- Five forms retain only ranked hypotheses: `pužža`, `kūžing`, `pisfing`, `šupping`, and
  `dūī`. Sour-milk `taṛifing` remains unlinked. These six cases are first-class nodes rather
  than forced etymologies.
- Same-lect homonyms are protected: sour-milk and slaughter `taṛifing` remain separate, and
  tongue `dūī` does not merge with the independently sourced “control” sense.

## Transcription

`conversion/emeneau-brahui.txt` is the explicit source route. Emeneau's underlined `gh` is
preserved in `Original` and mapped to display `ɣ`; vowel length and Dravidianist diacritics are
retained. Compiled output contains no replacement glyphs or source-specific conversion errors.

## Validation status

- The importer is deterministic and checksum-aware.
- The focused importer, profile, dialect, and checklist suite passed: **24 passed**.
- `make all` passed all seven data stages. The final build contains 417 references and applies
  all 11 source-driven graph assignments.
- Browser database transformation and staging passed. The compact database contains 470,910
  lemmas, 417 references, and 25,523 typed edges; `PRAGMA integrity_check` returned `ok`, and
  `PRAGMA foreign_key_check` returned no rows. `npm run check` reported 0 errors and six existing
  warnings.
- Browser inspection succeeded for `/entries/f_6voa4fsbvujpc` (`bēɣ-`) and
  `/references/emeneau1997brahui`: the former renders Emeneau's locator, DEDR 5078 link, prose,
  and `gh` -> `ɣ` alignment; the latter renders the bibliography/provenance, 19 source rows,
  the 18/1 Brahui–Gadaba distribution, and six unetymologised forms. A later attempt to repeat
  the remaining local routes was blocked by the browser security layer and was not bypassed.
- The complete repository suite currently reports **477 passed, 10 skipped, 2 failed**. Both
  failures are pre-existing global count assertions unrelated to this source:
  `test_marked_origins_are_borrowings_with_valid_targets` expects 597 rather than 595 marked
  origins, and `test_backstrom_poc_cloth_forms_link_to_pota_not_avajjharati` expects 19 rather
  than 14 Backstrom forms. The strict full-suite and remaining browser-inspection gates are
  therefore deferred; this review does not label the repository-wide checklist complete.

## Representative review routes

- `/entries/f_6voa4fsbvujpc` — accepted `bēɣ-` reflex with sound alignment.
- `/entries/f_5uv343fuclkso` — `kūžing` with two ranked hypotheses.
- `/entries/f_rpyanync5ohwc` — existing Brahui `dū` borrowing overlay.
- `/entries/d701` — `(h)ullī`/`ullī` reassignment and superseded DEDR 500 edge.
- `/entries/f_idmynejbzcznu` — Gadaba `cīkap-` borrowed from Telugu.
- `/references/emeneau1997brahui` — source metadata, coverage, provenance, and forms.

No commit, push, release, or deployment was requested or performed.
