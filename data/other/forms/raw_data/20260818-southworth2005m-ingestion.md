# Southworth 2005, pages 9–10 — ingestion review

## Source and scope

- Source: Franklin C. Southworth, “Prehistoric Implications of the Dravidian Element in the NIA
  Lexicon with Special Reference to Marathi,” author-hosted preprint,
  `https://ccat.sas.upenn.edu/~fsouth/DravidianElement.pdf`.
- Exact input: 14-page PDF, SHA-256
  `14242247d0bec684febbb34b2a44c8530d010497150ebec11800a5a02a236260`.
- Included: PDF/printed pages 9–10, Table 1 and Table 2A–B.
- The source states no licence. Only extracted lexical and comparative facts are installed; the
  PDF and page renders are not redistributed in the data repository.
- Applicable checklist addenda: dictionary/glossary, etymological/comparative, and OCR-heavy PDF.

## Extraction and accounting

- The PDF text layer loses the relevant diacritics as unmapped CID glyphs. The checked extraction
  therefore renders pages 9–10 with Ghostscript at 400 dpi and reproduces raw English Tesseract
  5.5.2 OCR (`--psm 6`). Every installed interpretation was then checked against the page image.
- Table 1 has 25 printed lexical rows. Five separately printed Old Marathi attestations expand
  these to 30 installed form rows: 25 Marathi (`M`) and 5 Old Marathi (`OM`).
- Table 2 has 23 comparative records. It produces 25 structured comparison blocks: item 11 has no
  Turner target, item 17 maps conservatively to two targets, and item 18 maps to three targets.
  Seven Table 2A records also have a checked Table 1 DEDR counterpart and therefore produce seven
  source-attributed cross-family comparison rows.
- The per-record audit accounts for all 48 source records (25 Table 1 + 23 Table 2). The seeded
  final sample contains 20 records and has 0 material errors.
- All 30 emitted keys are unique and stable. All 30 forms survive the compiled CLDF and have a
  rank-1 `borrowed` edge to the printed DEDR target.

## Editorial and transcription decisions

- Source `@` is preserved as the printed variable adjective ending.
- The source’s `ə` notation, defined as `[əː ~ eː]`, is represented as `ə̄` in the display layer.
- The source says final `i` and `u` are long in a monosyllable or before one consonant/zero;
  reviewed applications include `āī`, `māṇḍī`, `nīṭ`, `bāḷant(īṇ)`, Old Marathi `mecū`, and
  Old Marathi `ḍoī`.
- `ph` is normalized to `pʰ`. Source spellings remain available in `Original`; normalization is
  confined to `conversion/southworth-marathi.txt`.
- Forms ending in `-ṇe` are tagged `verb`; variable-`@` forms are tagged `adj`; all rows are tagged
  `loanword`. `phaḷ` additionally receives `uncertain` because Southworth marks its origin
  controversial.
- Southworth’s page 12 notation notes were consulted for `@`, `ə`, and vowel length. No source
  language, dialect, location, or coordinate was added: Marathi and Old Marathi already map to
  registered language IDs `M` and `OM`.

## Comparative-table policy, exclusions, and unresolved cases

- Table 2’s `+`, `-`, `?`, and blank cells are distribution evidence, not printed language forms.
  They are preserved verbatim in the audit and comparison prose; no unattested reflexes were
  fabricated from column marks.
- Cross-family rows are limited to items 1, 2, 6, 7, 8, 9, and 10, where the checked tables print
  both a Table 1 DEDR source and a Table 2 Turner/CDIAL target for the same item. They are modelled
  as Southworth-attributed Dravidian-to-Indo-Aryan loan claims, not independent confirmation.
  Item 2 (`phaḷ` / CDIAL 9051) is low-confidence because Southworth prints `(?)`; the other six
  are high-confidence descriptions of the paper’s explicit classification. No DEDR target is
  inferred for the remaining Table 2 rows, and item 11 has no Turner target.
- Headers, running prose, notation notes, and column labels are not lexical records. Table 2 item
  11 is retained on the page 9 Marathi `dāṭ` form because the source prints no Turner number.
- Item 20 prints Turner 5634 for `taḍāga` ‘pool’. CDIAL 5634 is `*taḍapphaḍ` ‘agitate’, while the
  unique exact form-and-gloss match is CDIAL 5635. The block is attached to 5635 with an explicit
  editorial correction; printed 5634 remains visible in the audit and app.
- Item 17 prints Turner 3276–8 for `*kutt(ir/ūr)a` ‘dog’. CDIAL 3276 is the homonymous ‘rent,
  lease’ entry; the exact in-range dog forms are 3277 and 3278, while related 3275 lies outside the
  printed range. Only 3277/3278 are linked. The conflict with 3276 is deliberately unresolved and
  remains visible in the audit.
- There are no unlinked Table 1 forms, source-defined variants, derivations, or alternate graph
  parents in this two-page scope. Those checklist gates are not applicable.

## Validation

- Importer output: 30 forms, 48 audited records, 25 comparison blocks, 20 sampled records; the
  comparison extractor adds seven audited cross-family claims (six high-confidence, one low).
- Focused tests: `23 passed` across the source, cross-family, sound-profile, and dialect suites.
- `make all`: passed all seven CLDF stages. The source contributes no row to `errors.txt`, no
  replacement character, and no unregistered language, dialect, or reference.
- Compiled assertions: 30 stable source keys, 30 durable forms, 30 borrowed DEDR edges, 25 Table 2
  blocks, seven cross-family comparison rows with resolvable endpoints and citations, and the
  corrected 5635 attachment.
- Checklist generator: 92 ingestion units; the Southworth unit and installed-row audit are fresh.
- Full repository suite: `448 passed, 10 skipped`.
- Browser database: isolated rebuild produced 469,787 lemmas, 391 references, and 604 total
  cross-family comparisons in a 79,495,168-byte SQLite file. `PRAGMA integrity_check` returned
  `ok`, `foreign_key_check` returned no rows, and the size guard passed. `npm run check` reported
  0 errors and 6 pre-existing warnings.
- Browser QA had no console warnings/errors. Representative pages:
  - `/entries/d1494`: the paper’s high-confidence `*kāẓ-` → CDIAL 3083 loan comparison appears
    with Southworth’s exact cross-table locator and evidence, alongside the independent DEDR claim.
  - `/entries/d4004`: `phaḷ` / CDIAL 9051 is visibly low-confidence and retains the controversial
    origin wording.
  - `/entries/2639`: reverse-endpoint rendering says the Indo-Aryan entry was possibly borrowed
    from DEDR 1109 and links back to the Proto-Dravidian entry.
  - `/entries/f_cxg3ioej4emr2`: Marathi `pʰaḷ`, uncertain borrowing from DEDR 4004, with combined
    page 9 and page 10 citation and Table 2 distribution.
  - `/entries/f_u4mq6fmjsatzq`: Old Marathi `mecū`, borrowed from DEDR 4722, with page 9 locator.
  - `/entries/d4004`: Proto-Dravidian `*paẓ-V-`, listing `pʰaḷ` among its reflexes.
  - `/entries/5635`: `taḍāga` ‘pool’, preserving and explaining the printed 5634 conflict.
  - `/entries/f_cdqe2wzuzvmme`: Marathi `dāṭ`, retaining item 11’s form-only comparison evidence.
  - `/references/southworth2005m`: partial-scope, provenance, editor, and OCR metadata.

No commit, push, release, deployment, or homepage changelog change was requested or performed.
