# Retrospective source-ingestion validation

Validated 2026-08-18 against the canonical checklist and the current sibling `jambu-static`
application. This record applies to every unit in `source_checklists/manifest.json`; each
source-specific checklist links here for the repository-wide gates.

Data pipeline: PASS for the canonical build and source-scoped gates; two unrelated repository
suite exceptions are recorded below.

- `make all` completed all seven stages: CLDF generation, reference linking, graph unification,
  durable form-ID assignment, concept assignment, alignment, and reference generation.
- The final canonical `make all` completed successfully after the test run restored fixture-mutated
  CLDF outputs.
- The final full suite run reached **432 passed, 10 skipped, 1 failed**. The sole remaining failure
  is the pre-existing Bote `kan` cross-family loan classification. It does not involve the
  Southworth source; its source-focused regression suite is **16 passed**.
- The compiled corpus has 500,088 durable form IDs, 3,116 concepts, and 1,543,712 aligned
  segments across 241,544 reflexes.
- All 92 installed ingestion units have non-empty forms, registered language/dialect identifiers,
  resolved bibliography keys, explicit sound-profile routes, audits, and focused regression
  coverage.
- `source_checklists/installed-record-audit.csv.gz` accounts for all **520,969** installed input
  rows with deterministic per-row SHA-256 digests.

Browser database and QA: PASS

- `npm run db:transform` completed with 470,190 compact lemmas and 391 references.
- The resulting database is 78,430,208 bytes, below the enforced 80,000,000-byte size guard.
- SQLite `PRAGMA integrity_check` returned `ok`; `PRAGMA foreign_key_check` returned no rows.
- `npm run check` completed with **0 errors** and 6 pre-existing Svelte warnings.
- `scripts/test_build_static_db.py` passed all 3 tests, including the independently sourced
  identical-form merge and its retained dialects, citations, aliases, and same-lect homonym.
- All 92 source units occur in the compact browser DB; there are no source units without browser
  rows.
- The staged database was served from a freshly restarted dev process. Final browser-console
  inspection found no errors.

## Representative browser QA

- Ordinary entry: `/entries/5131` renders *jambú*, its gloss, source, and derived terms.
- Borrowing chain: `/entries/n2571` resolves its durable alias and shows Proto-Nuristani
  `*iamarā` borrowed from Indo-Aryan `yamarāja`, including ancestry and alignment.
- Variant: `/entries/d4993-34` resolves to `muṛ̆uku`, marked as a variant of `muṛ̆ku` and linked
  through Proto-Dravidian `*muẓ-u-nk`.
- Alternate etymology: `/entries/10049` shows its accepted Proto-Indo-Iranian parent and the
  separate “also proposed” Indo-Aryan analysis, plus 64 reflexes across 46 languages.
- Unlinked OCR entry: `/entries/f_zgcmreutdcjxa` renders Ghatage `ādne dēsɨ` ‘to order’ with the
  OCR warning, verb tag, `Marati of Kasargod` dialect tag, and `ocr-review` tag.
- Ghatage source: `/references/ghatage-kasargod1970` identifies optical character recognition,
  renders 1,247 compact forms, supports word and `ocr-review` tag filtering, and exposes the
  exact `p. 137, entry 37` locator for the representative entry.
- Affected language/dialect: `/languages/M` renders `Marati of Kasargod` with 1,247 forms,
  quality B, and its Kasargod district description. A dialect map point is intentionally not
  asserted because the source does not supply a defensible elicitation coordinate.
- Concept membership: `/concepts/2398` reports MANGO with 202 forms and 98 unetymologised
  attestations, including Ghatage `ambo`.
- Toda: `/references/bhaskararao-toda2025?word=abak` renders the corrected dictionary source and
  the representative `abak` row after client filtering.
- Pinned legacy snapshot: `/references/patyal2` renders 66 forms from eight languages; its first
  page contains 50 linked, cited rows.
- Badaga source: `/references/hockings-pilotraichoor1992` renders the OCR provenance and all
  16,706 source rows, including 3,572 unetymologised rows.
- Reviewed Badaga entry: `/entries/f_hmjkffhyzp44y` renders *agaṭu madilu*, its noun analysis,
  DEDR 4692 link, and exact `p. 5, col. 2` citation.
- Review-pending Badaga entry: `/entries/f_id7i2lzuvr7ec` renders *Edekādu*, preserves source
  *Edeka:du*, carries the uncertainty marker, and links the printed DEDR 448/1438 claims.
- Badaga language: `/languages/Badaga` renders the existing S. Dravidian I registry record and
  does not invent a source dialect from the dictionary's pan-Badaga coverage.
- OCR workbench: `/dev/ocr` reports 9,973 pending and 20 reviewed articles, displays exact source
  crops and calibration fields, and distinguishes accepted from corrected decisions.
- Southworth Marathi: `/entries/f_cxg3ioej4emr2` renders `pʰaḷ` as an uncertain borrowing from
  Proto-Dravidian `*paẓ-V-`, with both the page 9 lexical locator and page 10 distribution locator.
- Southworth Old Marathi: `/entries/f_u4mq6fmjsatzq` renders `mecū`, its page 9 locator, and its
  borrowing chain to DEDR 4722.
- Southworth parent: `/entries/d4004` includes Marathi `pʰaḷ` among 98 reflexes in 24 languages.
- Corrected comparison: `/entries/5635` displays `taḍāga` ‘pool’, the printed Turner 5634, and the
  typed form-and-gloss correction; `/entries/f_cdqe2wzuzvmme` retains item 11 on Marathi `dāṭ`.
- Southworth source: `/references/southworth2005m` displays the partial page/table coverage,
  author-hosted URL, provenance, editor, and OCR extraction metadata.

## Remediation and residual review

- Removed and audited 14 blank Rajasthani rows and 3 blank Tharu2 rows.
- Kept two replacement-glyph-only Toda heads audit-only; 7,558 readable dictionary records emit
  8,859 installed rows after variant expansion.
- Added a reproducible, SHA-256-pinned snapshot route with stable keys and audits for nine early
  hand-curated sources; repaired the malformed Patyal snapshot record.
- Ghatage is accepted as OCR-derived rather than claimed as perfect transcription: 129 records
  are source-image verified, 1,115 remain explicitly `ocr-review`, one corrupt alternate is
  audit-only, and the deterministic 20-record final sample passes. The bibliography and every
  displayed form are marked as OCR-derived.
- Hockings and Pilot-Raichoor is installed as 9,993 structurally accounted articles expanding to
  16,706 rows. Twenty scan-backed decisions are reviewed, 9,973 remain explicitly queued for
  transcription review, and 93 printed DEDR citations remain visible but conservatively unlinked.
- Southworth pp. 9–10 account for 25 Table 1 records and 23 Table 2 records, emitting 30 Marathi
  and Old Marathi forms plus 25 comparison blocks. All forms use printed DEDR borrowing targets;
  the Turner 5634/5635 correction and 3276–3278 range conflict remain explicit and auditable.
- No release, deployment, commit, or push was requested or performed.
