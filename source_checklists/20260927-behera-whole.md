# Behera (2022) whole-report source-stage reconciliation

The mandatory ingestion checklist and its survey-wordlist, PDF, and comparative
source addenda were applied. This is an installed source stage, not a completed
compiled ingestion. The user's standing instruction defers full CLDF and browser
database builds, the full test suite, remote work, and publication.

## Source, extent, and method

- Gangadhar Behera, *A Sociolinguistic Profile of Korwa and Kodaku Tribes in
  Chhattisgarh and Jharkhand, India*, SIL JLSR 2022-014. Pinned 115-page PDF SHA-256
  `a8efbe88405e27024a7a6ec786cd6fde3e382f0eaf0d0081197d3880ed97eb0c`.
  Its original remains outside the installed data; only extracted facts are
  installed under the report's restrictive copying terms.
- All 25 Appendix B.5 lists (physical PDF pp. 66–90), all four Appendix C texts
  (pp. 93–108), and the report's outside lexical witnesses were inventoried.
  Native PDF text and original-page images were compared; OCR only aided the
  earlier independent wordlist comparison. The physical audit is
  `data/other/forms/raw_data/sil_korwa_kodaku_2022/whole-source-audit.jsonl`.
- The historical wordlist audit and manifest remain intact as a record of the
  original 4,458-row target-only import. The guarded whole-report importer has an
  exact-hash approval, transaction journal, retained baselines and receipt under
  `data/other/forms/raw_data/sil_korwa_kodaku_2022/whole-recovery/`.

## Counts and decisions

- Installed source rows: **6,864**, with **6,864 unique entry keys**. The legacy
  CSV now has 4,459 rows, preserving all 4,458 prior keys and adding one printed
  alternative. The additions CSV has 2,405 rows: Asuri 224, Sadri 1,359, Birjia
  209, Mundari 223, Korwa 300, Kodaku 90.
- The additions comprise 1,370 comparison responses, 1,033 text rows, and two
  qualified introductory word-origin examples. The text rows include 117 whole
  expressions/questions/answers; 790 continuous spans and 192 C.5 columns were
  reviewed, with 66 internal spans held where word gloss alignment is unresolved.
- The witness audit has 6,823 records: 6,135 historical wordlist decisions, 638
  text events, 48 outside witnesses and two undefined site-code records. Every
  installed key has a physical-source witness. A separate 332 held candidates
  comprise 246 Tanmai language-identity conflicts, 66 internal-alignment spans
  and 20 outside language-attribution holds. Another 26 outside witnesses are
  documentary; blanks and two undefined lowercase site codes remain audited.
- Birjia was added as a distinct base language and seven source localities as
  dialects. No coordinates were invented. Similarity groups do not imply
  cognacy, and no graph edge was inferred from them. Two introductory source
  claims remain attributed commentary, not ancestry edges.
- The legacy profile and original `Phonemic` values remain; additions keep the
  literal source transcription with conversion disabled and hyphens preserved.
  New `Native` and `Phonemic` values were not invented. 90 continuous-span
  grammatical metadata decisions and 26 C.5 column decisions were reviewed;
  ambiguous category/alignment cases retain typed uncertainty.

## Validation and remaining gates

- Independent final audit: fresh disjoint **20/20** original-page checks, zero
  material errors. Owner/independent reviews checked all 25 wordlist pages,
  all continuous partitions and source C.5 column layout. The installed eight
  canonical asset hashes and durable identity hash match the guarded receipt.
- Focused installed/recovery/historical tests: **19 passed**. Source metadata
  validation: **291 settings files, 284 citation keys**. The installed source
  parser preserves all rows and keys; all dialect tags have registered parents.
  A repeated installer invocation verified approved bytes without rewriting them.
- The full `make all` CLDF pipeline, compiled reference/identity/graph checks,
  full test suite, browser database refresh and representative app-page QA are
  **deferred by instruction**. Thus no compiled node count, generated edge
  count, or app-entry claim is made. No commit, push or release was performed.

Representative raw source rows for later app QA:

- Birjia `hoɖɔ` ‘body’ — `behera2022korwakodaku:comparison:item001:siteJ:response1`.
- Sadri `moɪ` ‘I’ — `behera2022korwakodaku:C.5:column:1:2`.
- Korwa `hʊʔ ʌkʌrɪja` ‘bite!, he bit’ —
  `silkorwakodaku2022:i183:K:f2:v2`, with grammatical reading deliberately
  unresolved at the alternative level.
