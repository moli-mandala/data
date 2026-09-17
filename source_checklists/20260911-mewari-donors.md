# Mewari-reference donor supplement

Status: authorized donor heads and overlay saved; full build and tests stopped at the user’s request.

## Scope and evidence

- 22 selected donor dictionary heads audited: 13 new curated parameters, 9 reuses. New heads map to existing Hindi-Urdu (12) and Indo-Aryan/Sanskrit (1); no new languages, dialects, or coordinates.
- 24 saved analyses, 94 assignment rows, 88 source records. The saptāha group has nine borrowed links to existing CDIAL 13161. 557 Mewari-reference records remain unlinked. Eighteen non-loan proposals from batch 007 still await review.
- Platts 1884 (public domain), Monier-Williams 1899 (public domain, Hyderabad XHTML derived from the 2012-10-25 XML; updated 2014-05-19), and the official CSTT Fundamental Glossary of Agriculture (English–Hindi–Dogri), accessed 2026-09-11. CSTT contributes only the Hindi cabbage cell; no open licence is asserted. Source URLs, exact headwords and short lexical evidence are preserved in the per-head audit. Whole dictionaries, Dogri control vocabulary and unselected senses are excluded.
- Platts waznī/wazanī was checked visually on electronic PDF page 2404 (reflowed PDF pagination, not a printed-page citation). No OCR contributed. The parameter audit is the reproducible selected-fact snapshot; no live scrape is required to regenerate output.

## Checklist gates

1–3. Source scope, selected lexical facts, exclusions and reproducible importer are recorded. Dictionary/glossary and etymological-source addenda apply; website addendum applies to selected online facts. The importer audits 22 heads and emits 13 five-column parameter rows, following existing Pashto donor-head practice.

4. Existing canonical languages only. No regional donor attestation or dialect/locality is invented. Coordinate and dialect-creation gates are inapplicable.

5–7. These are curated dictionary heads, not a new attestation corpus. Source Original and Native remain separately available in the audit. No distinct phonemic layer is claimed. Parameter display preserves reviewed Unicode in NFC; nāḵẖun → nāxun, candrá-mas → candrámas, and बंदगोभी → bandgobʰī are explicit audited conversions. Historical Arabic transcription distinctions in s̤abūt and ṣāḥib are retained. Source-symbol coverage/NFC checks cover all heads; no replacement characters are introduced. Source grammatical detail is not inserted into lexical glosses.

8. Bibliography provenance updated for Platts and new Monier-Williams/CSTT keys. Formatted reference verification is part of the isolated build.

9. Borrowings use immediate Hindi-Urdu or Sanskrit donor heads; compounds retain ordered component links. Hindi-Urdu transmission is an editorial working analysis, not a claim that Platts documents borrowing events in individual survey lects. Complete weight adjectives wazanī and wazn-dār have separate heads. Whole s̤abūt is kept distinct from the proof noun; body badan from the face homonym; pestle dasta from turban cloth. jībān/jīvan and panagobī/pantāgobī remain unresolved.

10. All 22 selected head facts were reviewed against source evidence, with deliberate homonym and transcription checks; the audit retains each decision. This is exhaustive review of a small selected supplement, not a random sample of an entire dictionary. No bulk parser or unidentified residual parsing class.

11. Four donor tests pass, covering deterministic emission, source/sense distinctions, all new stable IDs, reorder/gloss correction stability, and accepted overlay scope. Focused source/profile/dialect run: 19 passed, 2 failed, 1 skipped. The two failures concern existing dialect coordinates/quality; no dialect metadata changed. The skipped test requires the completed isolated build.

12. Temporary graph application: 94 edges plus 88 status updates; repeat application zero. Existing rows preserved. Production identity policy appended exactly 13 registry rows and preserved all prior bytes. The build reached alignment after identity assignment and concepts; it was terminated at the user’s request. Full-build and compiled donor checks are deferred. The first build exhausted disk during graph rewriting; its partial generated outputs were discarded by restarting the complete pipeline after removing stale copied legacy/alignment files and unused temporary model directories. Initial full pytest collection is blocked by two existing raw-source test files sharing module name test_preintegration_contract (Bhumij/Noira); the importlib-mode run was also terminated at the user’s request; no complete full-suite result is claimed.

13. Browser refresh not requested: database construction and app QA are inapplicable under the checklist's user-triggered policy. Shared compiled CLDF remains untouched. Representative IDs are recorded in the review manifest for later app inspection.

14. README, importer, parameter CSV, per-head audit, identity validation, overlay manifest, review and tests saved. No commit, push or deployment.

## Validation logs

- /tmp/mewari-donor-focused.log
- /tmp/mewari-donor-build.log
- /tmp/mewari-donor-full-tests.log

User instruction: “dont build!” Build and full-suite processes terminated (exit 143). No further builds or browser refreshes without an explicit request.
