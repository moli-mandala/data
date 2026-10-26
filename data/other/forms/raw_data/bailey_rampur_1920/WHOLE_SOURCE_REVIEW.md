# Paired Rampuri and Baghi whole-source recovery

Owner: rohru_finish. The ingestion checklist is active with glossary, comparative-table and OCR addenda. Canonical glossary rows and all keys are frozen: 288 Rampur and 296 Baghi. Whole-source installation, database builds and publication have not occurred.

The original pinned Bailey1920 scan has 310 PDF pages. Printed page n is one-based PDF n+26. `whole-source-text-inventory-20260927.jsonl` records embedded text for shared introduction113–118, paired chapter131–147 and boundary pages130/148. Its OCR is only a locator: pages132–133 are essentially absent from the text layer.

Original headings establish this scope:

- Shared phonetic/geographic context113–115; specifically Rampuri116 and Baghi116–117. Other lects on115–118 remain explicitly attributed controls. General Koci/Simla examples must not acquire invented lect-specific attribution.
- Paired chapter131–140: noun/pronoun/adjective paradigms and prose examples, cardinal/ordinal tables, adverbs/prepositions, copula and verb paradigms, morphology and comparative commentary. Printed column and R./B. labels govern attribution; a source-common head is distinguished from a lect-specific form.
- Twenty-two numbered sentences for **each** lect on141–142:44 physical responses, including the R12/B12 page break. Shared English glosses require scope review, since some pairs have different wording and separate translations. Baghi18 gives an internal water-word alternative that must retain its exact source boundary.
- Notes begin at the bottom of142 and continue143. They contain extra lexical examples, identifiable participial endings, source claims and other-lect comparisons; none may be omitted merely because they are commentary.
- Glossary144–147: the source explicitly says first forms belong to Rampuri and forms after the colon to Baghi. Existing independent glossary evidence is retained; the remaining recorded Rampur body-vowel uncertainty needs occurrence-specific review, not silent deletion.
- Page130 belongs to the preceding Rohru chapter; page148 begins Surkhuli. These boundaries exclude accidental neighboring-lect absorption.

`whole-paired-grammar-first-reading-20260927.tsv` preserves literal printed cells, including suffix-only cells. Its singular/plural columns apply only where the source labels them as such; elsewhere they are storage for the lect's explicit forms, with scope in `source_label`. The common-form field records genuinely shared head evidence. These are provisional first readings, not emitted canonical rows. A later reviewed derivative must preserve all raw cells before any explicitly supported paradigm expansion.

Remaining gates: finish every source-page first reading and unit census; independent whole reading; source claims and grammatical metadata; exact old-key preservation; paired staged proposals/profiles; focused parser tests; fresh independent output audits. Full CLDF/database/browser gates remain deferred under the current no-build instruction.
