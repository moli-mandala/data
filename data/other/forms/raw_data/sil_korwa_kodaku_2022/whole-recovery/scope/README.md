# Whole-report scope census

This is a scope handoff, **not accepted lexical extraction or source installation**. All modifications are confined to this scope directory. The source checklist's survey and comparative-source rules apply; complete interlinear words, expressions, questions, and attested answers also require recovery.

The cached original is `tmp/pdfs/korwa-kodaku-2022/source.pdf`, 115 pages, SHA256 `a8efbe88405e27024a7a6ec786cd6fde3e382f0eaf0d0081197d3880ed97eb0c`. PDF11 is printed1, continuing without offset changes through PDF115/printed105. Native text and positioned font spans are retained for every page. These are evidence for extraction, not independently accepted transcriptions. The source is copyrighted SIL2022, not a Creative Commons release; no republication of the complete PDF is proposed.

`whole-page-dispositions.jsonl` accounts for every page exactly once. `outside-lexical-scope-inventory.jsonl` has 26 grouped leads, including historical and ethnographic quotations on printed1–4 and explicit plural `[ku]` on printed49. These are deliberately not assigned languages or exported as words before attribution review. Clan/deity/place names are not automatically indigenous endonyms. The source's `/ə/, /ɐ/, /ʌ/` interchangeability statement (printed47 note8) is a source claim, not permission to overwrite literals.

## Confirmed recovery gaps

| Section | Original pages | Complete scope |
|---|---|---|
| B.5 | PDF66–90, printed56–80 | 25 lists ×210 prompts; retain blank responses, variants and undefined source codes |
| C.2 Job | PDF93–97, printed83–87 | 26 numbered units, 34 phonetic blocks; Pahadi Korwa, Rakkaya |
| C.3 Marriage | PDF97–99, printed87–89 | 12 numbered units, 12 phonetic blocks; Kodaku, Kanda |
| C.4 Life | PDF99–105, printed89–95 | 39 numbered units, 39 phonetic blocks; local Sadri, Dumortoli |
| C.5 Brave Hunter | PDF105–108, printed95–98 | 31 numbered/subnumbered units, 31 story phonetic blocks plus10 target questions; standard Sadri, Kesramal |

The 637 native marker blocks include numbering, questions, glosses, translations and answers; this is **not an emitted-row count**. All126 phonetic blocks are included. C.5's ten answers additionally contain target expressions inside backslash delimiters: they must not be dropped as English-only answers. C.2–C.4 English questions/answers must not generate invented target translations. C.4 has39 phonetic blocks but only38 recognized gloss markers and38 free-translation markers; retain this explicit alignment discrepancy until original reading resolves it. C.3 unit10 uses a script-g `\\ɡ` marker; the inventory preserves that spelling while classifying its function as gloss.

Table7 on PDF27/printed17 and the discussion on PDF28/printed18 establish the narrative languages/sites. The Brave Hunter's source is Abraham1995 AppendixC-11; the Korwa identity of test participants does not make the text Korwa. A fifth School story is mentioned as discontinued after eight subjects, but no transcript is printed. Its absence is documentary, not an extraction failure. This corrects an earlier discovery-note locator of PDF24 for the Brave Hunter attribution.

## Existing rows and reuse

Norton's `../legacy-review/reconciliation.json` and `canonical-row-baseline.jsonl` pin all4,458 existing ordered keys, with every key mapped exactly once. These represent the18 Korwa/Kodaku lists; seven lists A/B/J/T/U/V/k were excluded. The complete comparison response ledger is `../legacy-review/comparison-exclusions.jsonl`. Birjia J has211 audit records covering210 prompts,209 nonblank responses and two blank prompts23/24; record counts must not be confused with prompt counts. Undefined lowercase codes u/n stay specifically unresolved.

New occurrences may reuse an existing key only after exact form, sense, lect and attribution agreement; the new citation must survive. A matching form alone is insufficient. Existing source labels and grammar in paired English prompts also require metadata review. No old rows are deleted or rewritten by this census.

## Original checks and remaining gates

Seven uniquely named120dpi original images were viewed for scope: PDF27,59,93,97,99,105,108. They confirm Table7 routing, explicit `[ku]`, the four start/end boundaries, and C.5 question/answer layout. They do not establish fine phonetic glyph acceptance. All115 pages have native-text and font-position evidence; remaining prose leads need exact original reading and attribution decisions.

Asur's `../identity-review/all-source-routing-proposal.json` covers all25 lists and four texts. Tanmai remains a specific language-identity conflict, not a forced Korwa/Kodaku/Sadri assignment. Birjia requires a reviewed separate registry identity. Respondent biodata, maps, notation/methods, English questionnaires and references remain documentary evidence. No lexical forms should be synthesized from English survey responses or phonetic-key examples.

Next: complete original-based extraction of all seven comparison lists, four texts and outside leads; retain every witness and scoped exclusion. Then reconcile metadata, source claims, sound profiles, references, language/dialect routes, identity and graph, followed by independent output audit and focused actual-parser tests. Canonical installation, full build, database and browser gates remain unperformed and deferred; this report does not claim ingestion completion.
