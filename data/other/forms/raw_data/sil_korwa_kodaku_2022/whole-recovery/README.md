# Whole-report recovery

This directory recovers lexical evidence omitted by the original target-wordlist
import of Behera (2022). The original package README describes that historical
4,458-row import. The reviewed source files were installed through the guarded
transaction recorded in `installation-receipt.json`. This is a source-stage
installation; the compiled CLDF and browser database have not been refreshed.

## Coverage and accounting

The pinned 115-page publisher PDF is documented in the parent README. Its embedded
text layer supplies literal evidence, checked against original page renderings.
No new OCR transcription is substituted for the original. The PDF is not
redistributed, and no permissive licence is asserted for it.

All 25 Appendix B.5 wordlists, four Appendix C texts, and lexical evidence elsewhere
in the report are accounted for. The installed source files contain 6,864 rows:

- 4,459 repaired legacy Korwa/Kodaku rows, preserving all 4,458 existing entry keys;
- 1,370 additional comparison-list rows;
- 1,033 text rows, including 117 whole expressions/questions/answers;
- two explicitly attributed introductory word-origin examples.

The 2,405 additions comprise Asuri 224, Sadri 1,359, Birjia 209, Mundari 223,
Korwa 300, and Kodaku 90 rows. Birjia is a distinct base language;
seven additional survey/text localities are registered as dialects. Coordinates
remain blank where the evidence does not establish a point.

The whole-source witness audit contains 6,823 records: 6,135 historical wordlist
records, 638 text-layer events, 48 outside witnesses, and two undefined site-code
records. These units are not interchangeable with output rows. Explicit links
account for every installed source row and held candidate, including the superseded
V208 blank. The historical audit is retained as evidence of the earlier import.

## Unresolved evidence and editorial decisions

The 246 Tanmai comparison candidates remain held because the source's language
attribution conflicts. Twenty outside witnesses have unresolved language
attribution; 26 are documentary evidence. Two undefined lowercase site codes
are not silently mapped to uppercase codes. Blank wordlist responses stay audited.

Sixty-six internal text spans have uncertain word-level alignment. Their target
content remains represented in whole expressions; the uncertain child spans do
not become independently glossed lexical rows. Continuous interlinear glosses
were manually partitioned, not blindly zipped by token position. Source C.5 column
alignment is preserved even where its gloss order is linguistically unexpected,
with typed uncertainty. Source punctuation such as `sa.b` remains literal.

The repaired legacy rows retain their existing sound profile. New evidence uses
literal transcription with conversion disabled and meaningful hyphens preserved;
Native and Phonemic stay blank. Grammar is structured only where source evidence
supports it. Survey similarity groups create no etymological edges. Two attributed
word-origin claims are preserved as commentary, without inventing graph links.

Both source files use explicit entry keys in the initial pipeline deduper. Without
this policy, equal forms from different prompts or text occurrences can collapse
before their distinct analyses are preserved. Later compiled identity and merge
behavior still needs verification during the deferred full build.

## Reproduction and remaining gates

Preparation scripts in this directory assemble comparisons, text evidence,
language/dialect drafts, bibliography, transcription settings, and the witness
audit. `integration-proposal/` contains proposed assets and focused validation
reports. Owner and independent reviews retain exact input hashes and failed
samples; repaired outputs require fresh review rather than repinning old passes.

The guarded installer is under `installation/`. Its active exact-hash manifest,
transaction journal, retained baselines and receipt document the source-file
installation. A repeated invocation verifies all approved bytes without rewriting
them. The historical importer refuses to overwrite this whole-source recovery.

Focused checks cover parser survival, stable keys, field boundaries, registry
parents, bibliography preservation, transcription routing, and physical-source
accounting. Final acceptance, canonical source-file installation and installed-file
checks passed. Full CLDF generation, the full suite, compiled identity/graph
verification, and any browser-database refresh remain deferred. No release,
commit, or push is part of this source-stage work.
