# Lāḷas, Sheth and Kharia — ingestion work in progress

The complete SOURCE_INGESTION_CHECKLIST.md is active. Dictionary/glossary and
website/API addenda apply; comparative claims and PDF/OCR addenda apply where
used. User confirmed Lāḷas permission via SARVA, approved the Peterson repository
challenge, and then confirmed **all selected-source permissions**, including
Living Dictionaries, on 2026-09-11. No further reuse approval is pending.

## Kharia

The separately attributed Kharia Living Dictionary public SQLite snapshot of
2026-08-20 is installed: 452 source entries → 521 rich rows / compiled nodes,
68 variant edges, 453 unlinked nodes, no lexical exclusions. Six native-only
entries and 115 entries with source-notation review flags are retained. Complete
per-record audit, source metadata, Dhelki/Dudh registry rows, explicit preservation
profile and compiled tests are installed. The fresh extraction audit found 0/20
material errors. Detailed policy and validation are in
`20260911-kharia-living-review.md` and its adjacent audit artifacts.

Final combined focused tests: 52 passed (including 23 parser tests). All seven build stages completed; the
final manual-survey gate failed two unrelated survey assertions. Full suite with
importlib collection: 1,737 passed / 31 failed / 18 skipped. Therefore the full
validation gate remains open. Isolated local app QA passed source counts, variant
links, dialect filters, native-only search and ELEPHANT concept membership.
Packed DB 44.12 MB passes 50 MB limit; expanded 99.98 MB exceeds the 97 MB warning.
No release artifact was replaced and nothing was deployed.

Peterson (2009), DOI 10.5070/H90023672, remains a distinct source. The authorized
repository challenge was completed and the 228-page PDF opened in Chrome, but
browser export/download did not produce a local PDF. No additional Peterson
lexical content has been extracted or attributed to Living Dictionaries. Existing
Peterson-derived Jambu records remain untouched by this task.

## Sheth

The original DDSA Prakrit–Hindi transcription (advertised edition 1923–1928,
updated May 2026) is fully cached: 952 web pages, 41,638 headword articles; final
headword ह्रास hrāsa. Continuation placeholders: 546, 658, 678, 679, 687.
The linked first-edition scan ends on printed page 1278, plus supplementary
references/corrigenda, so web locators explicitly say DDSA web page. Printed
edition/supplement reconciliation is still unresolved. Visual reference-table QA
found duplicated PDF pages 8/10 and 9/11 and missing printed reference pages 10–11;
see `audits/20260911-sheth-frontmatter-review.json`.

Current proposal: 41,519 accepted-for-review articles → 63,741 rows; 119 malformed
native/roman pairs excluded with full raw audit. **No Sheth rows installed.**
References, paradigms, inline compounds, independent embedded headwords, bound
headword expansion, dialect mapping and final sound profile need further review.
No ancestry has been asserted from mere Sanskrit equivalents. Fixed parser error
classes include nested meaning duplicates, quoted See examples and compound
etymology leakage; 12 Sheth regression tests pass. The current proposal is not a
final 0/20 extraction audit or a validated import.

ISJS's modern English translation is separate and incomplete: its endpoint gave
20,912 entries (अ–दसु). It has not been substituted for the full dictionary.

## Lāḷas

Second edition 2013, DDSA revision November 2021. Acquisition script follows
sequential navigation and hashes every HTML page; explicit empty continuation
pages remain flagged. Final acquisition: 6,136 pages / 131,508
articles, navigation complete=True. Last headword: ह्वैयौड़ौ hvaiyauṛau.
There are 61 explicitly marked empty continuation pages, listed in the acquisition manifest. Cache: `tmp/lalasa-ddsa-20260911/`.
The publisher's linked first-edition frontmatter supplied the verified grammar
abbreviations. Poetry star is a poetic marker; क्रि.प्र. is verbal usage,
क्रि.प्रे. is causative. Do not confuse these with reconstruction or each other.

The complete proposal covers 131,508 articles: 131,267 accepted-for-review articles
→ 301,678 proposed rows, with 241 malformed native/roman pairs excluded and audited.
This is a development output, not a validated lexical count. A fresh seeded audit
(20260915) found 11/20 records with material errors or incomplete lexical structure:
inline citations in glosses, unquoted or unsplit See references, and unexpanded
feminine/alternate forms. Full raw/output comparisons are saved in
`audits/20260911-lalasa-full-snapshot-audit.json`. Other reference, dialect,
graph and sound-profile gates remain deferred.

The parser separates paired headwords, numbered/POS sections, quotations and
etymological prose. Derived-family and morphology regions remain lossless in audit.
Fixed classes include special-description abbreviation वि.वि., parenthetical See
labels, continued/duplicate printed numbering, numeric quantities, poetry markers,
and feminine prefixes swallowing the main definition. Eleven Lāḷas parser tests
pass. Source-wide idioms, unexpanded forms, reference resolution and transcription
review still prevent installation.
**No Lāḷas rows installed.** A structured SARVA export has been requested as a
possible way to resolve corrupted/displaced source fields; permission is already
confirmed and is not the question.

## Shared-workspace preservation

Existing Mewari, Vedda, western-survey and overlay changes were preserved.
The existing checklist generator deleted authored Markdown during stale-file
cleanup. That cleanup now only deletes files with its generated-checklist header;
a regression test proves authored reviews survive. Affected authored reviews were
recovered from their exact recorded creation/update commands and tracked originals.
No commit, push or deployment was performed. This report does not claim the three
requested ingestions are complete.
