# Cholanaickan Nair 2013 source-stage handoff

The Jambu source-ingestion checklist was read in full. Its PDF, grammar,
dictionary/glossary and comparative-source instructions apply. This is a
**source-stage installation**, not a claim that the full ingestion is complete.
The user barred a local full CLDF/database build and remote work during this
stage; those compiled gates remain open.

**Exact source and scope.** Ravi Sankar S. Nair, *Tribal Languages of Kerala*,
*Language in India* 13(7), July 2013, complete Cholanaickan chapter printed
and PDF pp. 19–21. Cached 58-page PDF SHA256
`4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af`.
The source provides a publication URL but no explicit reuse licence in the PDF.
Its p. 21 bibliography reports R. Muralidharan's 1988 unpublished dissertation;
we did not independently consult that work. This is the same physical Nair
volume as the installed Aranadan chapter, with a distinct chapter-specific
citation key, not a second edition.

**Accounting.** All three pages were visually inspected. The first reading
has 33 stable physical/editorial blocks and 116 form mentions, plus one
bibliography block without a form. The installed CSV has 104 rows. Twelve
mentions are audit-only: three ethnographic names and `aale` lacking secure
Cholanaickan-language attribution, one Tamil comparison, one repeated
identical `-indu` in a purported variant pair, three zero exponents, the
`-annu` vocative stem condition, and two comparative-only feminine markers.
The accepted audit has 117 records, preserving every historical source claim
and disposition. Six source-specific lexical/grammatical etymological statements
survive as attributed free text, with no inferred Proto-Dravidian graph links.

**Transcription and structure.** The source has a usable selectable text layer,
but the complete original images show 14 forms with a horizontal low line on
selected `n` letters absent from that layer. Nineteen visible low-line marks
are encoded as U+0332; their phonetic value is not inferred. `Form` and
`Original` retain source notation without conversion. The source offers no
independent IPA/phonemic tier, so `Phonemic` and `Native` are blank. Full
104-row symbol inventory is NFC with no replacement glyph. Grammar tags follow
explicit paragraph/table scope. Eleven source-explicit same-chapter variant
links are structured; six edge classes remain held. No borrowing or derivation
edge is inferred. The source's duplicate instrumental markers and `tereli`
“left (someone) open” in a hortative paragraph remain reviewable anomalies.

**Identity and references.** One new base language `Cholanaickan` was
registered with the existing South Dravidian I clade. The source gives a
Nilambur Valley community area but no elicitation coordinates; Glottocode and
coordinates remain blank pending independent primary identity evidence. No
named lect or stable site was asserted, so no new dialect is registered.
`nair2013cholanaickan` and reported-only `muralidharan1988cholanaikka` are in
`sources.bib`; Pybtex formats both, and exact shared-volume/canonical-source
provenance is recorded. Existing bibliography entries and language rows were
preserved. The raw 104-row CSV uses the former key with printed-page locators.

**Validation.** Independent complete first-reading review plus a final fresh,
disjoint 20/20 original-output audit passed with zero material errors
(`sampling-setup/final-freeze-original-audit-20260927338.json`). The guarded
installer pinned the PDF/proposal/audit/graph hashes, installed exact CSV SHA256
`771abcdfe9f6e0c25198397c8afd37e4d592ab982b34e3d8be8781eaa582c308`,
and preserved durable form identity state. A repeated install command is an
exact byte no-op. Six focused tests pass, including all unit/row accounting,
source tags/graph endpoints, low-line literals, canonical byte parity, and an
actual 104-row `make_cldf.parse_file` scope check with zero conversions/errors.
The installation receipt and the narrow bibliography title/provenance amendment
receipts are retained source-locally.

**Open gates.** No full `make all`, full pytest suite, compiled CLDF
source-key/ID/edge inspection, browser database build or live app QA was run.
The user explicitly controls database refresh. Representative app entries for
later QA: `nair2013cholanaickan:p19:accusative-example:form1` (underlined n and
variant), `nair2013cholanaickan:p20:personal-pronouns:form1` (person and literal
low line), and `nair2013cholanaickan:p21:lexical-list:form1` (ordinary unlinked
lexeme). No commit, push or release was made.
