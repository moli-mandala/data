# Kotguru full chapter recovery — source stage

Bailey, *The Languages of the Northern Himalayas* (1908), chapter III,
printed pp. 25–33 (one-based PDF47–55), is now transcribed across its complete
scope. The preceding Kotkhai chapter is a separate lect; p34 is blank and p35
begins the Kulu introduction. The original PDF remains at workspace
`tmp/pdfs/bailey-sainji/bailey1908.pdf`; its hash and boundary evidence are in
`scope-census-20260926.json`. The PDF text layer was only a locator: it omits
whole conjunction and verb tables, which were read from page images.

The installed inventory has 487 source units: 485 accepted units generate 610
forms; one general adjective-ending rule is excluded and one ink-obscured
vowel-stack occurrence remains a specific audit-only hold. This includes every noun/pronoun paradigm, adjective comparison,
adverb, preposition, conjunction, auxiliary and verb paradigm, all glossed
prose expressions, the 151 lexical-list cells, 29 cardinals, 17 ordinal and
fraction cells, four numeral prose examples, and all 22 translated specimens.
Printed sentence alternatives are preserved as complete alternative sentences.
No word-by-word segmentation of specimens or etymological inference is made.

All 112 existing entry keys survive in the installation; old form/gloss differences
are itemized in `legacy-corrections-staged-20260926.json`. Source-explicit suffixes
and ditto marks are expanded only when the stem and construction are explicit.
Unprinted forms abbreviated by “etc.” are not invented. Exact source patterns
and grammatical context remain in Notes; gender on genitive agreement forms
is distinguished from lexical referent gender. The source language remains
canonical `Kotguru`; neighboring or comparative lects do not change that mapping.

Breves, macrons, underdots, nasal stacks, and underlined digraphs are retained
literally. Contrastive italic spans are documented in Notes. The profile changes
w→v and ṅ→ŋ under the established house convention, preserves other source signs,
and removes sentence punctuation from normalized Form only. Printed source
inconsistencies are not silently harmonized. The unusual p29 final `uu` spellings
are retained literally and were confirmed by independent audit. The heavily inked final vowel stack in p28 “about thee” remains audit-only after independent adjudication. The legible tērī tā- and exact unresolved mark are recorded, without harmonizing from the parallel up-to expression.

`full-transcription.tsv` and `import_source.py` regenerate the installed CSV
and audit. Legacy pilot data/importer/audit remain retained alongside the
complete staging files. Independent fresh20 review (seed2026092671) found zero
material errors; its addendum verifies all sampled units unchanged after the
single hold adjudication. Seven installed-source regressions plus six staging
checks pass; literal profile and scoped parsing pass for all 610 forms.
The full CLDF/database, graph/reference, full-suite, and updated-database browser
gates remain deferred under the user's explicit no-build instruction. No release
or publication is implied. Regenerate from the workspace root with
`data/.venv/bin/python data/data/other/forms/raw_data/bailey_kotguru_1908/import_source.py --install`.


Only two page images are reused sequentially under workspace `tmp/kotguru-audit`.
They were removed after audit/installation evidence was recorded; the source
PDF, transcriptions, scripts, and audit evidence are retained.
