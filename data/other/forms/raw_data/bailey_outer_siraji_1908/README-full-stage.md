# Outer Siraji full chapter recovery — staged

Bailey, *The Languages of the Northern Himalayas* (1908), Outer Siraji chapter,
printed pp37–43 (PDF59–65), is transcribed in full, with four explicitly
Outer-attributed case-marker units from the shared introduction p36 (PDF58).
Inner Siraji begins p44. The public-domain original remains at workspace
`tmp/pdfs/bailey-sainji/bailey1908.pdf`, SHA256
`953f5da5ee9bc341cdf7eb558920e824dc6f15f7473a8c0d5c8134c401ad60b5`.
The shared introduction's geography and other-lect comparisons are contextual,
not additional Outer Siraji attestations. Its transcription description was
reviewed, including the dotted breve on i and contrastive italic u.

The complete census is 367 units: 364 accepted units yield 450 forms, while
three general suffix patterns lack lexical stems and remain audit-only. There
are no typography holds in this draft. All 79 formerly generic holds are
recovered literally. Included are every noun/pronoun paradigm, adjective and
comparison example, all adverbs/prepositions, auxiliaries and verb paradigms,
all whole ability/necessity expressions, 123 lexical cells, 46 cardinals,
nine ordinals, and all five numbered specimens. The latter retain their
printed numbering 6, 7, 17, 19, 20, which cross-refers to Kotguru. The complete
page/heading census and explicit expansion limits are recorded in
`full-scope-staged-20260926.json`.

The PDF text layer omits entire grammar tables (p38 is effectively empty), so
source page images were the authority. Only explicit forms and unambiguous
suffix/ditto substitutions were expanded. Sheep plural ellipses and paradigm
“etc.” did not license fabricated forms. Whole translated expressions remain
whole, with no guessed segmentation. Genitive agreement gender is explained
separately from the lexical referent. Morphological labels use registered tags;
comparative prose without a matching tag remains in Notes.

All 100 old entry keys survive. Corrections to pilot forms/glosses are itemized
in `legacy-corrections-staged-20260926.json`. The source language remains
canonical `OuterSiraji`; no narrower locality or speaker is invented.
Source breves, macrons, underdots, nasals, underlined digraphs, and inconsistent
spellings are preserved literally. Italic u within upright words is recorded
in Notes. The profile applies established w→v and ṅ→ŋ conventions and removes
sentence punctuation only from normalized Form; Original retains the print.
No unsupported phonemic or etymological interpretation is imposed.

`chapter-transcription-work.tsv` and `import_source_full.py` reproduce
`literal-staged.csv` and `full-staged-audit.jsonl`. Six focused tests pass,
including exact CSV/audit regeneration, all old keys, suffix and diacritic
classes, grammar, complete expressions, and 450/450 profile/scoped parsing.
Independent source audit is pending; canonical installation is not yet done.
Full CLDF/database, graph/reference, full-suite, and updated-database browser
gates are deferred under the user's no-build instruction. This is a source
stage, not a release. Two scratch images are overwritten sequentially and will
be deleted after audit/installation evidence is recorded; original PDFs and
source/audit files are retained.
