> Historical recovery progress record, superseded after installation by README.md and postinstall-verification-20260926.json. Canonical full source now installed; the frozen source inputs remain unchanged.

# Complete Rangri source recovery — in progress

The source-ingestion checklist is active with its survey/comparative-table and
scanned-source addenda. The canonical CSV remains the five-row pilot until all
sections are recovered, the proposed whole source is frozen, and an independent
raw-to-output audit and focused checks pass. No database, full CLDF build, full
suite or browser refresh is authorized.

## Scope and current evidence

- Shared grammatical sketch, printed 54–59: 280 physical units. The explicit
  standing rule on p.54 applies unqualified remarks to both Rangri and Malvi
  proper. Explicit Malvi/other-language comparisons remain controls.
- Rangri comparative-table column, printed 305–321: all 241 prompts independently
  reread and reconciled; source blanks 172, 174 and 201. The separate Malvi and
  Nimadi columns are controls. The table expands to 317 literal forms, with every
  abbreviation expansion recorded alongside the original cell.
- Complete Roman interlinear specimens, printed 249–251 and 254–256: 823 reviewed editorial aligned units from 827 physical fragments, yielding
  822 expressions before exact reuse; all six pages were reread against grayscale originals.
- Native witnesses, printed 248 and 252–253, plus the p.55 orthographic example:
  65 original lines and 823 atom-level alignments reviewed; two p.55 alternatives
  remain unpaired audit evidence. One place-name glyph uncertainty is retained.
- Printed 256–257 free English translation is contextual evidence, not a new set
  of lexical attestations. Printed 247 and 258 establish the specimen boundaries.

`whole-source-scope-20260926.json` records the complete scope. This document is a
progress record, not a claim of completed ingestion.

## Representation and transcription

`prepare_full.py` stages the complete source with canonical files untouched.
Its 1,170 rows derive from 1,393 rows before exact same-analysis reuse; 223 repeat
attestations preserve all citations and source-unit aliases in the audit. The
1,346 audit units include 78 excluded controls, 15 isolated bound-morphology
units, three source blanks, one grouped-expression continuation and two unpaired
native alternatives. Six simple-present cells each have three explicitly stated
functions; row expansion does not inflate their physical-attestation count.

`grammar-source-commentary-review-20260926.json` retains package-level linguistic
claims and links the independent 25-class commentary review. Per-attestation
claims survive in row Notes. Comparisons and cited authorities are attributed to
the source, with other-lect controls never silently assigned to Rangri.

The source's Roman notation is preserved literally, including macrons, nasal
marks, underdots, superscript a, word boundaries, hyphens and printed v/w
variation. The profile lowercases capitals and maps conventional ṅ to ŋ; Form retains the exact source spelling. Phonemic
remains blank because this is historical transcription, not independently supplied
IPA. An explicit profile-policy exception preserves source w/W rather than merging with v. Native script has its own field and alignment evidence.

The same-source grammar mārī-nē and table Māri-nē differ in vowel length; both
are preserved. The three nasalized what-forms kaĩ, kaī̃ and kā̃ī̃ remain distinct.
`grammar-targeted-class-rereview-20260926.json` records corrections and a full
six-page review following the diyō misreading, including i/ī, n/y, a/ā and v/w.
The secondary Nordic HTML is a comparison witness; original images govern
conflicting readings. The grayscale original overturned the binary-image reading
hā, hai: its open e clearly reads hē, hai. All six grammar pages were reread
against these superior grayscale witnesses; corrections remain recorded.

All five existing pilot Entry_Keys survive, including transcription corrections
to their forms. No cross-source rows are removed. Printed alternatives are kept
as separate keyed rows; the source does not establish etymological, donor or
directional variant relationships, so no unsupported graph links are added.

## Source and language metadata

Bibliographic key: `grierson1908malvirangri`, Grierson 1908 LSI IX, part II.
Original public-domain edition; original scan and alternative-original hashes
are recorded in the source provenance manifests. DSAL original grayscale pages
are retained; generated crops are overwritten and deleted after review.

The canonical base is existing `Malw` (Glottocode `malv1243`, legacy display name
“Malwai”), with registered dialect
`dialect:Malw:lsi1908-malvi-rangri:Rangri`. The source locates the specimens in
Dewas Junior Branch; an exact elicitation site is not supplied, so no point is
invented. The existing base-language metadata is not reinterpreted by this source.
All row references use printed-page and item/line/cell locators.

## Current validation and remaining gates

Five focused staging tests pass: complete table expansion, source-literal
corrections, grammar/source scope, stable pilot keys, and complete profile/tag
coverage. The complete 1,170-row profile and registered tags pass.

Still required before source-stage installation: whole-source frozen hashes and independent fresh seeded 20 audit, edge-case
checks, final profile/metadata/reference reconciliation, scoped parser and
source-local graph checks, importer parity, and final checklist/ledger updates.
Full build, compiled identity/graph/reference checks, full suite and browser QA
remain explicitly deferred by the user's instructions.
