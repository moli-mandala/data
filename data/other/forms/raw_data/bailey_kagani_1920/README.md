# Bailey1920 Kāgānī whole-source recovery

The approved r2 source-stage installation contains **1,306 forms and1,217 audit
units**, retaining all329 previous glossary keys. The one-row source-glyph
uncertainty propagation addendum was independently approved and installed; no new
literal reading or key was introduced. Full database/build/browser gates are
explicitly deferred, so this is not a claim of completed downstream ingestion.

The source is Thomas Grahame Bailey, *Linguistic Studies from the Himalayas*
(Royal Asiatic Society,1920), public domain. The retained310-page original is
`tmp/pdfs/bailey1920/bailey1920.pdf`, SHA256
`7a1577a2d2a24eca8b95e9b1ed700ef01f716d1069e4468f36a673611d093b39`.
Printed page n is one-based PDF n+26. The Internet Archive source is
<https://archive.org/details/linguisticstudie00bailrich>.

## Source coverage

The complete target chapter covers pronunciation and introduction87–90,
grammar92–101, Prodigal Son101–104, notes and the separate North Wind and Sun
opening105, and glossary106–109. The separately headed Bahramgala introduction91
is explicitly accounted as other-lect context, not silently assigned to Kāgānī.
The grammar columns change between number, noun and pronoun classes; they are
not two dialect columns. All identifiable bound morphology, complete translated
clauses and source-supported abbreviated paradigm expansions are retained.
Unprinted regular cells or stems are not generated.

The narratives retain362 primary parable atoms, six parentheticals and16
contiguous whole-expression spans; the second story retains34 atoms and three
whole-expression spans. Complete expressions preserve the printed interlinear
translation, not an invented free translation. Distinct parenthetical meanings
such as sheep/pigs and God/prophet are not variants. The three glossary
cross-references are explicitly resolved in the audit; exact second reuses its
ordinal row with both locators, while river/stream and spruce/fir are not forced
into semantic identity.

## Editorial and review evidence

All original first readings, the failed r1 output freeze and the accepted r2
freeze remain immutable. The full introduction review61units corrected two
issues: a plain italic l must not acquire an unprinted underdot, and Panjabi
chăḍḍṇā is a comparator rather than another target leave head. A source-wide
legacy metadata review checked261cells/329rows and corrected fatten
pălĕādū→pălĕādā with its stable key. It also restored explicit grammar labels,
correctly scoped the spoil phrase note and retained cross-reference evidence.

The first actual-output audit failed3/20units on metadata. Its class repair
reviewed96 interlinear/pronoun cases: English -ing alone does not establish a
present participle, and English by alone does not establish ergative case.
Source90 infinitive morphology supports gēṇā/băjēṇā; source93–94 paradigms and
actual clause subjects support person/number where claimed. Procedural tagging
instructions were removed from visible Notes. Fresh r2 audit passed20units/
23rows plus four targeted regression checks with zero material errors.

One specifically unresolved source glyph is preserved: p97 feminine third-plural
past äsīā̃ has two separated upper points, possibly a damaged macron rather than
diaeresis. The literal candidate and alternative āsīā̃ remain in the audit; the approved
metadata addendum propagates that uncertainty visibly without phonetic claims.
No generic typography holds remain. Approximate pronunciation and comparator
claims remain attributed to Bailey rather than converted into new etymologies.

## Identity, transcription and reproducibility

The existing base remains Northern Hindko (`nort2662`), with source dialect
`bailey1920-kagani`; no coordinates or new lect identities are invented.
Original forms retain source breves, underdots, underlining and tone strokes.
U030D/U0329 encode Bailey's literal vertical strokes, not IPA stress or
syllabicity. House display mappings w→v and ṅ→ŋ remain. No separate phonemic layer
or unsupported graph links are generated. All historical keys survive.

Use `install_whole_source.py --install` for coordinated regeneration. It requires
the fixed accepted freeze and hash-bound independent audit, replays r1 and r2 in
a small isolated source tree, validates every output byte, and accepts only the
historical or already-installed canonical state. YAML routes this coordinator;
the old `import_source.py --install` delegates to it and cannot overwrite the
whole chapter with329glossary rows. The untouched historical importer, CSV,
YAML, profile and audit are in `historical-glossary-installation/`.

Scoped actual-parser validation preserves ordered keys, Original, Notes, Source,
Tags and literal display for all1,306rows in NFC and NFD. The92-rule profile passes
house policy;17 focused historical/proposal/revision/canonical tests pass. Repeat
installation reproduces the approved bytes. Final evidence lives in
`whole-installation-result-20260927.json`, `whole-installed-validation-20260927.json`
and `whole-source-recovery/`.

Deferred gates: full CLDF/database rebuild, full suite, compiled persistent-ID and
graph validation, generated-reference refresh and representative browser entries.
No database was built, no remote was used, and no commit or release was made.
