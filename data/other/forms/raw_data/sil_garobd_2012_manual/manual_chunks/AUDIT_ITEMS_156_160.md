# Manual audit - ESR 2012-007 items 156-160

## Independence and source

- Sources: physical PDF pages 72-73 / visible printed pages 65-66.
- Primary review images: the authorized 300-dpi renders.
- Small-mark review: targeted 1200-dpi crops for barred `ɨ`, aspiration,
  glottal stop, ejective apostrophe, below-tied `d͜ʒ`, eng, `ɛ`, the tap `ɾ`,
  and inverted-breve-below sequences.
- Every form, group number, bracket code, repetition, source space, and the
  item-level `[not used]` disposition was read from the rendered source. OCR,
  PDF text, raw legacy glyphs, installed forms, and old audits did not supply
  or verify any reading.
- Site identities remained neutral printed codes `0`, `a`-`p` until after both
  ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 156-160 | Cumulative 1-160 |
| --- | ---: | ---: |
| Items | 5 | 160 |
| Printed response/disposition lines | 30 | 1,251 |
| Conceptual site cells | 85 | 2,720 |
| Ordinary attested cells | 68 | 2,624 |
| Source-conflict cells | 0 | 1 |
| Cells containing an attestation | 68 | 2,625 |
| Blank-only cells | 0 | 61 |
| Printed no-entry coordinates | 0 | 62 |
| Not-used cells | 17 | 34 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved source coordinates | 0 | 1 |
| Expanded attested response occurrences | 80 | 2,808 |

Item 159 is printed once as `[not used]` for the whole item. Its 17 conceptual
site cells are therefore `not_used`, not ordinary blanks and not 17 separate
printed disposition lines. Item 158/site `n` retains distinct group-2 `ata`
and group-3 `tʃamotʃ` assignments. Item 160/sites `a,b,c,d,e,g,h,l,m,n,o`
retain identical group-1/group-4 `hatur` assignments. There are no block-local
blanks, ambiguities, illegibles, source conflicts, or unresolved coordinates.
Cumulatively, item 12/site `p` remains unresolved.

Frozen SHA-256 values:

- generator: `53a665d8f0bed815942fb07b2c417e6bfdcd97c0369c2143786ade419daeb5e4`
- line ledger: `0802aa773610656505222c096db561e4b87226dff54f0cac0ad785cf6752be67`
- cell ledger: `d4c7acd0ca9f571f0cd7890233728337ebc7657fafdcb78bfa8b56437ffa17a1`

## Transcription decisions

- Item 156 preserves aspiration, barred `ɨ`, eng, `ʃ`, and the
  inverted-breve-below sequences in `kʃai̯` and `kʰsai̯`.
- Item 157 preserves glottal stops, ejective apostrophes, `ɛ`, below-tied
  `d͜ʒ`, aspiration, and the tap in `d͜ʒʰaɾu`.
- Item 158 preserves the two printed vowel orders in `kortʃali` and
  `kortʃila`, and the overlapping site `n` assignment.
- Item 159 preserves the source's exact whole-item `[not used]` disposition and
  mechanically applies its scope to all 17 conceptual site cells.
- Item 160 preserves the source space in `d͜ʒoŋ mnoʔ`, barred `ɨ`, glottal
  stop, and both printed `hatur` group assignments.

## Bracket expansion and reconciliation

The 30-line response/disposition ledger was frozen first. Mechanical expansion
produced 85 conceptual cells: 68 attestations and 17 cells carrying the single
whole-item not-used disposition. Attested bracket expansion produced 80
line-site occurrences because item 158 has one overlap and item 160 has eleven
repeated coordinates.

Only after both ledgers were frozen was the legacy audit opened. The comparison
contains 81 rows: 80 attested line-site records plus one whole-item disposition.

- 77 legacy-installed records and four legacy exclusions;
- three independently recovered attestations corresponding to excluded glyphs;
- one manual whole-item not-used disposition matching the legacy record;
- 47 exact codepoint matches among legacy-installed records;
- 30 codepoint differences among legacy-installed records;
- four codepoint differences among legacy-excluded records.

Reconciliation SHA-256:
`7c34686ea6f8528b2f84ccc3c1bff8f5c9961dacf2a16a8612481d19205c3313`.
Comparison results are audit metadata only and neither verify nor alter manual
readings.

## Deferred gates and remaining work

Items 161-307 remain unreviewed. The item 12/site `p` source contradiction
remains unresolved. Site/lect reconciliation, staging, sound-profile
conversion, bibliography/reference validation, shared integration, full
build, graph validation, and browser QA are deferred. No shared output was
changed.
