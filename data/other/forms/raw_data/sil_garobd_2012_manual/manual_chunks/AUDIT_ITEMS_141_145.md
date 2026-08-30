# Manual audit - ESR 2012-007 items 141-145

## Independence and source

- Source: physical PDF page 71 / visible printed page 64.
- Primary review image: the authorized 300-dpi render.
- Small-mark review: targeted 1200-dpi crops for barred `ɨ`, `ʊ`, aspiration,
  glottal stop, ejective apostrophe, below-tied `d͜ʒ`, eng, and
  inverted-breve-below sequences.
- Every form, group number, bracket code, repetition, source space, and explicit
  blank was read from the rendered source. OCR, PDF text, raw legacy glyphs,
  installed forms, and old audits did not supply or verify any reading.
- Site identities remained neutral printed codes `0`, `a`-`p` until after both
  ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 141-145 | Cumulative 1-145 |
| --- | ---: | ---: |
| Items | 5 | 145 |
| Printed response lines | 34 | 1,157 |
| Conceptual site cells | 85 | 2,465 |
| Ordinary attested cells | 84 | 2,405 |
| Source-conflict cells | 0 | 1 |
| Cells containing an attestation | 84 | 2,406 |
| Blank-only cells | 1 | 59 |
| Printed no-entry coordinates | 1 | 60 |
| Not-used cells | 0 | 0 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved source coordinates | 0 | 1 |
| Expanded attested response occurrences | 87 | 2,569 |

Item 145/site `p` is the block's sole explicit `no entry`. Item 144/site `e`
retains its identical group-1/group-7 form, while item 144/site `i` retains
three identical assignments in groups 2, 3, and 7. There are no block-local
not-used cells, ambiguities, illegibles, source conflicts, or unresolved source
coordinates. Cumulatively, item 12/site `p` remains unresolved.

Frozen SHA-256 values:

- generator: `f8eeb102e3465f91d03025c0b59126f5c96010c918d5937e341da539f58105aa`
- line ledger: `4c87f15329e87138a41c399c681b1208a65c605413473872daea9baa9b2d1e34`
- cell ledger: `9c399b83d4fc149b05ef38da2044869734ddcbc4c4b939e7b623feb56c76bb6b`

## Transcription decisions

- Item 141 preserves the `bimɨŋ` / `bimʊŋ` / `bimuŋ` vowel contrast, eng,
  and both barred vowels plus the source space in `kɨr tɨŋ`.
- Item 142 preserves `ʃoŋ` versus `goŋ`, barred `ɨ` in `tʃɨnoŋ`, the
  cluster in `tʃnoŋ`, and all bracket-code coverage.
- Item 143 preserves the ejective apostrophe in `nokʼ`, the
  inverted-breve-below sequence in `jii̯n`, aspiration and the literal slash
  string `bari / gʰor`, and eng in `piŋ`.
- Item 144 preserves glottal stops, aspiration, barred `ɨ`, `ɔ`, below-tied
  `d͜ʒ`, the inverted-breve-below sequence in `pʰɨrdao̯`, and every repeated
  site assignment.
- Item 145 preserves the explicit blank, both inverted-breve-below marks and
  source space in `kʰokai̯ dua̯r`, aspiration, ejective apostrophe, `ɛ`, and
  below-tied `d͜ʒ`.

## Bracket expansion and reconciliation

The 34-line ledger was frozen first. Mechanical bracket expansion produced 88
line-site records: 87 attestations and one printed blank. The conceptual ledger
contains exactly 85 cells because item 144/site `e` has two assignments and
item 144/site `i` has three.

Only after both ledgers were frozen was the legacy audit opened. The comparison
contains 88 rows:

- 80 legacy-installed records and eight legacy exclusions;
- seven independently recovered attestations corresponding to excluded glyphs;
- one manual blank matching the legacy printed-gap record;
- 40 exact codepoint matches among legacy-installed records;
- 40 codepoint differences among legacy-installed records;
- eight codepoint differences among legacy-excluded records.

Reconciliation SHA-256:
`029136298a0ae02f5a42a9798df2f58b91bc268ebe8c8231da50de221b6924f6`.
Comparison results are audit metadata only and neither verify nor alter manual
readings.

## Deferred gates and remaining work

Items 146-307 remain unreviewed. The item 12/site `p` source contradiction
remains unresolved. Site/lect reconciliation, staging, sound-profile
conversion, bibliography/reference validation, shared integration, full
build, graph validation, and browser QA are deferred. No shared output was
changed.
