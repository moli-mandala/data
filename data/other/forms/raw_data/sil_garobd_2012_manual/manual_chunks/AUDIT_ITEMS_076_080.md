# Manual audit - ESR 2012-007 items 76-80

## Independence and source

- Source: physical PDF pages 61-62 / visible printed pages 54-55.
- Primary review images: the authorized 300-dpi renders.
- Small-mark review: targeted 1200-dpi crops for below-ties, aspiration,
  barred `ɨ`, inverted-breve-below, glottal stop, and ejective apostrophe.
- Every form, group number, bracket code, and repetition was read from the
  rendered source. OCR, PDF text, raw legacy glyphs, installed forms, and old
  audits did not supply or verify any reading.
- Site identities remained neutral printed codes `0`, `a`-`p` until after both
  ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 76-80 | Cumulative 1-80 |
| --- | ---: | ---: |
| Items | 5 | 80 |
| Printed response lines | 40 | 614 |
| Conceptual site cells | 85 | 1,360 |
| Ordinary attested cells | 85 | 1,317 |
| Source-conflict cells | 0 | 1 |
| Cells containing an attestation | 85 | 1,318 |
| Blank-only cells | 0 | 42 |
| Printed no-entry coordinates | 0 | 43 |
| Not-used cells | 0 | 0 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved source coordinates | 0 | 1 |
| Expanded attested response occurrences | 97 | 1,425 |

All 85 block cells are attested. Item 76 preserves overlapping assignments at
sites `d` and `k`; item 77 preserves the repeated site `m`; and item 80 retains
all group-1/group-3 repetitions at sites `k,j,b,c,d,h,i,l,m`. There are no
block-local blanks, not-used cells, ambiguities, illegibles, source conflicts,
or unresolved coordinates. Cumulatively, item 12/site `p` remains unresolved.

Frozen SHA-256 values:

- generator: `030a606690b94a4434607f1c5ebabaf7582aca92b9699b88cbeca7d9fba1cfbd`
- line ledger: `c3e59a31ab8c000251143236e05b87f69424b186c5397a9e725d261a1e59e4e6`
- cell ledger: `175cd971fc73057b2ae3ef8d2744ade4669a8a8a9925f265c4dec748339e3f37`

## Transcription decisions

- Item 76 preserves the visibly printed below-tie in `tʃid͜ʒoŋ`, the overlapping
  site assignments, aspiration, ejective `ʼ`, and inverted-breve-below.
- Item 77 retains `ɛ`, `ŋ`, and final glottal stop exactly, including the
  overlapping site `m` responses `luklak | bɛŋboŋ`.
- Item 78 keeps `kɨi`, `kui`, `kʰsu`, and `ksu` distinct.
- Item 79 preserves all `ɛ`/`ɨ` distinctions and every inverted-breve-below mark
  in `mɛŋgao̯`, `mɛŋgou̯`, `bɨi̯ra`, `bilai̯`, and `mio̯`.
- Item 80 retains every group-1/group-3 repetition and the contrasts among
  `mɨʔsɨu̯`, `mɨsɨ`, `maʔsɨu̯`, `maʔsu`, `maʔʃu`, and `maʔtʃʰu`.

## Bracket expansion and reconciliation

The 40-line ledger was frozen first. Mechanical bracket expansion produced 97
attested line-site records. The conceptual ledger contains exactly 85 cells
because of the source's overlapping and repeated assignments.

Only after both ledgers were frozen was the legacy audit opened. The comparison
contains 97 rows:

- 76 legacy-installed records and 21 legacy exclusions;
- all 21 exclusions are independently recovered attestations;
- 30 exact codepoint matches among legacy-installed records;
- 46 codepoint differences among legacy-installed records;
- 21 codepoint differences among legacy-excluded records.

Reconciliation SHA-256:
`0a7f29e922c4ae0813366fdf42d13681530474fb6a3aa30ebe70652dcd8893d0`.
Comparison results are audit metadata only and neither verify nor alter manual
readings.

## Deferred gates and remaining work

Items 81-307 remain unreviewed. The item 12/site `p` source contradiction remains
unresolved. Site/lect reconciliation, staging, sound-profile conversion,
bibliography/reference validation, shared integration, full build, graph
validation, and browser QA are deferred. No shared output was changed.
