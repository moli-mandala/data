# Manual audit - ESR 2012-007 items 116-120

## Independence and source

- Sources: physical PDF page 67 / visible printed page 60, right column, and
  physical PDF page 68 / visible printed page 61, left column.
- Primary review images: the authorized 300-dpi renders.
- Small-mark review: targeted 1200-dpi crops for below-ties, alveolar taps,
  aspiration, barred `ɨ`, small-cap `ɪ`, inverted-breve-below, glottal stop,
  and ejective apostrophe.
- Every form, group number, bracket code, repetition, and blank was read from
  the rendered source. OCR, PDF text, raw legacy glyphs, installed forms, and
  old audits did not supply or verify any reading.
- Site identities remained neutral printed codes `0`, `a`-`p` until after both
  ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 116-120 | Cumulative 1-120 |
| --- | ---: | ---: |
| Items | 5 | 120 |
| Printed response lines | 45 | 951 |
| Conceptual site cells | 85 | 2,040 |
| Ordinary attested cells | 81 | 1,986 |
| Source-conflict cells | 0 | 1 |
| Cells containing an attestation | 81 | 1,987 |
| Blank-only cells | 4 | 53 |
| Printed no-entry coordinates | 4 | 54 |
| Not-used cells | 0 | 0 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved source coordinates | 0 | 1 |
| Expanded attested response occurrences | 85 | 2,139 |

The block blanks are item 119/sites `a,b,i,l`, printed together as group-0
`no entry`. Item 116 retains identical group-1/group-2 assignments at sites
`a,g,d,n`. There are no block-local not-used cells, ambiguities, illegibles,
source conflicts, or unresolved source coordinates. Cumulatively, item 12/site
`p` remains unresolved.

Frozen SHA-256 values:

- generator: `6627ad0895437ee76dcf60da4b136b2cca74f30ffc1899f1ed6a8ee3b3d958e2`
- line ledger: `e7a6013d725adde0c21c754c04bf96da122aa49f7397268ed99e230336d7be05`
- cell ledger: `d1e0bacb0c9226efb4b875afff9294dcc393847119c217800c06a0feb2f77885`

## Transcription decisions

- Item 116 retains four repeated group-1/group-2 assignments and preserves
  below-tied `d͜ʒ`, alveolar `ɾ`, aspiration, glottal stop, barred `ɨ`, and
  every inverted-breve-below mark.
- Item 117 preserves below-tied `d͜ʒ`, alveolar `ɾ`, aspiration, small-cap `ɪ`,
  glottal stop, ejectives, source spaces, and inverted-breve-below.
- Item 118 keeps alveolar `ɾ`, glottal stop, and the two distinct
  inverted-breve-below sequences in `tʃia̯ŋ` and `tʃiɛ̯ŋ`.
- Item 119 records all four printed no-entry coordinates and preserves barred
  `ɨ`, small-cap `ɪ`, aspiration, ejectives, and inverted-breve-below.
- Item 120 preserves barred `ɨ`, small-cap `ɪ`, aspiration, glottal stop,
  ejective `ʼ`, and alveolar `ɾ`.

## Bracket expansion and reconciliation

The 45-line ledger was frozen first. Mechanical bracket expansion produced 89
line-site records: 85 attestations and four no-entry records. The conceptual
ledger contains exactly 85 cells because four item-116 coordinates have a
second printed assignment.

Only after both ledgers were frozen was the legacy audit opened. The comparison
contains 89 rows:

- 74 legacy-installed records and 15 legacy exclusions;
- 11 independently recovered attestations corresponding to excluded glyphs;
- four exclusions corresponding to printed no-entry coordinates;
- 14 exact codepoint matches among legacy-installed records;
- 60 codepoint differences among legacy-installed records;
- 15 codepoint differences among legacy-excluded records.

Reconciliation SHA-256:
`3263099a3bdf537273ce7524cac653b484f422791ab806e74f5446175034bbb7`.
Comparison results are audit metadata only and neither verify nor alter manual
readings.

## Deferred gates and remaining work

Items 121-307 remain unreviewed. The item 12/site `p` source contradiction remains
unresolved. Site/lect reconciliation, staging, sound-profile conversion,
bibliography/reference validation, shared integration, full build, graph
validation, and browser QA are deferred. No shared output was changed.
