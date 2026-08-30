# Manual audit - ESR 2012-007 items 91-95

## Independence and source

- Sources: physical PDF page 63 / visible printed page 56, right column, and
  physical PDF page 64 / visible printed page 57, left column.
- Primary review images: the authorized 300-dpi renders.
- Small-mark review: targeted 1200-dpi crops for below-ties, aspiration,
  small-cap `ɪ`, inverted-breve-below, glottal stop, and ejective apostrophe.
- Every form, group number, bracket code, repetition, and blank was read from
  the rendered source. OCR, PDF text, raw legacy glyphs, installed forms, and
  old audits did not supply or verify any reading.
- Site identities remained neutral printed codes `0`, `a`-`p` until after both
  ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 91-95 | Cumulative 1-95 |
| --- | ---: | ---: |
| Items | 5 | 95 |
| Printed response lines | 44 | 731 |
| Conceptual site cells | 85 | 1,615 |
| Ordinary attested cells | 84 | 1,569 |
| Source-conflict cells | 0 | 1 |
| Cells containing an attestation | 84 | 1,570 |
| Blank-only cells | 1 | 45 |
| Printed no-entry coordinates | 1 | 46 |
| Not-used cells | 0 | 0 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved source coordinates | 0 | 1 |
| Expanded attested response occurrences | 89 | 1,685 |

The sole block blank is item 91/site `o`, printed as group-0 `no entry`.
Overlapping or repeated assignments are retained at item 93/sites `n,m` and
item 95/sites `g,h,n`. There are no block-local not-used cells, ambiguities,
illegibles, source conflicts, or unresolved source coordinates. Cumulatively,
item 12/site `p` remains unresolved.

Frozen SHA-256 values:

- generator: `549ebe13fb435f01b52c906cd9f4ddb986f40da7a71f1cdbdac945d1b9acfc20`
- line ledger: `07891cfdd77c3b8dd795794aa960e4eb0e84c5bd4648ca52163b6738ef142612`
- cell ledger: `6df79246e57c34fca7c9b453f59d78afb26a54346306ab636011898b656b5c12`

## Transcription decisions

- Item 91 preserves small-cap `ɪ`, aspiration, glottal stop, and the
  inverted-breve-below in `tau̯` and `dau̯`.
- Item 92 preserves the below-tie in `d͜ʒoŋ` and `d͜ʒoŋʔʃu`, plus aspiration
  and the inverted-breve-below in `kʰnia̯ŋ`.
- Item 93 retains the independently printed repeated `sɛʔlou̯` at site `n`
  and the overlapping site-`m` assignments. Glottal stop, aspiration,
  ejective `ʼ`, and every inverted-breve-below are preserved.
- Item 94 keeps the source parenthetical string `nija (tʃoŋ)` distinct from
  `nijatʃoŋ`, and preserves the below-tie, aspiration, and vowel sequences.
- Item 95 retains the separately printed repeated group-1/group-3 responses at
  sites `g,h,n`, and preserves aspiration, glottal stop, and all combining marks.

## Bracket expansion and reconciliation

The 44-line ledger was frozen first. Mechanical bracket expansion produced 90
line-site records: 89 attestations and one no-entry record. The conceptual
ledger contains exactly 85 cells because five attested coordinates have a
second printed assignment.

Only after both ledgers were frozen was the legacy audit opened. The comparison
contains 90 rows:

- 67 legacy-installed records and 23 legacy exclusions;
- 22 independently recovered attestations corresponding to excluded glyphs;
- one exclusion corresponding to the printed no-entry coordinate;
- 25 exact codepoint matches among legacy-installed records;
- 42 codepoint differences among legacy-installed records;
- 23 codepoint differences among legacy-excluded records.

Reconciliation SHA-256:
`38ce551b57b849b7d0c14cf6e290bc611e41850f3c2a6a95e3337db036b09884`.
Comparison results are audit metadata only and neither verify nor alter manual
readings.

## Deferred gates and remaining work

Items 96-307 remain unreviewed. The item 12/site `p` source contradiction remains
unresolved. Site/lect reconciliation, staging, sound-profile conversion,
bibliography/reference validation, shared integration, full build, graph
validation, and browser QA are deferred. No shared output was changed.
