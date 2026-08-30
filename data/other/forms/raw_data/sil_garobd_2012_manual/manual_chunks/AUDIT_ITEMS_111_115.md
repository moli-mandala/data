# Manual audit - ESR 2012-007 items 111-115

## Independence and source

- Sources: physical PDF page 66 / visible printed page 59, right column, and
  physical PDF page 67 / visible printed page 60, left column.
- Primary review images: the authorized 300-dpi renders.
- Small-mark review: targeted 1200-dpi crops for below-ties, aspiration,
  barred `ɨ`, small-cap `ɪ`, inverted-breve-below, and ejective apostrophe.
- Every form, group number, bracket code, repetition, and blank was read from
  the rendered source. OCR, PDF text, raw legacy glyphs, installed forms, and
  old audits did not supply or verify any reading.
- Site identities remained neutral printed codes `0`, `a`-`p` until after both
  ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 111-115 | Cumulative 1-115 |
| --- | ---: | ---: |
| Items | 5 | 115 |
| Printed response lines | 44 | 906 |
| Conceptual site cells | 85 | 1,955 |
| Ordinary attested cells | 84 | 1,905 |
| Source-conflict cells | 0 | 1 |
| Cells containing an attestation | 84 | 1,906 |
| Blank-only cells | 1 | 49 |
| Printed no-entry coordinates | 1 | 50 |
| Not-used cells | 0 | 0 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved source coordinates | 0 | 1 |
| Expanded attested response occurrences | 91 | 2,054 |

The sole block blank is item 111/site `p`, printed as group-0 `no entry`.
Item 113 retains identical group-1/group-2 `d͜ʒakʼpʰa` assignments at sites
`a,d,e,g,h,i,o`. There are no block-local not-used cells, ambiguities,
illegibles, source conflicts, or unresolved source coordinates. Cumulatively,
item 12/site `p` remains unresolved.

Frozen SHA-256 values:

- generator: `1e552cc2b05f0b962fa4bfb068f92a677e31e1d54ad9252b0d5a5119b699cf0a`
- line ledger: `d286f91ea25b411c0d9c50803375df9b87d5902404989a0fdd9da9e5d23a61c3`
- cell ledger: `3dd71532cbb08a693ef1ba13f2f97e4c2af82a4e96573805e7cf536e88efbcd2`

## Transcription decisions

- Item 111 preserves below-tied `d͜ʒ`, aspiration, ejective `ʼ`, source spaces,
  and inverted-breve-below in the three `gilai̯`/`toŋkai̯` forms.
- Item 112 keeps the distinct printed `d͜ʒ`, `tʃ`, and below-tied `t͜ʒ`
  onsets, plus aspiration, barred `ɨ`, ejectives, and inverted-breve-below.
- Item 113 retains the seven repeated group-1/group-2 assignments and preserves
  both below-tied affricates, aspiration, barred `ɨ`, small-cap `ɪ`, ejectives,
  and source spaces.
- Item 114 preserves below-tied `d͜ʒ`, ejectives, barred `ɨ`, and both
  inverted-breve-below sequences.
- Item 115 preserves every aspiration, barred `ɨ`, ejective, and affricate
  distinction across the fingernail forms.

## Bracket expansion and reconciliation

The 44-line ledger was frozen first. Mechanical bracket expansion produced 92
line-site records: 91 attestations and one no-entry record. The conceptual
ledger contains exactly 85 cells because seven item-113 coordinates have a
second printed assignment.

Only after both ledgers were frozen was the legacy audit opened. The comparison
contains 92 rows:

- 79 legacy-installed records and 13 legacy exclusions;
- 12 independently recovered attestations corresponding to excluded glyphs;
- one exclusion corresponding to the printed no-entry coordinate;
- eight exact codepoint matches among legacy-installed records;
- 71 codepoint differences among legacy-installed records;
- 13 codepoint differences among legacy-excluded records.

Reconciliation SHA-256:
`304f00b1ce473398e9a06ffd77d1bac03a7bcd34185cf37c5a99746204ebabaf`.
Comparison results are audit metadata only and neither verify nor alter manual
readings.

## Deferred gates and remaining work

Items 116-307 remain unreviewed. The item 12/site `p` source contradiction remains
unresolved. Site/lect reconciliation, staging, sound-profile conversion,
bibliography/reference validation, shared integration, full build, graph
validation, and browser QA are deferred. No shared output was changed.
