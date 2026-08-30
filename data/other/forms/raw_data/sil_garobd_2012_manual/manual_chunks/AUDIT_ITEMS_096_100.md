# Manual audit - ESR 2012-007 items 96-100

## Independence and source

- Sources: physical PDF page 64 / visible printed page 57, right column, and
  physical PDF page 65 / visible printed page 58, left column.
- Primary review images: the authorized 300-dpi renders.
- Small-mark review: targeted 1200-dpi crops for below-ties, palatalization,
  aspiration, barred `ɨ`, small-cap `ɪ`, inverted-breve-below, glottal stop,
  and ejective apostrophe.
- Every form, group number, bracket code, repetition, and blank was read from
  the rendered source. OCR, PDF text, raw legacy glyphs, installed forms, and
  old audits did not supply or verify any reading.
- Site identities remained neutral printed codes `0`, `a`-`p` until after both
  ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 96-100 | Cumulative 1-100 |
| --- | ---: | ---: |
| Items | 5 | 100 |
| Printed response lines | 45 | 776 |
| Conceptual site cells | 85 | 1,700 |
| Ordinary attested cells | 83 | 1,652 |
| Source-conflict cells | 0 | 1 |
| Cells containing an attestation | 83 | 1,653 |
| Blank-only cells | 2 | 47 |
| Printed no-entry coordinates | 2 | 48 |
| Not-used cells | 0 | 0 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved source coordinates | 0 | 1 |
| Expanded attested response occurrences | 88 | 1,773 |

The block blanks are item 96/site `p` and item 100/site `p`, each printed as
group-0 `no entry`. Repeated assignments are retained at item 97/sites
`d,e,l,m` and item 99/site `n`. There are no block-local not-used cells,
ambiguities, illegibles, source conflicts, or unresolved source coordinates.
Cumulatively, item 12/site `p` remains unresolved.

Frozen SHA-256 values:

- generator: `0bccc28eb30fb03d8e82e95857ae93b3fe9a27a7dfd5cdf8018f76cbc9c77dde`
- line ledger: `716e7158f56ba0923c070967786e2cabfe440dc25b5da4403062adf3b211acb1`
- cell ledger: `077c3601cdd952b5651223897e02e4111cf09294ecfb7f7221ccc173ff571c98`

## Transcription decisions

- Item 96 preserves palatalization in `nʲam`, barred `ɨ` in `pokɨda`, and
  ejective `ʼ` in `abrɛkʼ`.
- Item 97 keeps barred `ɨ`, aspiration, and all repeated group assignments;
  `ʃimal`, `ʃomol`, and `samal` each remain repeated where printed.
- Item 98 preserves the below-tie in `d͜ʒɨkai̯ŋ`, barred `ɨ`, and the
  inverted-breve-below sequences in both group-4 forms.
- Item 99 retains identical group-3/group-4 `ʃɛkʰou̯` at site `n`, and
  preserves aspiration, barred `ɨ`, small-cap `ɪ`, glottal stop, and
  inverted-breve-below.
- Item 100 preserves barred `ɨ`, aspiration, and ejective `ʼ` independently
  across the four group-1 forms.

## Bracket expansion and reconciliation

The 45-line ledger was frozen first. Mechanical bracket expansion produced 90
line-site records: 88 attestations and two no-entry records. The conceptual
ledger contains exactly 85 cells because five attested coordinates have a
second printed assignment.

Only after both ledgers were frozen was the legacy audit opened. The comparison
contains 90 rows:

- 81 legacy-installed records and nine legacy exclusions;
- seven independently recovered attestations corresponding to excluded glyphs;
- two exclusions corresponding to the printed no-entry coordinates;
- 27 exact codepoint matches among legacy-installed records;
- 54 codepoint differences among legacy-installed records;
- nine codepoint differences among legacy-excluded records.

Reconciliation SHA-256:
`457a0144b32851b74dff23e18457a8167c1a996dc81edace6e8d53de33ef1f69`.
Comparison results are audit metadata only and neither verify nor alter manual
readings.

## Deferred gates and remaining work

Items 101-307 remain unreviewed. The item 12/site `p` source contradiction remains
unresolved. Site/lect reconciliation, staging, sound-profile conversion,
bibliography/reference validation, shared integration, full build, graph
validation, and browser QA are deferred. No shared output was changed.
