# Manual audit - ESR 2012-007 items 31-35

## Independence and source

- Source: physical PDF page 56 / visible printed page 49.
- Items 31-34 are in the left column; item 35 is in the right column.
- Primary review image: the authorized 300-dpi render.
- Small-mark review: targeted 1200-dpi crops for non-syllabic marks,
  aspiration, barred `ɨ`, ejective apostrophe, and retroflex `ɖ`.
- Every form, group number, and bracket code was read from the rendered source.
  OCR, PDF text, raw legacy glyphs, installed forms, and old audits did not
  supply or verify any reading.
- Site identities remained neutral printed codes `0`, `a`-`p` until after both
  ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 31-35 | Cumulative 1-35 |
| --- | ---: | ---: |
| Items | 5 | 35 |
| Printed response lines | 32 | 275 |
| Conceptual site cells | 85 | 595 |
| Ordinary attested cells | 85 | 573 |
| Source-conflict cells | 0 | 1 |
| Cells containing an attestation | 85 | 574 |
| Blank-only cells | 0 | 21 |
| Printed no-entry coordinates | 0 | 22 |
| Not-used cells | 0 | 0 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved coordinates | 0 | 1 |
| Expanded attested response occurrences | 87 | 613 |

All 85 block cells are attested. There are no no-entry, not-used, ambiguous,
illegible, source-conflict, or unresolved block coordinates. Cumulatively, item
12/site `p` remains unresolved.

Frozen SHA-256 values:

- line ledger: `8056dd177f979822ab3c1bdf7105d076ae75a0c02229e28fd2a3de87609ad514`
- cell ledger: `626147e6e2409c0a897b464e8b6d751df0013b8ddc0e78212f62648cef8a0efe`

## Transcription decisions

- Item 32/site `e` deliberately retains `wala | wala` from groups 1 and 4.
  Item 35/site `m` retains `mai̯ | mai̯mɨn` from groups 2 and 4. These source
  repetitions produce 87 response occurrences across 85 conceptual cells.
- The inverted-breve-below mark is retained in `mai̯`, `mai̯ruŋ`, `mai̯roŋ`,
  `kʰao̯`, and `mai̯mɨn`.
- Retroflex `ɖ` is retained in `ɖʒanmot`, `ɖʒiba`, and `ɖʒa`; aspiration is
  retained in `hantʰam`, `hɛntʰam`, `ʃondʰa`, `pʰarokʼ`, `dʰan`, `kʰao̯`,
  and `bʰat`.
- Ordinary Latin `g` remains in `gasam` and `gasum`. The ejective apostrophe is
  preserved in `motʼ` and `pʰarokʼ`; `wallo` retains its printed double `l`.

## Bracket expansion and reconciliation

The 32-line ledger was frozen first. Mechanical bracket expansion produced 87
attested line-site records. The conceptual ledger contains exactly 85 cells
because of the two repeated-site patterns.

Only after both ledgers were frozen was the legacy audit opened. The comparison
contains 87 rows:

- 65 legacy-installed records and 22 legacy exclusions;
- all 22 exclusions are independently recovered attestations;
- 37 exact codepoint matches among legacy-installed records;
- 28 codepoint differences among legacy-installed records;
- 22 codepoint differences among legacy-excluded records.

Comparison results are audit metadata only and neither verify nor alter manual
readings.

## Deferred gates and remaining work

Items 36-307 remain unreviewed. The item 12/site `p` source contradiction remains
unresolved. Site/lect reconciliation, staging, sound-profile conversion,
bibliography/reference validation, shared integration, full build, graph
validation, and browser QA are deferred. No shared output was changed.
