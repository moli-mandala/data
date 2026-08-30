# Manual audit - ESR 2012-007 items 41-45

## Independence and source

- Source: physical PDF page 57 / visible printed page 50, left column.
- Primary review image: the authorized 300-dpi render.
- Small-mark review: targeted 1200-dpi crops for non-syllabic marks,
  aspiration, barred `ɨ`, ejective apostrophe, schwa, and retroflex `ɖ`.
- Every form, group number, bracket code, and blank was read from the rendered
  source. OCR, PDF text, raw legacy glyphs, installed forms, and old audits did
  not supply or verify any reading.
- Site identities remained neutral printed codes `0`, `a`-`p` until after both
  ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 41-45 | Cumulative 1-45 |
| --- | ---: | ---: |
| Items | 5 | 45 |
| Printed response lines | 28 | 326 |
| Conceptual site cells | 85 | 765 |
| Ordinary attested cells | 83 | 738 |
| Source-conflict cells | 0 | 1 |
| Cells containing an attestation | 83 | 739 |
| Blank-only cells | 2 | 26 |
| Printed no-entry coordinates | 2 | 27 |
| Not-used cells | 0 | 0 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved coordinates | 0 | 1 |
| Expanded attested response occurrences | 84 | 782 |

The block blanks are item 41/site `p` and item 42/site `p`, both printed as
group-0 `no entry`. Every other cell is attested. There are no block-local
ambiguities, illegibles, source conflicts, or unresolved coordinates.
Cumulatively, item 12/site `p` remains unresolved.

Frozen SHA-256 values:

- line ledger: `eee5522568ef0a005750f09f7889f03a65e01cce74ac8921edf0924691fc23e3`
- cell ledger: `9f0aa56406f35534200d20884d97f634abd12479af108bace87d9940dc431e88`

## Transcription decisions

- Item 45/site `b` retains identical `lai̯tʃak` responses in groups 2 and 4,
  giving 84 attestations across 83 attested conceptual cells.
- The inverted-breve-below mark remains in `bai̯gon`, `bantao̯`, `mantao̯`,
  `ə dia̯ŋ`, and `lai̯tʃak`; the leading schwa and space in `ə dia̯ŋ` are
  preserved as printed.
- Barred `ɨ` is retained in `barɨŋ` and `rɨka`; retroflex `ɖ` is retained in
  `ɖal`; aspiration remains in `pʰaŋ`, `panpʰaŋ`, and final `gatʃʰ`.
- Ordinary Latin `g` remains in `bai̯gon`, `bɛgun`, and `gatʃʰ`. The ejective
  apostrophe remains in `bidʒakʼ`.

## Bracket expansion and reconciliation

The 28-line ledger was frozen first. Mechanical bracket expansion produced 86
line-site records: 84 attestations and two no-entry records. The conceptual
ledger contains exactly 85 cells because of item 45/site `b`'s repetition.

Only after both ledgers were frozen was the legacy audit opened. The comparison
contains 86 rows:

- 69 legacy-installed records and 17 legacy exclusions;
- 15 independently recovered attestations corresponding to excluded glyphs;
- two exclusions corresponding to printed no-entry coordinates;
- 50 exact codepoint matches among legacy-installed records;
- 19 codepoint differences among legacy-installed records;
- 17 codepoint differences among legacy-excluded records.

Comparison results are audit metadata only and neither verify nor alter manual
readings.

## Deferred gates and remaining work

Items 46-307 remain unreviewed. The item 12/site `p` source contradiction remains
unresolved. Site/lect reconciliation, staging, sound-profile conversion,
bibliography/reference validation, shared integration, full build, graph
validation, and browser QA are deferred. No shared output was changed.
