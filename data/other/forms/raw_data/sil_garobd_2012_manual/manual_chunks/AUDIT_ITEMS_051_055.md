# Manual audit - ESR 2012-007 items 51-55

## Independence and source

- Source: physical PDF page 58 / visible printed page 51, both columns.
- Primary review image: the authorized 300-dpi render.
- Small-mark review: targeted 1200-dpi crops for aspiration, glottal stop,
  inverted-breve-below, barred `ɨ`, nasalization, schwa, and ejective apostrophe.
- Every form, group number, bracket code, and blank was read from the rendered
  source. OCR, PDF text, raw legacy glyphs, installed forms, and old audits did
  not supply or verify any reading.
- Site identities remained neutral printed codes `0`, `a`-`p` until after both
  ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 51-55 | Cumulative 1-55 |
| --- | ---: | ---: |
| Items | 5 | 55 |
| Printed response lines | 48 | 410 |
| Conceptual site cells | 85 | 935 |
| Ordinary attested cells | 82 | 905 |
| Source-conflict cells | 0 | 1 |
| Cells containing an attestation | 82 | 906 |
| Blank-only cells | 3 | 29 |
| Printed no-entry coordinates | 3 | 30 |
| Not-used cells | 0 | 0 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved coordinates | 0 | 1 |
| Expanded attested response occurrences | 90 | 960 |

The block blanks are item 51/site `p`, item 54/site `l`, and item 55/site `b`,
all printed as group-0 `no entry`. Every other cell is attested. There are no
block-local ambiguities, illegibles, source conflicts, or unresolved
coordinates. Cumulatively, item 12/site `p` remains unresolved.

Frozen SHA-256 values:

- line ledger: `4342065d879483cc2e6197f14122c2cfe94638f8db1252835e413e01e3263e89`
- cell ledger: `5158ca7583f94121b11c0d98427a7e7b4f7b0b9f6d23beb8c0fda94e05c0ceec`

## Transcription decisions

- Item 55 retains all repeated assignments: site `d` in groups 2 and 3; site
  `o` in groups 1 and 6; site `h` in groups 3 and A; sites `e,n` in groups 4
  and A; and four site-`f` occurrences across groups 2, 4, 4, and A.
- Barred `ɨ` remains in the item 52 `rɨk` form, both `sɨntɨ` forms, and the
  item 55 forms `bɨtʃʰri`, `pɨtʃʰi`, `bɨtʃʰɨri`, and `tʃɨlɨi̯`.
- Inverted-breve-below remains in the item 52 `ai̯` forms, `suʔpia̯ŋ`,
  `sɨntɨu̯`, `tʃɨlɨi̯`, and `kʰut lia̯ŋ`. The source space in `kʰut lia̯ŋ`
  is preserved.
- Glottal `ʔ`, aspiration, and ejective `ʼ` remain independently represented,
  including the contrast between item 53 `gatʃʰu` and unaspirated `gatʃu`.

## Bracket expansion and reconciliation

The 48-line ledger was frozen first. Mechanical bracket expansion produced 93
line-site records: 90 attestations and three no-entry records. The conceptual
ledger contains exactly 85 cells because of item 55's repeated assignments.

Only after both ledgers were frozen was the legacy audit opened. The comparison
contains 93 rows:

- 74 legacy-installed records and 19 legacy exclusions;
- 16 independently recovered attestations corresponding to excluded glyphs;
- three exclusions corresponding to printed no-entry coordinates;
- 24 exact codepoint matches among legacy-installed records;
- 50 codepoint differences among legacy-installed records;
- 19 codepoint differences among legacy-excluded records.

Comparison results are audit metadata only and neither verify nor alter manual
readings.

## Deferred gates and remaining work

Items 56-307 remain unreviewed. The item 12/site `p` source contradiction remains
unresolved. Site/lect reconciliation, staging, sound-profile conversion,
bibliography/reference validation, shared integration, full build, graph
validation, and browser QA are deferred. No shared output was changed.
