# Manual audit - ESR 2012-007 items 26-30

## Independence and source

- Source: physical PDF page 55 / visible printed page 48.
- Item 26 is in the left column; items 27-30 are in the right column.
- Primary review image: the authorized 300-dpi render.
- Small-mark review: targeted 1200-dpi crops for non-syllabic marks,
  aspiration, barred `ɨ`, small-cap `ɪ`, ejective apostrophe, and retroflex `ɖ`.
- Every form, group number, bracket code, and status was read from the rendered
  source. OCR, PDF text, raw legacy glyphs, installed forms, and old audits did
  not supply or verify any reading.
- Site identities remained neutral printed codes `0`, `a`-`p` until after both
  ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 26-30 | Cumulative 1-30 |
| --- | ---: | ---: |
| Items | 5 | 30 |
| Printed response lines | 31 | 243 |
| Conceptual site cells | 85 | 510 |
| Ordinary attested cells | 84 | 488 |
| Source-conflict cells | 0 | 1 |
| Cells containing an attestation | 84 | 489 |
| Blank-only cells | 1 | 21 |
| Printed no-entry coordinates | 1 | 22 |
| Not-used cells | 0 | 0 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved coordinates | 0 | 1 |
| Expanded attested response occurrences | 92 | 526 |

The sole block blank is item 30/site `p`, printed as group-0 `no entry`. Every
other conceptual cell is attested. There are no block-local source conflicts,
ambiguities, illegibles, or unresolved coordinates. Cumulatively, item 12/site
`p` remains unresolved.

Frozen SHA-256 values:

- line ledger: `5f33bc5c2c88f2121595993e035d910ef2bfaedfef173e570933736c79604c6e`
- cell ledger: `94d439754d876f919d531e12fa6fc225180b83fd16ee9522b208a7817e999a6f`

## Transcription decisions

- Item 28 deliberately repeats `ʃal [aefgino]` in groups 1 and 2 and
  `sɨŋɨi̯ [k]` in groups 4 and 6. Both printed occurrences are retained, giving
  25 attested response occurrences across its 17 conceptual cells.
- Small-cap `ɪ` is retained in `snɪm` and `dɪn`; barred `ɨ` is retained in
  `bɨlʃi`, `bɨlsi`, `sɨŋɨi̯`, `sɨŋ ŋei̯`, `pʰrɨŋ`, `sɨnsi`, and
  `bri pɨndɨŋ`.
- The inverted-breve-below mark is retained in `sɨŋɨi̯` and `sɨŋ ŋei̯`.
- Retroflex `ɖ` is retained in `ɖʒa` and the four item-30 forms containing
  `ɖʒ`; aspiration is retained in `bɔtʃʰor`, `pʰrɨŋ`, `pʰruŋ`, and
  `ʃalɖʒatʼtʰi`.
- The ejective apostrophe remains in `manatʼ`, `ʃalɖʒatʼtʃi`, and
  `ʃalɖʒatʼtʰi`; the item-30 group-1 affricate/aspiration distinction is not
  normalized away.

## Bracket expansion and reconciliation

The 31-line ledger was frozen first. Mechanical bracket expansion produced 93
line-site records: 92 attested occurrences and one no-entry record. The
conceptual ledger contains exactly 85 cells because of item 28's repetitions.

Only after both ledgers were frozen was the legacy audit opened. The comparison
contains 93 rows:

- 89 legacy-installed records and four legacy exclusions;
- three independently recovered attestations corresponding to excluded glyphs;
- one exclusion corresponding to the printed no-entry coordinate;
- 65 exact codepoint matches among legacy-installed records;
- 24 codepoint differences among legacy-installed records;
- four codepoint differences among legacy-excluded records.

Comparison results are audit metadata only and neither verify nor alter manual
readings.

## Deferred gates and remaining work

Items 31-307 remain unreviewed. The item 12/site `p` source contradiction remains
unresolved. Site/lect reconciliation, staging, sound-profile conversion,
bibliography/reference validation, shared integration, full build, graph
validation, and browser QA are deferred. No shared output was changed.
