# Manual audit - ESR 2012-007 items 81-85

## Independence and source

- Source: physical PDF pages 62-63 / visible printed pages 55-56.
- Primary review images: the authorized 300-dpi renders.
- Small-mark review: targeted 1200-dpi crops for below-ties, aspiration,
  palatalization, barred `ɨ`, inverted-breve-below, glottal stop, and ejective
  apostrophe.
- Every form, group number, bracket code, repetition, and blank was read from
  the rendered source. OCR, PDF text, raw legacy glyphs, installed forms, and
  old audits did not supply or verify any reading.
- Site identities remained neutral printed codes `0`, `a`-`p` until after both
  ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 81-85 | Cumulative 1-85 |
| --- | ---: | ---: |
| Items | 5 | 85 |
| Printed response lines | 34 | 648 |
| Conceptual site cells | 85 | 1,445 |
| Ordinary attested cells | 84 | 1,401 |
| Source-conflict cells | 0 | 1 |
| Cells containing an attestation | 84 | 1,402 |
| Blank-only cells | 1 | 43 |
| Printed no-entry coordinates | 1 | 44 |
| Not-used cells | 0 | 0 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved source coordinates | 0 | 1 |
| Expanded attested response occurrences | 86 | 1,511 |

The sole block blank is item 82/site `p`, printed as group-0 `no entry`.
Item 83 retains the group-1/group-2 repetitions at sites `l,m`; every other
block cell is singly attested. There are no block-local not-used cells,
ambiguities, illegibles, source conflicts, or unresolved source coordinates.
Cumulatively, item 12/site `p` remains unresolved.

Frozen SHA-256 values:

- generator: `9dbf2a332f1341f1b0f22b536ede39c881f5f20e92eec0a9a479f8e6c66d4aaa`
- line ledger: `5d14d168343fac96738a569aecfb6fdbd8fd7e62de550294a7a416417e351b02`
- cell ledger: `806497e9feb387435869521ae77e7e638a091dcd0494c4ebfad0bd753bae7158`

## Transcription decisions

- Item 81 preserves `moi̯ʃi`, `moʃi`, `muɨʃi`, and `muʃi` as distinct
  printed responses, together with the `ɨ`/`ɛ` contrasts in group 2.
- Item 82 retains `goroŋ`, `groŋ`, `koroŋ`, `rɨŋ`, and `ʃiŋ` independently.
- Item 83 preserves the repeated `diʔmi` assignments, inverted-breve-below in
  `kʰiʔmai̯` and `dimai̯`, and the visibly printed below-tie in `lɛd͜ʒ`.
- Item 84 keeps aspiration, barred `ɨ`, glottal stop, and ejective `ʼ`
  independent across the seven source lines.
- Item 85 preserves the palatalization in `snʲaŋ`, plus `wakʼ` and control
  response `ʃukor` without normalization.

## Bracket expansion and reconciliation

The 34-line ledger was frozen first. Mechanical bracket expansion produced 87
line-site records: 86 attestations and one no-entry record. The conceptual
ledger contains exactly 85 cells because item 83/sites `l,m` repeat.

Only after both ledgers were frozen was the legacy audit opened. The comparison
contains 87 rows:

- 78 legacy-installed records and nine legacy exclusions;
- eight independently recovered attestations corresponding to excluded glyphs;
- one exclusion corresponding to the printed no-entry coordinate;
- 22 exact codepoint matches among legacy-installed records;
- 56 codepoint differences among legacy-installed records;
- nine codepoint differences among legacy-excluded records.

Reconciliation SHA-256:
`3169cb391feb4d61211a8ba16c0cb6909d299fb9748b7be51c5ea39a842525a5`.
Comparison results are audit metadata only and neither verify nor alter manual
readings.

## Deferred gates and remaining work

Items 86-307 remain unreviewed. The item 12/site `p` source contradiction remains
unresolved. Site/lect reconciliation, staging, sound-profile conversion,
bibliography/reference validation, shared integration, full build, graph
validation, and browser QA are deferred. No shared output was changed.
