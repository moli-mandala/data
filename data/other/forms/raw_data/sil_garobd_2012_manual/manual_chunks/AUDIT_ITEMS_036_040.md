# Manual audit - ESR 2012-007 items 36-40

## Independence and source

- Source: physical PDF page 56 / visible printed page 49, right column.
- Primary review image: the authorized 300-dpi render.
- Small-mark review: targeted 1200-dpi crops for non-syllabic marks,
  aspiration, ejective apostrophe, and ordinary-versus-IPA glyph distinctions.
- Every form, group number, bracket code, and blank was read from the rendered
  source. OCR, PDF text, raw legacy glyphs, installed forms, and old audits did
  not supply or verify any reading.
- Site identities remained neutral printed codes `0`, `a`-`p` until after both
  ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 36-40 | Cumulative 1-40 |
| --- | ---: | ---: |
| Items | 5 | 40 |
| Printed response lines | 23 | 298 |
| Conceptual site cells | 85 | 680 |
| Ordinary attested cells | 82 | 655 |
| Source-conflict cells | 0 | 1 |
| Cells containing an attestation | 82 | 656 |
| Blank-only cells | 3 | 24 |
| Printed no-entry coordinates | 3 | 25 |
| Not-used cells | 0 | 0 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved coordinates | 0 | 1 |
| Expanded attested response occurrences | 85 | 698 |

The block blanks are item 36/site `p`, item 39/site `p`, and item 40/site
`p`, each printed as group-0 `no entry`. All other conceptual cells are
attested. There are no block-local ambiguities, illegibles, source conflicts,
or unresolved coordinates. Cumulatively, item 12/site `p` remains unresolved.

Frozen SHA-256 values:

- line ledger: `47bd07ff2f18e4b5d5980179bf7d5327897ff97e12a6f6e669a7a3a3ec499945`
- cell ledger: `b08479f603dda4896b116da64365cc39a1cd0ddf8794a072b34e7f5db4252b02`

## Transcription decisions

- Item 37/site `f` retains `mai̯kʰop | mai̯ragu` from groups 1 and 2.
  Item 38/site `m` retains `kʰan | alubʰuta`, while site `k` retains
  `pʰan | alu`. These source variants account for 85 response occurrences
  across 82 attested conceptual cells.
- The inverted-breve-below mark is retained in `mai̯kʰop`, `mai̯ragu`,
  `sorkʰao̯`, and `surkʰao̯`.
- Aspiration remains in all visibly marked forms, including `makʰu`,
  `mikʰopʼ`, `bʰutta`, `tʰa bultʃʰu`, `tʰa butʃul`, `kʰan`, `pʰan`,
  `alubʰuta`, `pʰulkopi`, and `badʰakopi`.
- The ejective apostrophe is preserved in `mikʰopʼ`. Spaces in the two item-38
  `tʰa ...` forms are source content. Ordinary Latin `g` remains in `gɔm`,
  `mai̯ragu`, `mɛragu`, and `aluguta`.

## Bracket expansion and reconciliation

The 23-line ledger was frozen first. Mechanical bracket expansion produced 88
line-site records: 85 attestations and three no-entry records. The conceptual
ledger contains exactly 85 cells because of the three repeated-site variants.

Only after both ledgers were frozen was the legacy audit opened. The comparison
contains 88 rows:

- 79 legacy-installed records and nine legacy exclusions;
- six independently recovered attestations corresponding to excluded glyphs;
- three exclusions corresponding to printed no-entry coordinates;
- nine exact codepoint matches among legacy-installed records;
- 70 codepoint differences among legacy-installed records;
- nine codepoint differences among legacy-excluded records.

Comparison results are audit metadata only and neither verify nor alter manual
readings.

## Deferred gates and remaining work

Items 41-307 remain unreviewed. The item 12/site `p` source contradiction remains
unresolved. Site/lect reconciliation, staging, sound-profile conversion,
bibliography/reference validation, shared integration, full build, graph
validation, and browser QA are deferred. No shared output was changed.
