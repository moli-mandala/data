# Manual audit - ESR 2012-007 items 106-110

## Independence and source

- Source: physical PDF page 66 / visible printed page 59, both columns.
- Primary review image: the authorized 300-dpi render.
- Small-mark review: targeted 1200-dpi crops for aspiration, barred `ɨ`,
  small-cap `ɪ`, inverted-breve-below, nasalization, glottal stop, and ejective
  apostrophe.
- Every form, group number, bracket code, repetition, and blank was read from
  the rendered source. OCR, PDF text, raw legacy glyphs, installed forms, and
  old audits did not supply or verify any reading.
- Site identities remained neutral printed codes `0`, `a`-`p` until after both
  ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 106-110 | Cumulative 1-110 |
| --- | ---: | ---: |
| Items | 5 | 110 |
| Printed response lines | 48 | 862 |
| Conceptual site cells | 85 | 1,870 |
| Ordinary attested cells | 84 | 1,821 |
| Source-conflict cells | 0 | 1 |
| Cells containing an attestation | 84 | 1,822 |
| Blank-only cells | 1 | 48 |
| Printed no-entry coordinates | 1 | 49 |
| Not-used cells | 0 | 0 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved source coordinates | 0 | 1 |
| Expanded attested response occurrences | 99 | 1,963 |

The sole block blank is item 106/site `p`, printed as group-0 `no entry`.
Repeated assignments are retained at item 106/sites `e,f,a,i,o`, item 107/sites
`a,i`, item 108/sites `l,m`, and item 109/sites `e,h,g,n`. There are no
block-local not-used cells, ambiguities, illegibles, source conflicts, or
unresolved source coordinates. Cumulatively, item 12/site `p` remains unresolved.

Frozen SHA-256 values:

- generator: `e7da78d07818ce45d766c0466661498d294202ee89bfe048022c93e456d87778`
- line ledger: `d921478d8ced98de6af4aa0513931e2416e0702a365b05ae1fb4ab962ad05b13`
- cell ledger: `e4895ceae4dea93311cd6a3692da92f3f9a8eefee5061645081521848ad2de10`

## Transcription decisions

- Item 106 preserves all three independently printed `pʰai̯tʰopʼ`
  assignments at sites `e,f`, both `pʰitʰopʼ` assignments at sites `a,i,o`,
  and every aspiration, ejective, small-cap `ɪ`, barred `ɨ`, and
  inverted-breve-below mark.
- Item 107 keeps repeated `kʰudumbok` at sites `a,i`, and preserves glottal
  stop, aspiration, barred `ɨ`, and inverted-breve-below.
- Item 108 keeps repeated `kʰutʃukʼ` at sites `l,m` and preserves aspiration
  and ejective `ʼ` independently.
- Item 109 retains four repeated group-3/group-4 assignments, all ejectives,
  barred `ɨ`, small-cap `ɪ`, and inverted-breve-below. Control form `dʒɪb`
  remains an untied `dʒ` sequence because no tie is printed.
- Item 110 preserves the inverted-breve-below in `moi̯n` and nasalization in
  `dãt`.

## Bracket expansion and reconciliation

The 48-line ledger was frozen first. Mechanical bracket expansion produced 100
line-site records: 99 attestations and one no-entry record. The conceptual
ledger contains exactly 85 cells because 15 attested coordinates have an
additional printed assignment.

Only after both ledgers were frozen was the legacy audit opened. The comparison
contains 100 rows:

- 76 legacy-installed records and 24 legacy exclusions;
- 23 independently recovered attestations corresponding to excluded glyphs;
- one exclusion corresponding to the printed no-entry coordinate;
- 15 exact codepoint matches among legacy-installed records;
- 61 codepoint differences among legacy-installed records;
- 24 codepoint differences among legacy-excluded records.

Reconciliation SHA-256:
`7a77a951489761fe2eb058f208456ef1c061d0f57b306ecfff4031d6a7a68a72`.
Comparison results are audit metadata only and neither verify nor alter manual
readings.

## Deferred gates and remaining work

Items 111-307 remain unreviewed. The item 12/site `p` source contradiction remains
unresolved. Site/lect reconciliation, staging, sound-profile conversion,
bibliography/reference validation, shared integration, full build, graph
validation, and browser QA are deferred. No shared output was changed.
