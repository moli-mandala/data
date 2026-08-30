# Manual audit - ESR 2012-007 items 86-90

## Independence and source

- Source: physical PDF page 63 / visible printed page 56, both columns.
- Primary review image: the authorized 300-dpi render.
- Small-mark review: targeted 1200-dpi crops for below-ties, dental diacritic,
  aspiration, small-cap `ɪ`, barred `ɨ`, inverted-breve-below, glottal stop,
  and ejective apostrophe.
- Every form, group number, bracket code, repetition, and blank was read from
  the rendered source. OCR, PDF text, raw legacy glyphs, installed forms, and
  old audits did not supply or verify any reading.
- Site identities remained neutral printed codes `0`, `a`-`p` until after both
  ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 86-90 | Cumulative 1-90 |
| --- | ---: | ---: |
| Items | 5 | 90 |
| Printed response lines | 39 | 687 |
| Conceptual site cells | 85 | 1,530 |
| Ordinary attested cells | 84 | 1,485 |
| Source-conflict cells | 0 | 1 |
| Cells containing an attestation | 84 | 1,486 |
| Blank-only cells | 1 | 44 |
| Printed no-entry coordinates | 1 | 45 |
| Not-used cells | 0 | 0 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved source coordinates | 0 | 1 |
| Expanded attested response occurrences | 85 | 1,596 |

The sole block blank is item 88/site `p`, printed as group-0 `no entry`. Item
90/site `o` retains its two independently printed assignments. Every other
attested cell has one response. There are no block-local not-used cells,
ambiguities, illegibles, source conflicts, or unresolved source coordinates.
Cumulatively, item 12/site `p` remains unresolved.

Frozen SHA-256 values:

- generator: `99edb269a972c566ea79ed9b723231f7310a834c9ac69bc3c0903cadf83ba3d4`
- line ledger: `6c8102eca4bc1fa0562f10d9ebea6db9e48faa4671ee0c87d453287922951841`
- cell ledger: `c0b9e9c712d8accfc0ebc4fd927ce0c6a0ab1743464199512784b32d0cf3b1de`

## Transcription decisions

- Item 86 preserves all inverted-breve-below marks, the below-tie in
  `mid͜ʒutʼ`, aspiration, glottal stop, and ejective `ʼ`.
- Item 87 keeps `sɨʔer`, `siʔɛr`, and `siɛr` distinct, and preserves the
  inverted-breve-below in `dou̯`, `dau̯`, and `tau̯`.
- Item 88 retains small-cap `ɪ` in `pitɪk`, barred `ɨ` in `bɨtʼtʃi` and
  `tɨi̯`, aspiration, and the dental diacritic in control response `d̪im`.
- Item 89 preserves aspiration and ejective `ʼ` independently in `naʔtʰokʼ`,
  `kʰa`, and `matʃʰ`.
- Item 90 retains both site-`o` forms (`gagakʼ | dogɛpʼ`) and all vowel,
  inverted-breve-below, glottal-stop, and ejective contrasts.

## Bracket expansion and reconciliation

The 39-line ledger was frozen first. Mechanical bracket expansion produced 86
line-site records: 85 attestations and one no-entry record. The conceptual
ledger contains exactly 85 cells because item 90/site `o` has two attestations.

Only after both ledgers were frozen was the legacy audit opened. The comparison
contains 86 rows:

- 58 legacy-installed records and 28 legacy exclusions;
- 27 independently recovered attestations corresponding to excluded glyphs;
- one exclusion corresponding to the printed no-entry coordinate;
- 19 exact codepoint matches among legacy-installed records;
- 39 codepoint differences among legacy-installed records;
- 28 codepoint differences among legacy-excluded records.

Reconciliation SHA-256:
`2993b4738562226847259127f330e65f98c8fb863a5aadcfcd417051effe54d1`.
Comparison results are audit metadata only and neither verify nor alter manual
readings.

## Deferred gates and remaining work

Items 91-307 remain unreviewed. The item 12/site `p` source contradiction remains
unresolved. Site/lect reconciliation, staging, sound-profile conversion,
bibliography/reference validation, shared integration, full build, graph
validation, and browser QA are deferred. No shared output was changed.
