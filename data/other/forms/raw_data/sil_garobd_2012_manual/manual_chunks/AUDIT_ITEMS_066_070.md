# Manual audit - ESR 2012-007 items 66-70

## Independence and source

- Source: physical PDF page 60 / visible printed page 53, both columns.
- Primary review image: the authorized 300-dpi render.
- Small-mark review: targeted 1200-dpi crops for the printed affricate tie,
  aspiration, barred `ɨ`, inverted-breve-below, glottal stop, and ejective
  apostrophe.
- Every form, group number, bracket code, repetition, and blank was read from
  the rendered source. OCR, PDF text, raw legacy glyphs, installed forms, and
  old audits did not supply or verify any reading.
- Site identities remained neutral printed codes `0`, `a`-`p` until after both
  ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 66-70 | Cumulative 1-70 |
| --- | ---: | ---: |
| Items | 5 | 70 |
| Printed response lines | 36 | 518 |
| Conceptual site cells | 85 | 1,190 |
| Ordinary attested cells | 83 | 1,154 |
| Source-conflict cells | 0 | 1 |
| Cells containing an attestation | 83 | 1,155 |
| Blank-only cells | 2 | 35 |
| Printed no-entry coordinates | 2 | 36 |
| Not-used cells | 0 | 0 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved coordinates | 0 | 1 |
| Expanded attested response occurrences | 87 | 1,237 |

The block blanks are item 67/site `p` and item 70/site `a`, both printed as
group-0 `no entry`. Every other cell is attested. Item 66 repeats sites `f,g`
and `a,o` in groups 1 and 2; all four repeated assignments are retained. There
are no block-local ambiguities, illegibles, source conflicts, or unresolved
coordinates. Cumulatively, item 12/site `p` remains unresolved.

Frozen SHA-256 values:

- generator: `fd92798956d6a1e95cb35f0157b9d183e9c2281ea6c032b16241d8188cca7b7f`
- line ledger: `31fa9ef75b740383d8b142cabb4599340036523c0db41679e856081b3de86fbf`
- cell ledger: `d60d6f21861bbd495682396fddeb2c3fb43c1e7fc7ccf6202ed540760f18df16`

## Transcription decisions

- Item 66 preserves the visibly printed below-tie in `d͜ʒ`, barred `ɨ`,
  glottal stop `ʔ`, aspiration, and ejective `ʼ` independently. The identical
  group-1/group-2 assignments of `d͜ʒaʔlukʼ` at sites `f,g` and
  `d͜ʒalɨkʼ` at sites `a,o` remain duplicated rather than collapsed.
- Item 67 retains the inverted-breve-below in `moŋmau̯` and `jao̯ba` and
  the barred vowel in `hatɨ`.
- Item 68 preserves `matʼsa` versus `matʼtʃʰa`, plus aspiration in
  `kʰla` and source-final `bagʰ`.
- Item 69 keeps the `bɨl`/`bul`/`pɨl` contrasts and the independent source
  responses `dɨŋ ŋem`, `dɨŋom`, `baluk`, and `bʰaluk`.
- Item 70 preserves aspiration in `matʼtʃʰok`, its contrast with
  `matʼtʃok`, and the inverted-breve-below in `matʃao̯` and `skao̯`.

## Bracket expansion and reconciliation

The 36-line ledger was frozen first. Mechanical bracket expansion produced 89
line-site records: 87 attestations and two no-entry records. The conceptual
ledger contains exactly 85 cells because item 66 has four repeated assignments.

Only after both ledgers were frozen was the legacy audit opened. The comparison
contains 89 rows:

- 80 legacy-installed records and nine legacy exclusions;
- seven independently recovered attestations corresponding to excluded glyphs;
- two exclusions corresponding to printed no-entry coordinates;
- 12 exact codepoint matches among legacy-installed records;
- 68 codepoint differences among legacy-installed records;
- nine codepoint differences among legacy-excluded records.

Reconciliation SHA-256:
`4d694bed05f66612d387af19111a9bbbf425ac5d226c08dfe93abab1fc581251`.
Comparison results are audit metadata only and neither verify nor alter manual
readings.

## Deferred gates and remaining work

Items 71-307 remain unreviewed. The item 12/site `p` source contradiction remains
unresolved. Site/lect reconciliation, staging, sound-profile conversion,
bibliography/reference validation, shared integration, full build, graph
validation, and browser QA are deferred. No shared output was changed.
