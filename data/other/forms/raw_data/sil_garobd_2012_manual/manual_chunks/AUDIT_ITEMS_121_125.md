# Manual audit - ESR 2012-007 items 121-125

## Independence and source

- Source: physical PDF page 68 / visible printed page 61, both columns.
- Primary review image: the authorized 300-dpi render.
- Small-mark review: targeted 1200-dpi crops for aspiration, barred `ɨ`,
  below-tied `d͜ʒ`, glottal stop, ejective apostrophe, and
  inverted-breve-below sequences.
- Every form, group number, bracket code, repetition, and blank was read from
  the rendered source. OCR, PDF text, raw legacy glyphs, installed forms, and
  old audits did not supply or verify any reading.
- Site identities remained neutral printed codes `0`, `a`-`p` until after both
  ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 121-125 | Cumulative 1-125 |
| --- | ---: | ---: |
| Items | 5 | 125 |
| Printed response lines | 42 | 993 |
| Conceptual site cells | 85 | 2,125 |
| Ordinary attested cells | 81 | 2,067 |
| Source-conflict cells | 0 | 1 |
| Cells containing an attestation | 81 | 2,068 |
| Blank-only cells | 4 | 57 |
| Printed no-entry coordinates | 4 | 58 |
| Not-used cells | 0 | 0 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved source coordinates | 0 | 1 |
| Expanded attested response occurrences | 83 | 2,222 |

The block blanks are item 122/sites `b,e` and item 124/sites `b,p`, each
printed as group-0 `no entry`. Item 123/site `m` retains the printed group-2
and group-7 `pipukʼ` repetitions. Item 124/site `l` retains distinct group-1
`pikʰa` and group-3 `d͜ʒaŋgi` assignments. There are no block-local not-used
cells, ambiguities, illegibles, source conflicts, or unresolved source
coordinates. Cumulatively, item 12/site `p` remains unresolved.

Frozen SHA-256 values:

- generator: `965e42560404412013c8716fd3fd3ebbe5e179ccab7e474262752117c725f567`
- line ledger: `daaa170a517e2ddd5d817dca376fbfa6a201d25a25f9294335c00083e58ed168`
- cell ledger: `7e51c2e1b55f96f22276938b19c545d6c7c89f048892bb310417ceb4b88051d7`

## Transcription decisions

- Item 121 preserves the printed glottal stop in `hanʔtʃʰi`, barred `ɨ`,
  aspiration, and the inverted-breve-below sequence in `tʰɨi̯`.
- Item 122 preserves distinct `ruʔutʃia` and `rutʃia̯`, below-tied `d͜ʒ`,
  barred `ɨ`, aspiration, ejectives, and `tuŋgoa̯`.
- Item 123 preserves barred `ɨ`, ejectives, `kʰlao̯`, `lau̯baʔ`, and the
  repeated site-`m` form in two printed groups.
- Item 124 preserves the two no-entry coordinates, aspiration, ejective,
  below-tied `d͜ʒ`, `ridɔi̯`, and the overlapping site-`l` assignments.
- Item 125 preserves below-tied `d͜ʒ`, barred `ɨ`, aspiration, and the
  contrast between `pʰat` and `pʰatʼ`.

## Bracket expansion and reconciliation

The 42-line ledger was frozen first. Mechanical bracket expansion produced 87
line-site records: 83 attestations and four no-entry records. The conceptual
ledger contains exactly 85 cells because item 123/site `m` and item 124/site
`l` each have two printed assignments.

Only after both ledgers were frozen was the legacy audit opened. The comparison
contains 87 rows:

- 73 legacy-installed records and 14 legacy exclusions;
- ten independently recovered attestations corresponding to excluded glyphs;
- four exclusions corresponding to printed no-entry coordinates;
- 11 exact codepoint matches among legacy-installed records;
- 62 codepoint differences among legacy-installed records;
- 14 codepoint differences among legacy-excluded records.

Reconciliation SHA-256:
`4653a44430eb4c5f2d7b76f1d2a2d7dec08eb97d883d63edbc5182d28a0e5f0b`.
Comparison results are audit metadata only and neither verify nor alter manual
readings.

## Deferred gates and remaining work

Items 126-307 remain unreviewed. The item 12/site `p` source contradiction
remains unresolved. Site/lect reconciliation, staging, sound-profile
conversion, bibliography/reference validation, shared integration, full
build, graph validation, and browser QA are deferred. No shared output was
changed.
