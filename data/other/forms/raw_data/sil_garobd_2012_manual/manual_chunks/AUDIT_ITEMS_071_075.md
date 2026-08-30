# Manual audit - ESR 2012-007 items 71-75

## Independence and source

- Source: physical PDF pages 60-61 / visible printed pages 53-54.
- Primary review images: the authorized 300-dpi renders.
- Small-mark review: targeted 1200-dpi crops for below-ties, aspiration,
  barred `ɨ`, small-cap `ɪ`, inverted-breve-below, glottal stop, ejective
  apostrophe, and `ɸ`.
- Every form, group number, bracket code, repetition, and blank was read from
  the rendered source. OCR, PDF text, raw legacy glyphs, installed forms, and
  old audits did not supply or verify any reading.
- Site identities remained neutral printed codes `0`, `a`-`p` until after both
  ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 71-75 | Cumulative 1-75 |
| --- | ---: | ---: |
| Items | 5 | 75 |
| Printed response lines | 56 | 574 |
| Conceptual site cells | 85 | 1,275 |
| Ordinary attested cells | 78 | 1,232 |
| Source-conflict cells | 0 | 1 |
| Cells containing an attestation | 78 | 1,233 |
| Blank-only cells | 7 | 42 |
| Printed no-entry coordinates | 7 | 43 |
| Not-used cells | 0 | 0 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved source coordinates | 0 | 1 |
| Expanded attested response occurrences | 91 | 1,328 |

The block blanks are item 71/site `f`, item 72/sites `l,o,p`, and item 75/sites
`l,o,p`, all printed as group-0 `no entry`. Every other cell is attested. Item
72/site `j` carries two distinct printed assignments, while item 73 preserves
all group-1/group-3 and group-2/group-6 repetitions plus the overlapping
group-1 assignments at site `i`. There are no block-local ambiguities,
illegibles, source conflicts, or unresolved source coordinates. Cumulatively,
item 12/site `p` remains unresolved.

Frozen SHA-256 values:

- generator: `40585fd529b19f5c9701c0499a735c32baa36ebba41b49d1a3fbb125a32af47f`
- line ledger: `1e2cd3d7b78ef0b3ae75dcd66a9c2ff37ab36fc1fa18a4232248f511aba24d07`
- cell ledger: `70bcde80a33a1e42596edc23ada60acb4aa1af5143770f3caf18aa871f58a0bf`

## Transcription decisions

- Item 71 preserves inverted-breve-below in `kao̯ i`, barred `ɨ`, glottal
  stop, and the spacing contrast between `tʃɨ riʔ` and `tʃrɨʔ`.
- Item 72 keeps the two separately printed `kʰorgoʃ` lines, the visibly
  printed below-tie in `hed͜ʒabari`, and the `ɛ`/`ɨ` distinctions.
- Item 73 retains every repeated line and site assignment. The source contrasts
  `tʃɨpʼbu`, `tʃɨpʼpʰu`, `tʃɨpʼpu`, `tʃʰɨpʼpʰu`, `tʃupʼbu`, and `tʃupʼpu`,
  and preserves `ɸ`, inverted-breve-below, and aspiration independently.
- Item 74 preserves the `ɨ`/`ɛ`/`ɪ`/`i` contrasts and all aspirated stops.
- Item 75 retains `ɨ` versus `ɪ`, aspiration, ejective `ʼ`, and the
  inverted-breve-below in `malɛŋkʰao̯` without normalization.

## Bracket expansion and reconciliation

The 56-line ledger was frozen first. Mechanical bracket expansion produced 98
line-site records: 91 attestations and seven no-entry records. The conceptual
ledger contains exactly 85 cells because of the source's repeated and overlapping
assignments in items 72 and 73.

Only after both ledgers were frozen was the legacy audit opened. It contains 97
matching records; the manually clear item 72/site `j`/group 1 response
`kʰorgoʃ` is absent from the legacy audit. The 98-row comparison therefore has:

- 79 legacy-installed records, 18 legacy exclusions, and one missing legacy record;
- 11 independently recovered attestations corresponding to excluded glyphs;
- seven exclusions corresponding to printed no-entry coordinates;
- 18 exact codepoint matches among legacy-installed records;
- 61 codepoint differences among legacy-installed records;
- 18 codepoint differences among legacy-excluded records;
- one manual source record absent from the legacy audit at item 72/site `j`/group 1.

Reconciliation SHA-256:
`08ca85d3e1151deae69367537e73be0679f074a04b281d37f55570e2c631f7f9`.
The missing legacy row is a reconciliation discrepancy, not an unresolved source
reading. Comparison results are audit metadata only and neither verify nor alter
manual readings.

## Deferred gates and remaining work

Items 76-307 remain unreviewed. The item 12/site `p` source contradiction remains
unresolved. Site/lect reconciliation, staging, sound-profile conversion,
bibliography/reference validation, shared integration, full build, graph
validation, and browser QA are deferred. No shared output was changed.
