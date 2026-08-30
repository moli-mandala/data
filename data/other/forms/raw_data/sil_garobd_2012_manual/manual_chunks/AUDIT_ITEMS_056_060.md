# Manual audit - ESR 2012-007 items 56-60

## Independence and source

- Source: physical PDF page 59 / visible printed page 52, both columns.
- Primary review image: the authorized 300-dpi render.
- Small-mark review: targeted 1200-dpi crops for aspiration, barred `ɨ`,
  inverted-breve-below, open vowels, and ejective apostrophe.
- Every form, group number, bracket code, and blank was read from the rendered
  source. OCR, PDF text, raw legacy glyphs, installed forms, and old audits did
  not supply or verify any reading.
- Site identities remained neutral printed codes `0`, `a`-`p` until after both
  ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 56-60 | Cumulative 1-60 |
| --- | ---: | ---: |
| Items | 5 | 60 |
| Printed response lines | 38 | 448 |
| Conceptual site cells | 85 | 1,020 |
| Ordinary attested cells | 83 | 988 |
| Source-conflict cells | 0 | 1 |
| Cells containing an attestation | 83 | 989 |
| Blank-only cells | 2 | 31 |
| Printed no-entry coordinates | 2 | 32 |
| Not-used cells | 0 | 0 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved coordinates | 0 | 1 |
| Expanded attested response occurrences | 107 | 1,067 |

The block blanks are item 56/site `l` and item 60/site `p`, both printed as
group-0 `no entry`. Every other cell is attested. There are no block-local
ambiguities, illegibles, source conflicts, or unresolved coordinates.
Cumulatively, item 12/site `p` remains unresolved.

Frozen SHA-256 values:

- line ledger: `fdef82b2af8e98483258f1969f6d9326e35ca8b8055de88f06a6eac0a646d3c2`
- cell ledger: `52602badeea291130e1ca8fea4529429343b70d9499932aa6ddeb45005ca3780`

## Transcription decisions

- Item 56 retains every group repetition: `girɨtʼ`, `gorutʼ`, and `grɨtʼ` in
  groups 1 and 2; `kʰrui̯tʼ` in groups 1 and 5; `golotʼ` in groups 2 and 6;
  and `kʰlui̯t` in groups 5 and 6.
- Item 57 retains group-1/group-5 repetitions for `goja`, `guwa`, and
  `guwai̯`, plus the group-2/group-5 repetition of `gui`.
- Barred `ɨ`, aspiration, inverted-breve-below, and ejective `ʼ` are retained
  independently. Item 59 preserves `tʃɨu̯`, `kia̯t`, and the `ɛ`/`ɔ`
  contrast in `mɛra`, `kɛt`, and `mɔd`.
- Item 58's 15-code `tʃun` line and two-code `tʃunu` line mechanically cover
  all 17 sites without normalization or inferred copying.

## Bracket expansion and reconciliation

The 38-line ledger was frozen first. Mechanical bracket expansion produced 109
line-site records: 107 attestations and two no-entry records. The conceptual
ledger contains exactly 85 cells because of the repetitions in items 56 and 57.

Only after both ledgers were frozen was the legacy audit opened. The comparison
contains 109 rows:

- 92 legacy-installed records and 17 legacy exclusions;
- 15 independently recovered attestations corresponding to excluded glyphs;
- two exclusions corresponding to printed no-entry coordinates;
- 32 exact codepoint matches among legacy-installed records;
- 60 codepoint differences among legacy-installed records;
- 17 codepoint differences among legacy-excluded records.

Comparison results are audit metadata only and neither verify nor alter manual
readings.

## Deferred gates and remaining work

Items 61-307 remain unreviewed. The item 12/site `p` source contradiction remains
unresolved. Site/lect reconciliation, staging, sound-profile conversion,
bibliography/reference validation, shared integration, full build, graph
validation, and browser QA are deferred. No shared output was changed.
