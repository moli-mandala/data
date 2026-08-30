# Manual audit - ESR 2012-007 items 21-25

## Independence and source

- Sources: physical PDF pages 54-55 / visible printed pages 47-48.
- Items 21-22 are in the right column of physical page 54; items 23-25 are in
  the left column of physical page 55.
- Primary review images: the authorized 300-dpi renders.
- Small-mark review: targeted 1200-dpi crops for non-syllabic marks,
  aspiration, barred `ɨ`, glottal stop, and retroflex `ɖ`.
- Every form, group number, bracket code, and status was read from the rendered
  source. OCR, PDF text, raw legacy glyphs, installed forms, and old audits did
  not supply or verify any reading.
- Site identities remained neutral printed codes `0`, `a`-`p` until after both
  ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 21-25 | Cumulative 1-25 |
| --- | ---: | ---: |
| Items | 5 | 25 |
| Printed response lines | 41 | 212 |
| Conceptual site cells | 85 | 425 |
| Ordinary attested cells | 81 | 404 |
| Source-conflict cells | 0 | 1 |
| Cells containing an attestation | 81 | 405 |
| Blank-only cells | 4 | 20 |
| Printed no-entry coordinates | 4 | 21 |
| Not-used cells | 0 | 0 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved coordinates | 0 | 1 |
| Expanded attested response occurrences | 81 | 434 |

The four block blanks are item 21/site `p`, item 23/site `l`, item 24/site
`l`, and item 25/site `a`, each printed as group-0 `no entry`. Every other cell
is attested. There are no block-local source conflicts, ambiguities, illegibles,
or unresolved coordinates. Cumulatively, item 12/site `p` remains unresolved.

Frozen SHA-256 values:

- line ledger: `6942f1666d7938eafb28b435429357d999aa7a3f947fd00dba27b9692dd21c2b`
- cell ledger: `c9c3d36d60bb925658b5b36de8ccc190f6db92bd4f8c49e478d3b03d8645c1ae`

## Transcription decisions

- The small arc below a vowel is combining inverted breve below. It is retained
  in `tai̯ni`, `midʒao̯`, `mui̯ja`, `monao̯`, `mot nao̯`, `hatai̯`,
  `hatʰai̯`, and `tiu̯`.
- Barred `ɨ` remains distinct from ordinary `i` in `hɨnta`, `mɨja`,
  `ambɨn`, `ambɨnu`, `rɨtip`, and `ɨrtip`.
- Item 24 preserves the visibly printed retroflex `ɖ` in `ɖʒɛlo` and the
  alphanumeric similarity-group label `A` for `ɨrtip [jk]`.
- Ordinary printed Latin `g` is retained in `gotokal / kalke`, `gɛnɛ`, and
  `agamikal`; it is not normalized to IPA `ɡ`.
- Printed slashes and spaces are diplomatic source content, retained in
  `gotokal / kalke` and `mot nao̯`.
- Item 21's `rupa` line expands to exactly the sixteen sites other than `p`.

## Bracket expansion and reconciliation

The 41-line ledger was frozen first. Mechanical bracket expansion produced 85
line-site records: 81 attestations and four no-entry records. The conceptual
ledger also contains exactly 85 cells because no site repeats across source
lines in this block.

Only after both ledgers were frozen was the legacy audit opened. The comparison
contains 85 rows:

- 67 legacy-installed records and 18 legacy exclusions;
- 14 independently recovered attestations corresponding to excluded glyphs;
- four exclusions corresponding to printed no-entry coordinates;
- 45 exact codepoint matches among legacy-installed records;
- 22 codepoint differences among legacy-installed records;
- 18 codepoint differences among legacy-excluded records.

Comparison results are audit metadata only and neither verify nor alter manual
readings.

## Deferred gates and remaining work

Items 26-307 remain unreviewed. The item 12/site `p` source contradiction remains
unresolved. Site/lect reconciliation, staging, sound-profile conversion,
bibliography/reference validation, shared integration, full build, graph
validation, and browser QA are deferred. No shared output was changed.
