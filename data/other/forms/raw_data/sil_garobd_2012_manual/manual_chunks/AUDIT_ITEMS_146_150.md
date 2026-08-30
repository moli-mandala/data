# Manual audit - ESR 2012-007 items 146-150

## Independence and source

- Sources: physical PDF pages 71-72 / visible printed pages 64-65.
- Primary review images: the authorized 300-dpi renders.
- Small-mark review: targeted 1200-dpi crops for barred `ɨ`, `ɪ`, aspiration,
  glottal stop, ejective apostrophe, eng, `ɛ`, `ɔ`, and
  inverted-breve-below sequences.
- Every form, group number, bracket code, repetition, source space, literal
  slash, and explicit blank was read from the rendered source. OCR, PDF text,
  raw legacy glyphs, installed forms, and old audits did not supply or verify
  any reading.
- Site identities remained neutral printed codes `0`, `a`-`p` until after both
  ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 146-150 | Cumulative 1-150 |
| --- | ---: | ---: |
| Items | 5 | 150 |
| Printed response lines | 33 | 1,190 |
| Conceptual site cells | 85 | 2,550 |
| Ordinary attested cells | 83 | 2,488 |
| Source-conflict cells | 0 | 1 |
| Cells containing an attestation | 83 | 2,489 |
| Blank-only cells | 2 | 61 |
| Printed no-entry coordinates | 2 | 62 |
| Not-used cells | 0 | 0 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved source coordinates | 0 | 1 |
| Expanded attested response occurrences | 87 | 2,656 |

Items 146 and 149/site `p` are explicit `no entry` coordinates. Item 146/sites
`l,m` retain identical group-1/group-5 `nukʰuŋ` assignments, and item 146/site
`c` retains identical group-2/group-5 `nukʰuraŋ` assignments. There are no
block-local not-used cells, ambiguities, illegibles, source conflicts, or
unresolved source coordinates. Cumulatively, item 12/site `p` remains
unresolved.

Frozen SHA-256 values:

- generator: `5c582a1001cbd687b62d40580b349acae0c4519ae2991236a52f602944d0e6a8`
- line ledger: `cf1d8cc378bb0bda8e43c0b1423a790a1a5be72139641a764d8174ecebd21879`
- cell ledger: `1c005924a106851f44ef745f9cfedaab0ce978c6ccf8c51f5a5137fb17d1266f`

## Transcription decisions

- Item 146 preserves both ejective apostrophes in `nokʼkʰɨŋ`, aspiration,
  barred `ɨ`, eng, the literal slash string `tʃʰad / tʃal`, the explicit
  blank, and all repeated assignments.
- Item 147 preserves `ɛ` in `bɛra` and `dɛal`, barred `ɨ`, the glottal stop,
  and the source space in `kɨn ruʔ`.
- Item 148 preserves all aspiration contrasts, `ɪ` and `ʃ` in `balɪʃ`, the
  `balus` / `baluʃ` contrast, and the cluster and glottal stop in
  `kʰonkʰlɪʔ`.
- Item 149 preserves `ɔ` in `kɔmbol` and the explicit site-`p` blank.
- Item 150 preserves eng versus `n`, aspiration, vowel contrasts, the glottal
  stop in `suluʔ`, and the inverted-breve-below sequence in `sulutei̯`.

## Bracket expansion and reconciliation

The 33-line ledger was frozen first. Mechanical bracket expansion produced 89
line-site records: 87 attestations and two printed blanks. The conceptual
ledger contains exactly 85 cells because item 146 has three repeated
coordinates.

Only after both ledgers were frozen was the legacy audit opened. The comparison
contains 89 rows:

- 85 legacy-installed records and four legacy exclusions;
- two independently recovered attestations corresponding to excluded glyphs;
- two manual blanks matching legacy printed-gap records;
- 40 exact codepoint matches among legacy-installed records;
- 45 codepoint differences among legacy-installed records;
- four codepoint differences among legacy-excluded records.

Reconciliation SHA-256:
`ae0b653a07d32b210302c6c72c4d9af2fe6280abd27fb62fd6449589660112e0`.
Comparison results are audit metadata only and neither verify nor alter manual
readings.

## Deferred gates and remaining work

Items 151-307 remain unreviewed. The item 12/site `p` source contradiction
remains unresolved. Site/lect reconciliation, staging, sound-profile
conversion, bibliography/reference validation, shared integration, full
build, graph validation, and browser QA are deferred. No shared output was
changed.
