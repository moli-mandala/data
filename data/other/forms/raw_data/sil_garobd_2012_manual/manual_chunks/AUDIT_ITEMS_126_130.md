# Manual audit - ESR 2012-007 items 126-130

## Independence and source

- Source: physical PDF page 69 / visible printed page 62, both columns.
- Primary review image: the authorized 300-dpi render.
- Small-mark review: targeted 1200-dpi crops for aspiration, barred `ɨ`,
  glottal stop, ejective apostrophe, eng, and inverted-breve-below sequences.
- Every form, group number, bracket code, repetition, and blank was read from
  the rendered source. OCR, PDF text, raw legacy glyphs, installed forms, and
  old audits did not supply or verify any reading.
- Site identities remained neutral printed codes `0`, `a`-`p` until after both
  ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 126-130 | Cumulative 1-130 |
| --- | ---: | ---: |
| Items | 5 | 130 |
| Printed response lines | 40 | 1,033 |
| Conceptual site cells | 85 | 2,210 |
| Ordinary attested cells | 84 | 2,151 |
| Source-conflict cells | 0 | 1 |
| Cells containing an attestation | 84 | 2,152 |
| Blank-only cells | 1 | 58 |
| Printed no-entry coordinates | 1 | 59 |
| Not-used cells | 0 | 0 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved source coordinates | 0 | 1 |
| Expanded attested response occurrences | 85 | 2,307 |

The sole block blank is item 126/site `h`, printed as group-0 `no entry`.
Item 126/site `d` retains distinct group-1 `bimaŋ` and group-3 `bɛʔɛn`
assignments. There are no block-local not-used cells, ambiguities, illegibles,
source conflicts, or unresolved source coordinates. Cumulatively, item 12/site
`p` remains unresolved.

Frozen SHA-256 values:

- generator: `4faf755adb8823ee67d958885c2fe29de3585d50bc79e224822afff72e133231`
- line ledger: `b0553472a7d792971c297157e4f2dfec3b9d0552b6ef16f9eb14ec30813da7ba`
- cell ledger: `752d486fa187b7e572d96298753198ceda5a546275b3fe13dbf8cf785de0045d`

## Transcription decisions

- Item 126 preserves eng, barred `ɨ`, aspiration, ejective, `randai̯`, the
  multiword `mɨm pʰat brɨ`, and the overlapping site-`d` assignments.
- Item 127 preserves the `mandai̯`/`mandei̯` contrast, ejective `morotʼ`,
  barred `ɨ`, and the inverted-breve-below sequence in `brɨu̯`.
- Item 128 preserves four distinct group-1 forms, glottal stops, aspiration,
  and both engs in `kʰoŋkoraŋ`.
- Item 129 preserves the `mitʃɨkʼʃa`/`mitʃikʼʃa` contrast, aspiration,
  ejectives, both inverted-breve-below marks in `rao̯kmao̯`, and barred `ɨ`
  in `gɨwuɨ̯` and `mohɨla`.
- Item 130 preserves ejective plus aspiration in `apʼpʰa` and the printed
  grouping of all father terms.

## Bracket expansion and reconciliation

The 40-line ledger was frozen first. Mechanical bracket expansion produced 86
line-site records: 85 attestations and one no-entry record. The conceptual
ledger contains exactly 85 cells because item 126/site `d` has two printed
assignments.

Only after both ledgers were frozen was the legacy audit opened. The comparison
contains 86 rows:

- 74 legacy-installed records and 12 legacy exclusions;
- 11 independently recovered attestations corresponding to excluded glyphs;
- one exclusion corresponding to the printed no-entry coordinate;
- 32 exact codepoint matches among legacy-installed records;
- 42 codepoint differences among legacy-installed records;
- 12 codepoint differences among legacy-excluded records.

Reconciliation SHA-256:
`0879ed1a08df313bd9be34e97bdc68f851ef414e2268245318822c75c4a2f846`.
Comparison results are audit metadata only and neither verify nor alter manual
readings.

## Deferred gates and remaining work

Items 131-307 remain unreviewed. The item 12/site `p` source contradiction
remains unresolved. Site/lect reconciliation, staging, sound-profile
conversion, bibliography/reference validation, shared integration, full
build, graph validation, and browser QA are deferred. No shared output was
changed.
