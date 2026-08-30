# Manual audit - ESR 2012-007 items 151-155

## Independence and source

- Source: physical PDF page 72 / visible printed page 65.
- Primary review image: the authorized 300-dpi render.
- Small-mark review: targeted 1200-dpi crops for barred `ɨ`, aspiration,
  glottal stop, ejective apostrophe, below-tied `d͜ʒ`, eng, `ɛ`, `ɔ`, and
  inverted-breve-below sequences.
- Every form, group number, bracket code, repetition, source space, and the
  item-level `[not used]` disposition was read from the rendered source. OCR,
  PDF text, raw legacy glyphs, installed forms, and old audits did not supply
  or verify any reading.
- Site identities remained neutral printed codes `0`, `a`-`p` until after both
  ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 151-155 | Cumulative 1-155 |
| --- | ---: | ---: |
| Items | 5 | 155 |
| Printed response/disposition lines | 31 | 1,221 |
| Conceptual site cells | 85 | 2,635 |
| Ordinary attested cells | 68 | 2,556 |
| Source-conflict cells | 0 | 1 |
| Cells containing an attestation | 68 | 2,557 |
| Blank-only cells | 0 | 61 |
| Printed no-entry coordinates | 0 | 62 |
| Not-used cells | 17 | 17 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved source coordinates | 0 | 1 |
| Expanded attested response occurrences | 72 | 2,728 |

Item 152 is printed once as `[not used]` for the whole item. Its 17 conceptual
site cells are therefore `not_used`, not ordinary blanks and not 17 separate
printed disposition lines. Item 155/sites `g,h` retain identical group-2/group-5
`ʃutʃʰi` assignments, and sites `d,n` retain identical group-2/group-5
`ʃutʃi` assignments. There are no block-local blanks, ambiguities, illegibles,
source conflicts, or unresolved source coordinates. Cumulatively, item 12/site
`p` remains unresolved.

Frozen SHA-256 values:

- generator: `34806b7635d59ac9f85907b45a2d0672643ce49c0c5a159e24b48e35e85b2b97`
- line ledger: `4a27e46d006b8ed41aaf9b0eaf91b0512d2c26509e1a41be64a8a59977f0ce8e`
- cell ledger: `30a0027ed820e86bd60012401bbbd15ea907436efe0c5faec83cee656f16dbf8`

## Transcription decisions

- Item 151 preserves the glottal stop in `baʔara`, aspiration and barred `ɨ`,
  below-tied `d͜ʒ`, the inverted-breve-below sequence in `d͜ʒai̯n`, and the
  source space and aspiration in `d͜ʒai̯n pʰoŋ`.
- Item 152 preserves the source's exact whole-item `[not used]` disposition and
  mechanically applies its scope to all 17 conceptual site cells.
- Item 153 preserves `ʃ` versus `s`, the ejective apostrophe in `pantʃakʼ`,
  and the inverted-breve-below sequence in `duwai̯`.
- Item 154 preserves `ɛ`, ejective and aspirated segments in `lɛkʼkʰa`, `ɔ`
  in `kɔt`, and below-tied `d͜ʒ` in `kagod͜ʒ`.
- Item 155 preserves eng, `ɛ`, aspiration contrasts, barred `ɨ`, source spacing
  in `tʰɨr ria`, and all repeated group-2/group-5 assignments.

## Bracket expansion and reconciliation

The 31-line response/disposition ledger was frozen first. Mechanical expansion
produced 85 conceptual cells: 68 attestations and 17 cells carrying the single
whole-item not-used disposition. Attested bracket expansion produced 72
line-site occurrences because item 155 has four repeated coordinates.

Only after both ledgers were frozen was the legacy audit opened. The comparison
contains 73 rows: 72 attested line-site records plus one whole-item disposition.

- 68 legacy-installed records and five legacy exclusions;
- four independently recovered attestations corresponding to excluded glyphs;
- one manual whole-item not-used disposition matching the legacy record;
- 35 exact codepoint matches among legacy-installed records;
- 33 codepoint differences among legacy-installed records;
- five codepoint differences among legacy-excluded records.

Reconciliation SHA-256:
`54fe3b2903309427829597b3a422e083a089fa31831ce848930fb46188c739ae`.
Comparison results are audit metadata only and neither verify nor alter manual
readings.

## Deferred gates and remaining work

Items 156-307 remain unreviewed. The item 12/site `p` source contradiction
remains unresolved. Site/lect reconciliation, staging, sound-profile
conversion, bibliography/reference validation, shared integration, full
build, graph validation, and browser QA are deferred. No shared output was
changed.
