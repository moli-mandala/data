# Manual audit - ESR 2012-007 items 101-105

## Independence and source

- Source: physical PDF page 65 / visible printed page 58, both columns.
- Primary review image: the authorized 300-dpi render.
- Small-mark review: targeted 1200-dpi crops for the alveolar tap, aspiration,
  barred `ɨ`, small-cap `ɪ`, near-close `ʊ`, palatalization,
  inverted-breve-below, and ejective apostrophe.
- Every form, group number, bracket code, and repetition was read from the
  rendered source. OCR, PDF text, raw legacy glyphs, installed forms, and old
  audits did not supply or verify any reading.
- Site identities remained neutral printed codes `0`, `a`-`p` until after both
  ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 101-105 | Cumulative 1-105 |
| --- | ---: | ---: |
| Items | 5 | 105 |
| Printed response lines | 38 | 814 |
| Conceptual site cells | 85 | 1,785 |
| Ordinary attested cells | 85 | 1,737 |
| Source-conflict cells | 0 | 1 |
| Cells containing an attestation | 85 | 1,738 |
| Blank-only cells | 0 | 47 |
| Printed no-entry coordinates | 0 | 48 |
| Not-used cells | 0 | 0 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved source coordinates | 0 | 1 |
| Expanded attested response occurrences | 91 | 1,864 |

All 85 block cells are attested. Identical group-1/group-2 assignments are
retained at item 103/sites `l,m,g,h,i,d`. There are no block-local blanks,
not-used cells, ambiguities, illegibles, source conflicts, or unresolved source
coordinates. Cumulatively, item 12/site `p` remains unresolved.

Frozen SHA-256 values:

- generator: `ab8a082aec08be494ad468008544006d8c9e7e8fb4d8bb61456126b394ed18d2`
- line ledger: `f2d5815abb6464c56c8f233ba5065639d44658fee35125d82ada2b325ddfc8f1`
- cell ledger: `21674f63024b05a1aaf172b1aab290da0ac89db2f68de7444b38fbbf87cac330`

## Transcription decisions

- Item 101 preserves the ejective in `gɨtʼdok`, barred `ɨ`, aspiration, and
  alveolar tap in `tokɾɛŋ`; the two separately printed `kraŋ` lines remain
  distinct source rows.
- Item 102 preserves palatalization in `snʲɨk`, aspiration, barred `ɨ`, and
  the inverted-breve-below in `kʰau̯` and `snia̯k`.
- Item 103 keeps all six repeated group-1/group-2 assignments and preserves
  alveolar `ɾ`, barred `ɨ`, aspiration, and ejective `ʼ`.
- Item 104 keeps `lɨmʊt` distinct from `lɨmut`, preserving the printed
  near-close `ʊ`, barred `ɨ`, aspiration, and ejective `ʼ`.
- Item 105 preserves barred `ɨ` and aspiration in the printed ear forms.

## Bracket expansion and reconciliation

The 38-line ledger was frozen first. Mechanical bracket expansion produced 91
attested line-site records. The conceptual ledger contains exactly 85 cells
because six item-103 coordinates have a second printed assignment.

Only after both ledgers were frozen was the legacy audit opened. The comparison
contains 91 rows:

- 86 legacy-installed records and five legacy exclusions;
- five independently recovered attestations corresponding to excluded glyphs;
- 38 exact codepoint matches among legacy-installed records;
- 48 codepoint differences among legacy-installed records;
- five codepoint differences among legacy-excluded records.

Reconciliation SHA-256:
`0d830ad11d5902271d3b4759bc850812e5cfbb910e5bb88ebf5abf57729b965d`.
Comparison results are audit metadata only and neither verify nor alter manual
readings.

## Deferred gates and remaining work

Items 106-307 remain unreviewed. The item 12/site `p` source contradiction remains
unresolved. Site/lect reconciliation, staging, sound-profile conversion,
bibliography/reference validation, shared integration, full build, graph
validation, and browser QA are deferred. No shared output was changed.
