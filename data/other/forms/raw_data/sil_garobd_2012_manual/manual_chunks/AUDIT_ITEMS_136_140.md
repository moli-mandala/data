# Manual audit - ESR 2012-007 items 136-140

## Independence and source

- Sources: physical PDF pages 70-71 / visible printed pages 63-64.
- Primary review images: the authorized 300-dpi renders.
- Small-mark review: targeted 1200-dpi crops for aspiration, barred `ɨ`,
  below-tied `d͜ʒ`, eng, ejective apostrophe, and inverted-breve-below
  sequences.
- Every form, group number, bracket code, repetition, and source space was read
  from the rendered source. OCR, PDF text, raw legacy glyphs, installed forms,
  and old audits did not supply or verify any reading.
- Site identities remained neutral printed codes `0`, `a`-`p` until after both
  ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 136-140 | Cumulative 1-140 |
| --- | ---: | ---: |
| Items | 5 | 140 |
| Printed response lines | 41 | 1,123 |
| Conceptual site cells | 85 | 2,380 |
| Ordinary attested cells | 85 | 2,321 |
| Source-conflict cells | 0 | 1 |
| Cells containing an attestation | 85 | 2,322 |
| Blank-only cells | 0 | 58 |
| Printed no-entry coordinates | 0 | 59 |
| Not-used cells | 0 | 0 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved source coordinates | 0 | 1 |
| Expanded attested response occurrences | 89 | 2,482 |

All 85 block cells are attested. Four coordinates retain overlapping printed
assignments: item 136/site `c` (`dada | kaka`), item 137/site `c`
(`ad͜ʒa | bai̯`), item 138/site `e` (`d͜ʒoŋ | d͜ʒod͜ʒoŋ`), and item
140/site `g` (`bad͜ʒu | bei̯ʃa`). There are no block-local blanks,
not-used cells, ambiguities, illegibles, source conflicts, or unresolved
source coordinates. Cumulatively, item 12/site `p` remains unresolved.

Frozen SHA-256 values:

- generator: `db405239fefbd7348c51f6faf59ad9a15534776d20afc06592a5d8573237aca8`
- line ledger: `1d9ce5295e306441c43cc8a02a9bd6c50886d5310c1405fb10147268c89ff573`
- cell ledger: `f25022fa7b90ee7348048c578d251dbbee31d8ad307b5782be8ab301fb9c0a86`

## Transcription decisions

- Item 136 preserves the overlapping site-`c` groups, aspiration and the
  inverted-breve-below sequences in `pʰao̯ tʃuŋguwa` and `bɔro bʰai̯`,
  barred `ɨ`, parentheses, and all source spaces.
- Item 137 preserves the overlapping site-`c` groups, below-tied `d͜ʒ`,
  aspiration, barred `ɨ`, the literal slash string `bɔro bon / didi`, and
  both source spaces in `hɨn min rao̯k mao̯`.
- Item 138 preserves the overlapping site-`e` groups, below-tied `d͜ʒ`, eng,
  barred-vowel and nasal contrasts in `hɨmbu` versus `hɨnbu`, ejective
  apostrophe, aspiration, and the continuous form `d͜ʒod͜ʒoŋ`.
- Item 139 preserves the group-1 alternatives, aspiration, ejective
  apostrophe, inverted-breve-below sequences, and the source's single token
  `rao̯kmao̯` after `hɨnbu`.
- Item 140 preserves the overlapping site-`g` groups, below-tied `d͜ʒ`, the
  `ʃ`/`s` contrast, vowel distinctions, aspiration, ejective apostrophe, eng,
  and source spacing in `ma lɔk`.

## Bracket expansion and reconciliation

The 41-line ledger was frozen first. Mechanical bracket expansion produced 89
attested line-site records. The conceptual ledger contains exactly 85 cells
because four coordinates have two printed assignments.

Only after both ledgers were frozen was the legacy audit opened. The comparison
contains 89 rows:

- 71 legacy-installed records and 18 legacy exclusions;
- 18 independently recovered attestations corresponding to excluded glyphs;
- 33 exact codepoint matches among legacy-installed records;
- 38 codepoint differences among legacy-installed records;
- 18 codepoint differences among legacy-excluded records.

Reconciliation SHA-256:
`819c7429977d313df812e8f914a682b2b4ac68aecb550092090e0a963bfc08a7`.
Comparison results are audit metadata only and neither verify nor alter manual
readings.

## Deferred gates and remaining work

Items 141-307 remain unreviewed. The item 12/site `p` source contradiction
remains unresolved. Site/lect reconciliation, staging, sound-profile
conversion, bibliography/reference validation, shared integration, full
build, graph validation, and browser QA are deferred. No shared output was
changed.
