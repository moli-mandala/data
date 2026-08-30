# Manual audit - ESR 2012-007 items 11-15

## Independence and source

- Source: physical PDF page 53 / visible printed page 46.
- Item 11 is in the left column; items 12-15 are in the right column.
- Primary review image: the authorized 300-dpi render.
- Small-mark review: targeted 1200-dpi crops for non-syllabic marks,
  aspiration, barred `ɨ`, dental `t̪`, and glottal stop.
- Every form, group number, bracket code, and status was read from the rendered
  source. OCR, PDF text, raw legacy glyphs, installed forms, and old audits did
  not supply or verify any reading.
- Site identities remained the neutral printed codes `0`, `a`-`p` until after
  both manual ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 11-15 | Cumulative 1-15 |
| --- | ---: | ---: |
| Items | 5 | 15 |
| Printed response lines | 34 | 122 |
| Conceptual site cells | 85 | 255 |
| Ordinary attested cells | 79 | 239 |
| Source-conflict cells | 1 | 1 |
| Cells containing an attestation | 80 | 240 |
| Blank-only cells | 5 | 15 |
| Printed no-entry coordinates | 6 | 16 |
| Not-used cells | 0 | 0 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved coordinates | 1 | 1 |
| Expanded attested response occurrences | 80 | 254 |

The six printed no-entry coordinates are item 11/sites `a,f,h,n` and item
12/sites `l,p`. Item 12/site `p` is not a blank-only cell: the same page also
prints group-6 `dɔm [p]`. Both lines are retained. The conceptual cell is marked
`source_conflict`, and that exact coordinate is the block's sole unresolved case.
There is no ambiguity about what either line prints.

Frozen SHA-256 values:

- line ledger: `3eacb09dda9f6a108741150f35a725d38553f894882d5696c9e38d77fc98a751`
- cell ledger: `798064191bbfb60d8b2c8a9ef6deda8231126b07a708c4ca48785dc2f55e4e92`

## Transcription decisions

- The small arc below a vowel is combining inverted breve below. It is retained
  in `duriou̯`, `tɨi̯`, `kmao̯`, `tei̯ kʰar`, `tei̯ muŋ`, and `kmia̯n`.
- The diplomatic layer distinguishes barred `ɨ`, small-cap `ɪ`, and ordinary
  `i`; printed aspiration is `ʰ`; printed glottal stop is `ʔ`.
- Item 15 group 3 retains the dental diacritic in `mat̪i`.
- IPA `ɡ` is retained in forms such as `ʃaɡal`, `haʔroŋɡa`, `ɡum`, and `ɡaŋ`.
- The item 12/site `p` contradiction is source evidence, not normalized away:
  its cell stores attested form `dɔm`, cites both source lines, and records the
  simultaneous printed group-0 no-entry status.

## Bracket expansion and reconciliation

The 34-line ledger was frozen first. Mechanical bracket expansion produced 86
line-site records because item 12/site `p` occurs once on the no-entry line and
once on an attested line. The conceptual ledger still contains exactly 85 cells.

Only after both ledgers were frozen was the legacy audit opened. The post-freeze
comparison contains 86 rows:

- 72 legacy-installed records and 14 legacy exclusions;
- 8 independently recovered attestations corresponding to formerly excluded
  glyph sequences;
- 6 legacy exclusions corresponding to printed no-entry coordinates;
- 41 exact codepoint matches among legacy-installed records;
- 31 codepoint differences among legacy-installed records;
- 14 codepoint differences among legacy-excluded records.

Those matches and differences are audit metadata only. They do not establish
correctness and never feed back into the manual generator or ledgers.

## Deferred gates and remaining work

Items 16-307 remain unreviewed. The item 12/site `p` source contradiction remains
unresolved. Site-name and lect reconciliation, source-local staging, sound-profile
conversion, bibliography/reference validation, shared integration, full build,
graph validation, and browser QA are deferred. No shared output was changed.
