# Manual audit - ESR 2012-007 items 16-20

## Independence and source

- Source: physical PDF page 54 / visible printed page 47.
- Items 16-18 are in the left column; items 19-20 are in the right column.
- Primary review image: the authorized 300-dpi render.
- Small-mark review: targeted 1200-dpi crops for non-syllabic marks,
  aspiration, barred `ɨ`, glottal stop, ejective apostrophe, and retroflex `ɖ`.
- Every form, group number, bracket code, and status was read from the rendered
  source. OCR, PDF text, raw legacy glyphs, installed forms, and old audits did
  not supply or verify any reading.
- Site identities remained the neutral printed codes `0`, `a`-`p` until after
  both manual ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 16-20 | Cumulative 1-20 |
| --- | ---: | ---: |
| Items | 5 | 20 |
| Printed response lines | 49 | 171 |
| Conceptual site cells | 85 | 340 |
| Ordinary attested cells | 84 | 323 |
| Source-conflict cells | 0 | 1 |
| Cells containing an attestation | 84 | 324 |
| Blank-only cells | 1 | 16 |
| Printed no-entry coordinates | 1 | 17 |
| Not-used cells | 0 | 0 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved coordinates | 0 | 1 |
| Expanded attested response occurrences | 99 | 353 |

The block's sole blank is item 17/site `f`, printed as group-0 `no entry`.
Every other conceptual cell is attested. There are no block-local ambiguities,
illegibles, source conflicts, or unresolved coordinates. Cumulatively, item
12/site `p` remains unresolved because the source prints both no-entry and `dɔm`.

Frozen SHA-256 values:

- line ledger: `f23624bd2856ca7ac46b8d314595a928730b2fa2aac1e8b9741044733c2a0a58`
- cell ledger: `b8f3d4d35b74deea079506b20090692e54670a591961cdb92bbe5a3de9b597a5`

## Transcription decisions

- Repeated source groups are retained rather than collapsed. For example,
  item 16/site `i` stores `hadɨbɛkʼ | hadɨbɛkʼ` from groups 2 and 3, and item
  19/sites `a,o` retain identical group-1 and group-2 responses.
- Item 16 preserves the printed slash response `haʔdilɛka / kadoŋ`, the
  ejective apostrophe `ʼ`, barred `ɨ`, and glottal stop `ʔ`.
- Item 17 preserves the ordinary printed Latin `g` in `hagundula`, distinct
  from IPA `ɡ`, plus aspiration in `hantʃʰɛŋ`, `mɛŋpɨrpʰu`, `habukʰu`, and
  `dʰula`.
- The small arc below a vowel is combining inverted breve below. It is retained
  in `loŋtʰai̯`, `roŋtʰai̯`, `mao̯`, `ɖʒia̯p`, and `ɖʒmia̯k`.
- Item 19 retains the visibly printed retroflex `ɖ` in `ɖʒia̯p`, `ɖʒiɛp`, and
  `ɖʒmia̯k`; it is not normalized to ordinary `d`.
- Item 20's final line covers exactly the fourteen codes
  `0,a,b,c,d,e,f,g,h,i,l,m,n,o`; sites `j,k,p` retain the separate group-1
  responses printed above it.

## Bracket expansion and reconciliation

The 49-line ledger was frozen first. Mechanical bracket expansion produced 100
line-site records: 99 attested response occurrences and one printed no-entry.
The conceptual ledger contains exactly 85 cells.

Only after both ledgers were frozen was the legacy audit opened. The post-freeze
comparison contains 100 rows:

- 86 legacy-installed records and 14 legacy exclusions;
- 13 independently recovered attestations corresponding to formerly excluded
  glyph sequences;
- one legacy exclusion corresponding to the printed no-entry coordinate;
- 38 exact codepoint matches among legacy-installed records;
- 48 codepoint differences among legacy-installed records;
- 14 codepoint differences among legacy-excluded records.

Those matches and differences are audit metadata only. They do not establish
correctness and never feed back into the manual generator or ledgers.

## Deferred gates and remaining work

Items 21-307 remain unreviewed. The earlier item 12/site `p` source contradiction
remains unresolved. Site-name and lect reconciliation, source-local staging,
sound-profile conversion, bibliography/reference validation, shared integration,
full build, graph validation, and browser QA are deferred. No shared output was
changed.
