# Manual audit - ESR 2012-007 items 1-5

## Independence and source

- Source: physical PDF page 52, visible printed page 45, left and right columns.
- Primary review image: the authorized 300-dpi render `page-052.png`.
- Small-mark review: targeted 1200-dpi crops for non-syllabic marks,
  aspiration, modifier apostrophes, tilde, barred `ɨ`, and bracket code `i`.
- Every form, group number, bracket code, and status was read from the rendered
  source. OCR, PDF text, raw legacy glyphs, installed forms, and old audits did
  not supply or verify any reading.
- Site identities remained the neutral printed codes `0`, `a`-`p` until after
  both manual ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Count |
| --- | ---: |
| Items | 5 |
| Printed response lines | 47 |
| Conceptual site cells | 85 |
| Attested cells | 84 |
| Source blanks | 1 |
| Not-used cells | 0 |
| Ambiguous cells | 0 |
| Illegible cells | 0 |
| Unresolved coordinates | 0 |
| Expanded attested response occurrences | 95 |

The sole blank is exactly `item 5 / site p / group 0`, where the source prints
`no entry`. There are no other exclusions in items 1-5.

Frozen SHA-256 values:

- line ledger: `938ef6b0947f057ea2f3ba187b6b83f60fae159a5acf1536dd04f377e1718e96`
- cell ledger: `18cb09f9aeda5d2dfdf47f98f59102e9c31a296500846ee658a59885c07f25c9`

## Transcription decisions

- The diplomatic layer uses IPA `ɡ`, `ɨ`, `ŋ`, `ʃ`, `ʒ`, `ɛ`, and `ɔ` for the
  corresponding printed symbols.
- Printed superscript aspiration is `ʰ`; the printed glottal stop is `ʔ`.
- The raised apostrophe after final consonants is modifier apostrophe `ʼ`.
- The small arc below a vowel is combining inverted breve below, as in `o̯`
  and `i̯`; it was not silently discarded or interpreted through legacy data.
- The nasalized vowel in item 3 group 7 is preserved as `ã`.
- Spaces inside multiword responses are retained (`jao̯ bri`, `tʰao̯ bti`,
  `tʃaŋ ɨi̯`, `tʃaŋ ai̯`).
- The source's repeated response groups are retained as distinct response
  occurrences. Examples include item 3 sites `g,h` (`dʒonakʼ`, groups 3 and
  6) and item 4 sites `a,d,g,h,o`, `e`, `f`, and `i` across repeated groups.

## Bracket expansion and reconciliation

The 47-line ledger was frozen first. Bracket-code expansion then accounted for
every `(item, printed-site-code)` coordinate exactly once at the conceptual-cell
level while retaining multiple response occurrences inside a cell.

Only after the 85-cell ledger was frozen was the legacy audit opened. The
post-freeze comparison contains 96 rows (95 attested occurrences plus the blank):

- 85 legacy-installed records and 11 legacy exclusions;
- 10 independently recovered attestations corresponding to formerly excluded
  glyph sequences;
- one legacy exclusion corresponding to the explicit source blank;
- 46 exact codepoint matches among legacy-installed records;
- 39 codepoint differences among legacy-installed records;
- 11 codepoint differences among legacy-excluded records.

Those matches and differences are audit metadata only. They do not establish
correctness and never feed back into the manual generator or ledgers.

## Deferred gates and remaining work

Items 6-307 remain unreviewed. Site-name and lect reconciliation, source-local
staging, sound-profile conversion, bibliography/reference validation, shared
integration, full build, graph validation, and browser QA are deferred. No shared
output was changed by this checkpoint.
