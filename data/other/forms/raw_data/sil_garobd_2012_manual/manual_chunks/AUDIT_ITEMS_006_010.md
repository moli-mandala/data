# Manual audit - ESR 2012-007 items 6-10

## Independence and source

- Items 6-7: physical PDF page 52 / visible printed page 45, right column.
- Items 8-10: physical PDF page 53 / visible printed page 46, left column.
- Primary review images: the authorized 300-dpi renders.
- Small-mark review: targeted 1200-dpi crops for non-syllabic marks,
  aspiration, modifier apostrophes, barred `ɨ`, small-cap `ɪ`, and glottal stop.
- Every form, group number, bracket code, and status was read from the rendered
  source. OCR, PDF text, raw legacy glyphs, installed forms, and old audits did
  not supply or verify any reading.
- Site identities remained the neutral printed codes `0`, `a`-`p` until after
  both manual ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 6-10 | Cumulative 1-10 |
| --- | ---: | ---: |
| Items | 5 | 10 |
| Printed response lines | 41 | 88 |
| Conceptual site cells | 85 | 170 |
| Attested cells | 76 | 160 |
| Source blanks | 9 | 10 |
| Not-used cells | 0 | 0 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved coordinates | 0 | 0 |
| Expanded attested response occurrences | 79 | 174 |

The nine exact blank coordinates are:

- item 7: sites `f`, `l`, `o`, `p`;
- item 9: sites `f`, `n`;
- item 10: sites `b`, `o`, `p`.

Each comes from a printed group-0 `no entry` line. There are no other
exclusions in items 6-10.

Frozen SHA-256 values:

- line ledger: `cd0950f37f1587e044ec8aa678d6309002907469e6a977345db55999e8394aed`
- cell ledger: `5aecba4aae2f5edbb8422c797a72815529ac12d755d985e900ae2dd83db6b337`

## Transcription decisions

- Parenthesized printed strings in lightning and thunder responses are retained
  inside the diplomatic form, for example `(mɨkʼkʰa) hɛlapuŋa`.
- Printed superscript aspiration is `ʰ`; final raised apostrophe is modifier
  apostrophe `ʼ`; printed glottal stop is `ʔ`.
- The small arc below a vowel is combining inverted breve below. It is retained
  in `pʰleo̯`, `balua̯`, `lou̯ni`, and `lau̯ni`.
- Item 8 distinguishes `lɨŋ ɪr` at sites `j,k` from `lɨŋ ir` at site `p`.
- Multiple responses within one conceptual cell remain ordered and repeated:
  item 7/site `n` has `ramdʰonukʼ | rɔŋdʰɔnu`, both group 4; item 10/sites
  `e,f` retain `kʰum prɛʔta | kʰum prɛʔta`, groups 1 and 2.
- Spaces and parentheses are source characters, not later analysis.

## Bracket expansion and reconciliation

The 41-line ledger was frozen first. Mechanical bracket-code expansion then
accounted for every `(item, printed-site-code)` coordinate while retaining
ordered alternatives and repeated printed responses.

Only after the 85-cell ledger was frozen was the legacy audit opened. The
post-freeze comparison contains 88 rows (79 attested occurrences plus 9 blanks):

- 64 legacy-installed records and 24 legacy exclusions;
- 15 independently recovered attestations corresponding to formerly excluded
  glyph sequences;
- 9 legacy exclusions corresponding to explicit source blanks;
- 12 exact codepoint matches among legacy-installed records;
- 52 codepoint differences among legacy-installed records;
- 24 codepoint differences among legacy-excluded records.

Those matches and differences are audit metadata only. They do not establish
correctness and never feed back into the manual generator or ledgers.

## Deferred gates and remaining work

Items 11-307 remain unreviewed. Site-name and lect reconciliation, source-local
staging, sound-profile conversion, bibliography/reference validation, shared
integration, full build, graph validation, and browser QA are deferred. No shared
output was changed by this checkpoint.
