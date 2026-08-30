# Manual audit - ESR 2012-007 items 61-65

## Independence and source

- Source: physical PDF pages 59-60 / visible printed pages 52-53.
- Primary review images: the authorized 300-dpi renders.
- Small-mark review: targeted 1200-dpi crops for palatalization, glottal stop,
  aspiration, barred `ɨ`, inverted-breve-below, open vowels, and ejective
  apostrophe.
- Every form, group number, bracket code, and blank was read from the rendered
  source. OCR, PDF text, raw legacy glyphs, installed forms, and old audits did
  not supply or verify any reading.
- Site identities remained neutral printed codes `0`, `a`-`p` until after both
  ledgers were frozen. Code `0` remains control/audit-only.

## Frozen block census

| Measure | Block 61-65 | Cumulative 1-65 |
| --- | ---: | ---: |
| Items | 5 | 65 |
| Printed response lines | 34 | 482 |
| Conceptual site cells | 85 | 1,105 |
| Ordinary attested cells | 83 | 1,071 |
| Source-conflict cells | 0 | 1 |
| Cells containing an attestation | 83 | 1,072 |
| Blank-only cells | 2 | 33 |
| Printed no-entry coordinates | 2 | 34 |
| Not-used cells | 0 | 0 |
| Ambiguous cells | 0 | 0 |
| Illegible cells | 0 | 0 |
| Unresolved coordinates | 0 | 1 |
| Expanded attested response occurrences | 83 | 1,150 |

The block blanks are item 64/site `p` and item 65/site `p`, both printed as
group-0 `no entry`. Every other cell is attested. There are no block-local
ambiguities, illegibles, source conflicts, or unresolved coordinates.
Cumulatively, item 12/site `p` remains unresolved.

Frozen SHA-256 values:

- generator: `429b954f3fcf4d7c459fc3eba821e08f94af6252f38e80841333f8e8be15a1fd`
- line ledger: `0cf581b3c0130c4f61b1107021210054a608c959267fd8972ec6946e59cdb432`
- cell ledger: `330257bfecd41354754894070ae3f49142b96c5f49d61445123895dc8f98f9a8`

## Transcription decisions

- Item 61 preserves palatalization in `sonʲɛŋ`, the glottal stop and
  inverted-breve-below in `suʔ nia̯ŋ`, and the control response `tɛl`.
- Item 62 keeps the aspiration/ejective contrast of `pɛkʼen`, the `ɛ` vowel
  in `bɛʔɛn`/`pɛʔɛn`, and the independent forms `mɨn`, `mim`, `randai̯`,
  and `maŋʃo`.
- Item 63 preserves the source contrasts among `kʰai̯ʃum`, `kʰasɨm`, `ʃum`,
  `sɨm`, and `sum`, plus the printed control alternatives `lɔbon / nun`.
- Item 64 retains both inverted-breve-below marks in `pia̯o̯`, and keeps the
  vowel and sibilant distinctions among `rai̯sun`, `raʃun`, `rɔʃun`, and
  `ruʃun`. The control response is `pɛa̯dʒ`.
- Item 65 preserves `nasɨn dukʼkʰi`, `nasɨn gipʼpok`, and the source's
  `rɔʃon`/`rɔʃun`/`ruʃun` distinctions without normalization.

## Bracket expansion and reconciliation

The 34-line ledger was frozen first. Mechanical bracket expansion produced 85
line-site records: 83 attestations and two no-entry records. There are no
repeated site assignments in this block, so the conceptual ledger also contains
exactly 85 cells.

Only after both ledgers were frozen was the legacy audit opened. The comparison
contains 85 rows:

- 72 legacy-installed records and 13 legacy exclusions;
- 11 independently recovered attestations corresponding to excluded glyphs;
- two exclusions corresponding to printed no-entry coordinates;
- 47 exact codepoint matches among legacy-installed records;
- 25 codepoint differences among legacy-installed records;
- 13 codepoint differences among legacy-excluded records.

Reconciliation SHA-256:
`39aa8a7318daed6db56079cf79370a8cad21843e35aa9b1baef415f368cbe729`.
Comparison results are audit metadata only and neither verify nor alter manual
readings.

## Deferred gates and remaining work

Items 66-307 remain unreviewed. The item 12/site `p` source contradiction remains
unresolved. Site/lect reconciliation, staging, sound-profile conversion,
bibliography/reference validation, shared integration, full build, graph
validation, and browser QA are deferred. No shared output was changed.
