# Manual-source audit

The authoritative source is the SIL ESR 2013-016 PDF pinned in `source_manifest.json`. The live SIL
file endpoint returned a Cloudflare HTML challenge, so the byte-identical archival target was
recovered from the 2024-07-24 Wayback capture and pinned by SHA-256.

Appendix B.3 occupies physical PDF pages 22--34 (printed pages 18--30). The report describes the
elicitation instrument as a standard 210-item wordlist, but the published tables contain 194 prompt
rows: the preceding methods section explicitly says problematic items were omitted from the data
lists. With two lists, Rongdani and Maituri, the published topology is therefore 388 cells.

## Complete manual review

- Reviewed cells: 388 of 388.
- Attested cells: 387.
- Source blanks: 1 (`S148_MTR`, explicitly printed `no data`).
- Ambiguous cells: 0.
- Illegible cells: 0.
- Unresolved readings: 0.
- Pending cells: 0.
- Response occurrences: 399, because Rongdani S001, S016, S032, S048, S081, S100, S102, S114, S183, S189, and S190 plus Maituri S137 each print two alternatives; the explicit S148 Maituri `no data` cell contributes none.
- Installed forms: 0 from this audit package; the pre-existing legacy installation remains unchanged.

Every retained form or source blank in the twenty chunks was hand-keyed cell by cell from 600/1200-dpi rendered
source images and visually rechecked at targeted zoom. This is a 100% visual review of all 388 published
cells. PDF text, OCR, and existing datasets were used only to locate tables or, after independent
entry was frozen, to reconcile duplicate coverage. They never supplied, seeded, completed,
normalized, inferred, or verified a lexical reading. The source's acute accents, aspiration,
unreleased marks, affricate notation, spaces, and palatal offglide were preserved diplomatically and
the resulting strings are NFC-normalized.

The unresolved ledger is empty after exhaustive review. No published cell is ambiguous or illegible.

## Post-entry inventory

Only after the first independent 20-cell chunk was frozen, the repository was searched for overlap. The
source already has a legacy Unicode extraction (`data/other/forms/20260813-rabha.csv`) containing
400 expanded response rows: 205 Rongdani and 195 Maituri. It represents the same 194 printed prompt
rows and already has shared citation, dialect, and sound-profile wiring. That existing installation
was not read before manual entry and supplied no transcription evidence.

The first comparison found eighteen exact cells plus two exact alternatives in S001, with three
token-level differences. Each difference triggered a new inspection of the rendered source, without
using the installed value as evidence. The image shows `n` in Rongdani S001's second alternate and a
small inverted breve on `ʃ` in both S008 forms; these three image-evidenced corrections make all 21
response occurrences agree exactly with the installed extraction.

The second 20-cell chunk was likewise frozen before comparison. Nineteen of its 21 expanded response
occurrences agree exactly. The only two differences are Rongdani S016 alternative 1 and S017: the
scan visibly places the small inverted breve on `ʃ` (`tʃ̑ɑ...`), while the legacy extraction stores
the combining mark after `ɑ` (`tʃɑ̑...`). The source-image reading is retained; the legacy file is
neither evidence nor changed.

The third 20-cell chunk was frozen from the 1200-dpi scans of physical pages 23--24 before duplicate
comparison. Initial differences prompted image-only reinspection of the p.23 line wrap, the p.24
`ʋ` glyph, and the p.24 pestle pair's `g`/`ɡ`, medial nasal, and barred-vowel distinctions. After
those visual corrections, all 20 response occurrences agree exactly with the legacy extraction.
Agreement remains a post-entry audit result, never transcription evidence. No shared file was changed
in any checkpoint.

The fourth 20-cell chunk was frozen from physical pages 24--25 before duplicate comparison. Eighteen
of its 21 response occurrences agree exactly with the legacy extraction. Three image-evidenced
distinctions remain: the small inverted breve is visibly on `ʃ` in S032 Rongdani alternative 2; S034
Rongdani visibly prints double-storey `g`; and both small inverted breves are visibly on `ʃ` in S038
Rongdani. These are representation differences, not unresolved readings, and no shared file changed.

The fifth 20-cell chunk was frozen from physical p.25 before duplicate comparison. Eighteen of its
21 occurrences agree exactly with legacy. Three source-image distinctions remain: S044 Rongdani
prints a small inverted breve on `ʃ` plus nasalized small-capital `ɪ̃`; S044 Maituri and S045 Maituri
place the small inverted breve on `ʃ`, rather than on the following `k`. These are visually resolved
representation differences, not unresolved readings, and no shared file changed.

The sixth 20-cell chunk was frozen from physical pp.25--26 before duplicate comparison. Seventeen of
its 20 occurrences agree exactly with legacy. Three source-image distinctions remain: Rongdani S052
and both S060 forms visibly place the small inverted breve on `ʃ`, while legacy attaches it to the
following vowel. Targeted scan reinspection also visibly confirmed engma in both S055 forms and the
Maituri S057 medial open back vowel before the final chunk was pinned. All readings are resolved and
the legacy installation remains unchanged.

The seventh 20-cell chunk was frozen from physical p.26 before duplicate comparison. Fourteen of its
20 occurrences agree exactly with legacy. Six source-image distinctions remain: both S061 forms have
scan-resolved segment and combining-mark readings; both S067 forms and S070 Maituri visibly retain
the source's double-storey `g`; and S068 Maituri visibly ends in engma. These are resolved diplomatic
distinctions, not ambiguities or illegibilities, and the legacy installation remains unchanged.

The eighth 20-cell chunk was frozen from physical p.27 before duplicate comparison. Fifteen of its
20 occurrences agree exactly with legacy. The five differences all preserve the source's visibly
double-storey `g` in S076 Rongdani/Maituri, S077 Rongdani/Maituri, and S078 Maituri. A post-entry
mismatch on S072 triggered a scan-only targeted reinspection, which visibly resolved Rongdani schwa
versus Maituri open back vowel and the Maituri final unreleased mark before the chunk was pinned. No
reading is ambiguous or illegible, and the legacy installation remains unchanged.

The ninth 20-cell chunk was frozen from physical pp.27--28 before duplicate comparison. It contains
21 response occurrences because Rongdani S081 prints two comma-separated alternatives. Initial
differences at S082 and S085 triggered scan-only targeted reinspection, visibly resolving barred `ɨ`,
the Rongdani final acute accent, and Maituri small-capital `ɪ`. After these image-only corrections,
all 21 occurrences agree exactly with legacy. Agreement remains a post-entry audit result, no reading
is ambiguous or illegible, and the legacy installation remains unchanged.

The tenth 20-cell chunk was frozen from physical p.28 before duplicate comparison. It contains 21
response occurrences because Rongdani S100 prints two line-separated alternatives. Twenty
occurrences agree exactly with legacy. Maituri S096 differs only because the scan visibly places the
small inverted breve on `ʃ`, while legacy attaches it to the following vowel. Post-entry mismatches
at S091 and S092 also triggered scan-only targeted reinspection, which visibly resolved barred `ɨ`
and small-capital `ɪ` before the chunk was pinned. All cells remain resolved and legacy is unchanged.

The eleventh 20-cell chunk was frozen from physical pp.28--29 before duplicate comparison. It has 21
response occurrences because Rongdani S102 prints two comma-separated alternatives. Post-entry
mismatches at S101 and S105 triggered scan-only targeted reinspection, which visibly resolved the
accented open back vowel in `sɑ́bɾɑ` and Rongdani normal `r` versus Maituri tap in the younger-brother
forms. After those image-only corrections, all 21 occurrences agree exactly with legacy. All cells
remain resolved and the legacy installation remains unchanged.

The twelfth 20-cell chunk was frozen from physical pp.29--30 before duplicate comparison. It has 21
response occurrences because Rongdani S114 prints two comma-separated alternatives. A single
post-entry mismatch at Rongdani S118 triggered source-image-only targeted reinspection, which
visibly confirmed medial schwa in `ɾɑŋsəɾi`. After that image-only correction, all 21 occurrences
agree exactly with the legacy extraction. This agreement supplied no reading, all cells remain
resolved, and the legacy installation remains unchanged.

The thirteenth 20-cell chunk was frozen from physical p.30 before duplicate comparison and has 20
response occurrences. Post-entry mismatches triggered source-image-only targeted reinspection.
The scan visibly confirms Latin `a` in S126 Rongdani and medial `n` in both S130 forms, making those
exact legacy agreements. Two resolved diplomatic distinctions remain: S121 Rongdani retains the
source's double-storey `g` rather than legacy IPA `ɡ`, and S125 Rongdani retains the visibly
superscript palatal offglide `ʲ` rather than legacy baseline `j`. Thus 18 of 20 occurrences agree
exactly; no cell is ambiguous or illegible, and legacy remains unchanged.

The fourteenth 20-cell chunk was frozen from physical pp.30--31 before duplicate comparison and has
21 response occurrences because Maituri S137 prints two line-separated alternatives. Scan-only
reinspection resolved Rongdani S132/S133 `q`, Maituri S137 first-alternative double-storey `g`,
Maituri S138 length, and Rongdani S140 length. Sixteen occurrences agree exactly with legacy. Five
resolved diplomatic distinctions remain: small inverted breve placement at S134 Rongdani, S136
Maituri, and both S139 forms, plus double-storey `g` versus legacy IPA `ɡ` in Maituri S137's first
alternative. No cell is ambiguous or illegible, and legacy remains unchanged.

The fifteenth 20-cell chunk was frozen from physical pp.31--32 before duplicate comparison. It has
19 attested occurrences and one explicit source blank: Maituri S148 prints the textual notation
`no data`, which is excluded from lexical occurrences rather than installed as a form. Fourteen of
the 19 attested occurrences agree exactly with legacy. Five resolved source distinctions remain:
the small inverted breve is visibly on `ʃ` in both S142 and both S149 forms, and Rongdani S148
retains the source's double-storey `g` rather than legacy IPA `ɡ`. No cell is ambiguous or illegible,
and legacy remains unchanged.

The sixteenth 20-cell chunk was independently frozen from physical p.32 before duplicate comparison.
All twenty cells are attested, with no blank, ambiguity, illegibility, or unresolved reading.
Post-entry mismatches triggered source-image-only targeted reinspection of both S153 final unreleased
marks, the S155--S156 open back vowels, the spanning affricate ties at S157--S158, and both S159 open
back vowels. Those image-only checks yield fifteen exact legacy agreements. Five resolved diplomatic
differences remain: the scan visibly prints double-storey `g` in Maituri S156, Maituri S158, Rongdani
S159, and both S160 forms, while legacy uses IPA `ɡ`. No shared file was changed.

The seventeenth 20-cell chunk was independently frozen from physical pp.32--33 before duplicate
comparison. All twenty cells are attested, with no blank, ambiguity, illegibility, or unresolved
reading. Post-entry mismatches triggered source-image-only targeted reinspection of the Maituri S161
engma, the S163 barred/open-back vowels and combining marks, Rongdani S164 accented schwa, Rongdani
S165 barred vowel, Maituri S168 barred vowel, Rongdani S169 barred vowel, and Rongdani S170 combining
mark. Seventeen occurrences agree exactly with legacy. Three resolved combining-mark distinctions
remain: the scan visibly places the small inverted breve on `ʃ` in both S163 forms and Rongdani S170,
while legacy attaches it to a later segment. No shared file was changed.

The eighteenth 20-cell chunk was independently frozen from physical p.33 before duplicate comparison.
All twenty cells are attested, with no blank, ambiguity, illegibility, or unresolved reading.
Post-entry mismatches triggered source-image-only targeted reinspection of the Maituri S172 and S176
barred vowels. Those corrections yield sixteen exact legacy agreements. Four resolved diplomatic
differences remain: both S171 forms and both S172 forms retain the scan's double-storey `g`, while
legacy uses IPA `ɡ`. No shared file was changed.

The nineteenth 20-cell chunk was independently frozen from physical pp.33--34 before duplicate
comparison. All twenty cells are attested and expand to twenty-three response occurrences because
Rongdani S183, S189, and S190 each print two comma-delimited alternatives. A single post-entry
mismatch at Maituri S182 triggered source-image-only targeted reinspection; the 1200-dpi crop
visibly confirms barred `ɨ` in `ɾɨʋɑ`. After that image-only correction, all twenty-three
occurrences agree exactly with the legacy extraction. No cell is ambiguous or illegible, and no
shared file was changed.

The twentieth and final eight-cell chunk was independently frozen from physical p.34 before duplicate
comparison. All eight cells are attested. Targeted rendered-page inspection visibly resolved the
Rongdani `i` versus Maituri barred `ɨ` contrast at S191 and S192, Maituri S192's final superscript
palatal offglide, and the medial taps in both S193 and both S194 forms. All eight occurrences agree
exactly with the legacy extraction. No cell is ambiguous or illegible, and no shared file was changed.
