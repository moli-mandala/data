# Kurumba Appendix C transcription decisions

- Authority is the rendered scan, inspected cell by cell. The corrupt embedded
  OCR is a locating/comparison aid only and must never be copied as accepted
  text without independent visual reading.
- Preserve the source's phonetic notation exactly in `Manual_Form`, including
  diacritics, length, spaces, alternatives, and qualifiers. Apply NFC only when
  staging; house-transcription conversion belongs to the later sound profile.
- Use `attested` only for a visually read form, `blank` for a visually confirmed
  empty response cell, `ambiguous` when marks permit more than one reading, and
  `illegible` when no responsible transcription is possible. Never guess.
- Every non-pending cell records the exact physical/printed page, item, list,
  confidence, review method, reviewer, and any ambiguity. Ambiguous and
  illegible cells are also copied to `unresolved_readings.tsv`.
- The report's similarity judgments are synchronic lexical-similarity evidence,
  not etymological or cognate claims. No `Parameter_ID`, `Cognateset`, or
  `Etymology` is inferred from resemblance.
- The four comparison lists are retained as lexical attestations and explicitly
  marked `control`; they are not silently mixed with the fifteen target lists.
- On physical p.217 the wedge-shaped source vowel is transcribed `ʌ`; visible
  retroflex glyphs are transcribed `ɖ ɳ ɭ ɻ`, dental-marked `t` as `t̪`, and
  visible `ŋ ɛ ɔ ʃ ʂ` distinctly. Printed ASCII-like colons and raised
  apostrophes are retained literally as `:` and `'`; their phonetic
  interpretation is deferred to source-profile review.
- Physical p.217 associates Tamil `su:rjʌn` with item 19 `sky` and `vʌnʌm`
  with item 20 `sun`, despite the apparent semantic reversal. The printed row
  association is preserved and noted rather than silently corrected.
