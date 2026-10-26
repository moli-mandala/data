# Independent Koya wordlist notation recovery

This is a source-literal review proposal, not an installed source or approved IPA transcription. Canonical inputs, the parent's whole-source proposal, its failed audit, and the original PDF remain unchanged.

## Coverage and evidence

The complete target columns on PDF pages 82–112 were reread: JAG, CHI, POD, UTN, BHG, BHM, and MAL. All 1,420 ledger slots are accounted for: 1,419 physically printed target cells, including 1,401 nonblank attestations, plus the genuinely absent MAL200 slot. MAL61–80 are not printed and are not fabricated. TEL/ORI on PDF113–123 are controls outside this target-notation review; this review does not claim a new glyph-by-glyph transcription of those 420 control slots. The parent's narrative/comparative recovery remains a separate input.

The historical first readings remain in `notation-literal-pages-20260927.json`. The keyed derivative retains every previous ledger literal, first literal reading, source page/site/item/column, and applicable narrow or encoding decision. There are 904 literal differences from the earlier ledger, 102 narrow adjudications, and 20 explicitly keyed iota-encoding decisions. The 123 remaining first-reading notes describe source distinctions and recovery decisions; all specifically requested narrow checks have been completed. Typed uncertainties remain on 78 cells rather than being excluded or harmonized.

Original PDF: `tmp/pdfs/koya/JLSR2021_029.pdf`, SHA-256 `a6541e0d2397849ce7c36961b3849f3b2c1f1c267036cfa1a3f6025796e14e7d`. Printed folios equal PDF page minus five. Narrow records include exact crop coordinates and scales; source-local detail images preserve independently discussed glyph cases.

## Transcription decisions

- Preserve source single dots (`·`) separately from colons (`:`); the legend distinguishes lengthened from extra lengthened. No inferred duration or automatic doubling is approved here.
- Preserve raised vowels and aspiration, dental marks, source word boundaries, alternatives, and unparenthesized whole answers. Do not import older editorial parentheses or truncated suffixes.
- Encode the source's iota-shaped glyph as `ɩ` only in the 20 explicitly reviewed records. This is a literal-glyph decision, not an IPA conversion.
- Preserve uncertain raised round marks provisionally as `ᵒ`, with explicit uncertainty. The original strokes do not support a confident degree sign, unreleased consonant, or vowel interpretation.
- All 41 question-shaped-sign cells retain literal `?`/`ʔ` spellings with unresolved phonetic-versus-collector interpretation. This supersedes categorical claims in historical notes without deleting those notes.
- Keep specific alternatives for unresolved glyphs, including father-final beta-like/a, mortar diaeresis/tilde, MAL84 n/ŋ, BHM48 n/m, and handwritten t/ʈ. Do not infer phonemes or use neighboring witnesses to regularize them.
- Canceled material in POD186 is separately described; the visible replacement is provisionally `eɾund ᵘaɣt̪ɛ`, with ɛ/e uncertainty. Detached unexplained marks elsewhere remain explicitly recorded.

## Validation and integration requirements

`validate_notation_review.py` verifies exact target-key coverage against the ledger, preservation of old literals and locators, absent MAL200, no private-use/replacement/surrogate characters, normalization round-trip, explicit sign-interpretation uncertainty, null IPA fields, and representative dot/colon, cancellation, and glyph decisions. The character inventory is saved in `notation-validation-20260927.json`.

The canonical `sil-koya` profile lacks literal symbols `: ? · ä ē ɩ ᵊ ᵒ ᵘ ⁱ`; this inventory is not a profile approval. Integration must choose a documented literal-preservation route or source-supported mapping before parsing. Keep original source notation distinct from normalized display/IPA, and avoid assigning phonemic values to unresolved shapes.

The review-only `literal-integration/` now combines 1,437 corrected wordlist rows with the unchanged 210 narrative and 18 comparative rows: 1,665 active rows and 2,119 audit units. One old false split at UTN198 is reconciled through a proposed `:v2` to `:v1` alias; original `va:val` is one answer. The new explicit literal profile passes the actual parser with exact wordlist display preservation and blank Phonemic. All three files pass key/Original/Notes/citation/tag preservation, registered lect routing, source-key identity, delimiter-safe references, and closed variant edges. The historical failed freeze is hash-verified unchanged.

Pending gates: independent whole-output audit and alias acceptance; guarded installation. Full CLDF/database build, full suite, and browser QA remain deferred under the user's instructions. This handoff authorizes none of those operations.
