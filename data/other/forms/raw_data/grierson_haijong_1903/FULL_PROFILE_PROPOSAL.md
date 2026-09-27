# Haijong full-source profile proposal

`full-profile-proposed.txt` covers all 896 reviewed candidate rows returned by `prepare_full_source.prepare_roman_assembly()`. It is a display-preservation profile, not reconstructed IPA. The column name `IPA` is the existing tokenizer interface; it does not license populating `Phonemic` with these values. The canonical profile is unchanged.

The source's general transcription discussion (LSI V.1, preliminary pp. iii–iv; cached original-volume OCR headed “SYSTEM OF TRANSLITERATION ADOPTED”) distinguishes transliteration, phonetic notation, nasal marks, and raised letters. The Bengali introduction also discusses nasalization and consonant compounds. The Haijong chapter pp.214–215 supplies grammatical examples, not a complete new pronunciation key. These statements do not justify assigning a new phonetic interpretation to every unusual glyph in these specific witnesses. In particular, the three raised s-shaped glyphs remain under explicit transcription uncertainty in the source audit. Corrupt OCR of the general symbol key is not used to invent mappings.

Mappings:

- Lowercase source capitalization for display. Preserve every source capitalization in `Original`.
- Map literal `ṅ` to house `ŋ`, and `w` to house `v`, as required by `data/profile_policy.py`. These are conventional display mappings, not additional historical or phonemic claims.
- Preserve literal macrons, breves, carons, diaeresis, underdots, nasal tilde and low-line clusters. Distinct `ă`/`ǎ`/`ā`, `ï`/`i`, underlined `n̲g̲` versus `ṅ`, and plain `z` versus `ẓ` remain distinct. Do not reinterpret underlined `t̲s̲` or `s̲h̲` through an inferred phonetic value.
- Preserve raised `ʸ`, `ᵛ`, and uncertain `ˢ` literally; do not expand or delete them. Source uncertainty remains attached to the corresponding rows.
- Preserve spaces and meaningful hyphens. Set `preserve_hyphens: true` in source YAML.
- Remove sentence-final `.` and `?` only in display; all 17 occurrences are verified terminal, and exact punctuation remains in `Original` and source evidence. The profile has punctuation rules; the focused verifier rejects any future nonterminal occurrence so it requires editorial review rather than silently deleting internal punctuation.
- Normalize to NFC before and after tokenization. Rules cover whole combining graphemes, preserving low-line and tilde sequences.

Run `data/.venv/bin/python data/data/other/forms/raw_data/grierson_haijong_1903/verify_profile_proposal.py` from the workspace root. It regenerates only this source-local proposal/report, tests all candidate rows against the declared transformation, checks 13 consequential examples, and calls the shared house-output policy for every rule. All 896 forms pass; 77 observed clusters/rules, zero uncovered glyphs, zero shared-policy violations. The report pins the proposal and source key/form inventory hashes and confirms the old canonical profile hash. Final canonical integration and rechecking changed assembly inputs belong to the source owner. Database/full-build/browser gates remain deferred under the user's explicit no-build instruction.
