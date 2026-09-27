# CUJ Asur transcription decisions

The dictionary labels its Roman layer `fonipa` but uses c/j for affricates and y
for the palatal glide. Compare dictionary juɽu (p.38) and jom (p.39) with Khalid,
*A Phonological Sketch of Asur* (2020), pp.261 and 267: /ʤuɽu/, /ʤom/.
The pinned paper is identified in manifest.json. It uses t/d for dental stops
(p.253 n.1), and explicitly lists y and w as glides (pp.259–260).

- Raw Form and Phonemic retain the printed Roman head; Native retains the
  mechanically decoded Devanagari. Original is supplied by the build from raw Form.
- c/j/y retain their dictionary values. w → v; ʈ/ɖ/ɽ → ṭ/ḍ/ṛ.
- Vː → macron, mː → mm. Printed nasalization survives, including ãː → ā̃.
  The paper treats length/nasalization as nonphonemic; this does not authorize
  deleting distinctions present in this dictionary's phonetic transcription.
- ʔ, ʰ, and nasal vowels remain. The rare superscripts ᵍ (oɽeʔᵍ, p.15)
  and ᵏ (kʰokᵏro, p.24) are preserved and flagged for transcription review.
  The paper describes unreleased g but does not establish the dictionary's
  precise use of these superscripts; no deletion or phonemic expansion is made.
- Hyphens and equals signs survive as printed affix/clitic boundary notation.
- Entries lacking Roman transcription retain native spelling in Form/Original
  and Native, with Phonemic empty. Identity Devanagari rules are preservation,
  not an invented transliteration or inferred pronunciation.
- Eight printed dotted-circle sequences remain visible and uncertain. CID127
  is unresolved and its headword-only record must remain audit-only.
- No dialect is inferred from the university location. Existing base Asuri is
  reused; the source does not specify one uniform named variety for the whole work.

The source YAML now routes installed rows through this profile. The actual
parse_file path preserves all 2,106 rows and source layers with zero conversion
errors. Scoped house-policy checks pass. Final compiled-layer verification
remains outstanding until the consolidated full build.
