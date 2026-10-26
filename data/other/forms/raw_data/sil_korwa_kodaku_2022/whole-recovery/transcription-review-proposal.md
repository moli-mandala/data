# Proposed transcription policy for whole-report additions

The existing wordlist file retains its established `sil-korwa-kodaku`
conversion. Its reviewed lexical repairs use that same route. A separate
additions file can use file-level `convert: false` while citing the same
single canonical bibliography key. The isolated routing check proves that
moving the old profile to its old file's defaults preserves all 4,458 old
row settings; it does not by itself approve these proposals.

For additions, preserve the source's phonetic notation literally in Form and
Original, normalized to NFC. Do not infer a distinct Phonemic layer: printed49
explicitly calls the wordlists and texts field transcriptions without
phonological or grammatical analysis. Prose romanizations such as Kora also
have no independently justified phonemic conversion. Native remains blank
where no native-script witness exists.

The complete current snapshot census includes 1,616 comparison candidates,
117 whole expressions/questions/answers, 192 column groups, 790 continuous
gloss spans and 48 outside witnesses. These counts overlap physically and
include held/documentary material; they are not installed or unique lexical
counts. It contains 81 distinct NFC characters and no PUA, replacement,
control or non-NFC issue. Input hashes and per-symbol examples are recorded
in `integration-proposal/symbol-inventory-status.json` and
`integration-proposal/symbol-inventory.json`. Rerun after candidate changes.

Preserve vowel-height distinctions i/ɪ and u/ʊ, explicit length ː,
nasalization, breve, dental and other diacritics, script g, rhotic types,
aspiration and word boundaries. Printed47 note8's equivalence convention
for ə/ɐ/ʌ is an attributed source statement, not a reason to erase the
different literal witnesses. Do not extend legacy i→ī/u→ū mappings to every
new transcription without additional evidence.

The source-looking `sa.b` in C.4 unit6 needs independent glyph review before
acceptance; a literal preservation route cannot certify that punctuation is
correctly extracted. The slash-separated community names are documentary
witnesses, not silently expanded target-language variants. Independent
source/alignment review still controls lexical acceptance.

This is the proposed conservative route, pending the independent reviewer.
The complete routed comparison output already passes the actual parser in
NFC and NFD with all 15 fields and ordered keys retained. Text and final
combined corpus checks remain required. No canonical profile or YAML has
been changed by this proposal.
