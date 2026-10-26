# Proposed literal route for newly recovered evidence

This is a source-local recommendation, not an installed configuration. It
preserves the existing 4,458 rows and their conversion behavior.

## Routing

Move the existing `sil-korwa-kodaku` transcription rule from the
`behera2022korwakodaku` citation block to `defaults.transcription` in the same
legacy YAML file. Its old CSV still selects exactly that profile. Keep the
source-level `forms.split_alternates: false` and `notes.audit_only: true`.

Put newly recovered rows in a separate canonical file with file-default
`transcription: {convert: false, preserve_hyphens: true}`. The same real
bibliographic citation is used in both files. No invented bibliography key or
source-specific parser branch is necessary. SourceMeta tries source rules
before file defaults, so merely adding a new file default while retaining the
old source-level profile would not work.

## Scholarly rationale and fields

The source calls the narrative p lines phonetic transcription. Its exact
symbol distinctions must survive: i/ɪ, u/ʊ, a/ɐ/ʌ, r/ɾ/ɽ, nasalization,
shortness, explicit length, dental/laminal marks, aspiration, and source word
boundaries. The existing profile lengthens plain i/u and merges other vowel
qualities; applying it to all new evidence is not justified merely by its use
on the older wordlists. No silent repair of unusual o/u dental marks, nasal
marks on consonants, or literal g/ɡ is proposed.

- Raw CSV Form and derived Original preserve the reviewed source string, NFC
  only. The new display Form stays literal under `convert: false`.
- Leave the new Phonemic field blank with an explicit source-field reason.
  Printed p49 (PDF59) says both wordlists and transcribed texts are field
  transcriptions that have not undergone phonological or grammatical
  analysis. Form/Original already preserve this authorial phonetic evidence;
  no distinct phonemic layer is established. Outside prose romanizations also
  lack a separate phonemic analysis. Existing old Phonemic values are unchanged.
- Native remains blank unless a distinct native-script witness is printed.
- Source punctuation, internal spaces, hyphens, and repeat tokens remain
  literal. They are not stripped by a preservation route.
- Whole/group glosses and uncertainty reasons retain their source scope. A
  missing free translation is not filled using a neighbouring expression.
- Current interlinear Notes concern editorial marker recovery, gloss scope,
  or transcription review. Audit-only treatment is appropriate. Explicit
  grammatical labels are structured separately; raw analyses remain in audit.

The existing symbol census records 81 NFC characters, no PUA/replacement/control
or NFC defects, and the exact combining clusters. This establishes encoding
coverage, not a phonological interpretation. Refresh it after final corrections.

## Required focused proof before installation

1. Compare old and proposed SourceMeta transcription rules for both old
   languages and every old input row; old route remains the same profile.
2. Parse all old rows before/after with the real generic parser and compare
   Entry_Key, Form, Original, Phonemic, Native, Gloss, Tags, and references.
3. Parse the complete new CSV through its actual YAML. Assert exact reviewed
   Form/Original/Phonemic, no replacement characters, unchanged key count,
   no comma splitting, and source-leading/trailing hyphen preservation.
4. Verify the symbol inventory under NFC and NFD; preserve rather than delete
   an unexplained grapheme or source notation mark.
5. Keep identities and unrelated source/YAML behavior unchanged. Full CLDF/DB
   gates remain deferred under the user's current restrictions.
