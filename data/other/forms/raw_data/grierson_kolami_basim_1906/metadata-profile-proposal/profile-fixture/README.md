This is a preliminary coverage fixture for unaccepted readings, not an installed or approved profile. It covers all 561 physical specimen atoms, 224 independently read grammar witnesses, five current outside candidates and one unresolved alternative. Synthetic policy controls are counted separately.

The staged display profile lowercases and maps ṅ/Ṅ to ŋ, including decomposed input. Original is preserved exactly in the corpus. Quantity, nasal placement, ñ, optional parentheses, source punctuation, spaces and hyphens remain literal; the IPA-named profile column is the existing conversion API field, not an assertion of phonemic transcription.

`prepare_fixture.py` reproducibly pins inputs and generates the corpus, Unicode inventory and profile. `test_preliminary_profile.py` uses the real `segments.tokenizer.Tokenizer` conversion API on every form in NFC and NFD. It does not replace or patch the parser. Generic parser boundary punctuation trimming is a separate existing behavior; this direct-tokenizer fixture does not approve that behavior for final emitted forms.

Refresh from the final accepted corrected candidate before final profile acceptance, parser/routing validation or installation. No canonical settings, profiles or registry files are changed here.
