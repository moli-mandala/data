# Source-local preservation profile

The accepted Ho survey layer is already a manually reviewed diplomatic Unicode
transcription. The proposed display conversion is therefore NFC identity after
removing only the printed lexical-similarity group labels. Word boundaries,
commas and source alternatives, glottal/question marks, length, aspiration,
retroflexion, nasalization, and the legacy survey-alphabet distinctions are
preserved. No blind phonological character replacement is authorized.

`symbol_inventory.tsv` enumerates every character in the 2,900 staged target
forms and records its preservation decision. Shared routing in `utils.py` and
`make_cldf.py`, and the corresponding global sound-profile test, remain deferred
to the consolidated integration pass described in `INTEGRATION.md`.
