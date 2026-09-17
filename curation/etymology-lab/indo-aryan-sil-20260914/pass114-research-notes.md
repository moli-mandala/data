# Pass 114: Hindi vegetable variants

Thirty Hindi control/comparison records selected: sixteen fulgobi/phulgobi cauliflower and fourteen bandgobi/bandigobi cabbage. Every relation is variant to an existing Hindi lexical entry. The whole compound is preserved.

The cauliflower Hindi cell was visually rechecked in vegetable-primary-page-11.png, item 4. Cabbage uses the previously preserved CSTT primary transcription; no fresh CSTT PDF verification is claimed. Source evidence is copied into pass114-primary-articles.json. Bandigobi retains its epenthetic i and the note distinguishes that pronunciation inference from the glossary spelling.

Unresolved leads: jaun ɵhulgobi requires source-glyph checking, and band tomato cannot be assumed to mean cabbage. Pattāgobi compounds have a clear source attestation in the same vegetable image but lack a ready whole-word Hindi entry in the inspected canonical nodes; no donor node was added. None of these remaining forms was adjudicated solely from spelling resemblance.

## Validator change and verification

The first attempted save was rejected before mutation because validate_assignments omitted variant from its allowed kinds, despite apply_assignments and the graph schema already supporting it. The validator now accepts variant. A regression test checks replacement of a prior ancestry edge, clearing unlinked status, and zero changes on repeat application. Focused test file: 17 passed, 1 previously recorded hard-coded corpus-count failure (2603 versus 2604); no full-suite success is claimed.

The subsequent save succeeded: 30 rows, 60 temporary-graph changes, repeat zero. Prior overlay rows and unrelated edges preserved; compiled forms/edges and identity registry unchanged.
