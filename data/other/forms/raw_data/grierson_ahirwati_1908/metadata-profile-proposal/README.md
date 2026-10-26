# Preliminary whole-source configuration fixture

These source-local proposals do not install or approve a lexical source. The
ingestion checklist remains active, including comparative-source, grammar,
native-script and interlinear requirements. The reviewed display convention is
lowercase plus ṅ/Ṅ→ŋ, including decomposed input. All other quantities, raised
letters, hyphens, nasal marks and punctuation remain literal. Phonemic stays
blank; the profile's legacy `IPA` column supplies display conversion only.
Source Original values and native witnesses remain separate and unchanged.

The 736-case fixture includes 98 grammar interpretations, the current No37
390-row proposal, 133 reviewed No38 physical Roman atoms, 23 proposed whole
spans, 18 quoted interpretations, all 62 outside interpretations and two
physical components of the Hissar whole example, plus ten policy controls.
These are conversion coverage counts, not counts of accepted lexical rows or
independent printed tokens. Audit-only controls remain in the inventory to
prevent missing symbols later. No37 now uses metadata-proposal-revision2;
the historical 389-row draft is not overwritten.

`prepare_proposal.py` pins the exact current inputs, stages one SourceMeta YAML,
one primary bibliography entry, five historical lect contributions, a cluster
inventory and the preliminary profile. `validate_proposal.py` uses the actual
SourceMeta constructor with existing canonical YAMLs plus the isolated proposal,
the registered language/dialect helpers, pybtex's production plain formatter,
and segments Tokenizer. It checks every corpus case under NFC and NFD and
enumerates each display delta. It does not monkeypatch the parser or claim an
installed `parse_file` result before final combined output exists.

The five lects are Gurgaon and mixed Jhajjar/Rohtak Ahirwati under existing
Ahirwadi, and Sirsa, Hissar and Bikaner Bagri under existing Bagri. Coordinates
and lect Glottocodes are blank. Modern Jhambaus coordinates and the Haryanvi
`hissar` dialect are not reused. Explicit Bangaru and Mewati comparisons retain
their registered base identities without automatic historical lect inference.

The one new reference identifies the actual 1908 volume rather than inventing a
chapter title. Its selected scope includes the complete grammar and both
specimens plus applicable Bagri comparison spans. Original title and 1927
title correction are distinguished. Later addenda and named contextual
authorities remain documentary evidence, not invented lexical attestations.
OCR was used for locating source leads; original-page manual transcription is
the source of proposed lexical fields, so OCR is false. Authorial borrowing and
morphology claims justify `etymology_provenance: source` without creating edges.

Pending gates: No37 metadata/output acceptance, No38 native and whole-boundary
acceptance, final combined output and refreshed symbol/pin coverage, guarded
importer creation and command registration, canonical CSV/audit provenance,
installation approval, installed parser/dedupe/reference tests, and all deferred
full-build/browser gates. The configuration deliberately has no fabricated
importer command. Input drift must be disclosed before refreshing receipts.
