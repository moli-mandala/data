Every `*.csv` directly in this directory is ingested as a lexical source. Its per-source settings
(transcription profile, dedupe rule, excluded languages, reference metadata, …) live beside it in
`<same stem>.yaml`; the schema is documented in `source_meta.py` at the repo root and the file is
required for every source. Curated etymology
decisions for a source live beside it in `etymologies/<same stem>.csv` (schema
`Form_ID,Etymon_ID,Kind,Rank,Status,Source,Notes,Pos`, keyed by persistent form ID); see
`etymology_assignments.py` at the repo root. `etymologies/_pending.csv` is an inbox that the build
files automatically.

Format for the CSVs:
- Language ID
- Param ID
- Form
- Gloss
- Native script
- IPA
- Notes
- Source

\d+.*?[ɦh]\n