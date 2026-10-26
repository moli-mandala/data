# Guarded whole-source installation proposal

No canonical source installation is authorized by the existence of these files.
`install_whole_source.py --check` verifies exact reproduction of revision 2,
all 742 historical keys, and metadata continuity. `--install` additionally
requires `root-installation-approval.json` with the exact fields/hashes recorded
in `regeneration-check.json`, status `approved`, `material_errors: 0`, and a
nonempty `independent_review_sha256` mapping from package-relative review paths
to their verified hashes.

The immutable `legacy-742.csv` replaces reliance on the live canonical CSV for
regeneration. The frozen original assembler and exporter remain untouched.
Revision 2 applies only the two approved identity decisions. No graph edges or
IPA are manufactured. Auxiliary work/article claims stay explicitly secondary
through Niranjan, with source-language attribution unresolved where appropriate.

The metadata proposal changes only `defaults.importer`. Its route points here;
the old importer is retained solely as historical thematic-stage evidence and
must not be run to overwrite a whole-source installation. Shared bibliography
and language registry records are reviewed separately, never rewritten by this
coordinator.

After independent acceptance and root approval, run installation in the existing
local tmux pane. Then perform, sequentially:

1. Repeat `--install`; require byte-identical canonical CSV, YAML and audit.
2. Run `validate_whole_export.py --input-csv data/other/forms/20260921-niranjan-chakma.csv --preview-dir data/other/forms/raw_data/niranjan_chakma_2010/whole-installation`
   from the data repository. This invokes the real parser and records all
   lexical, Native, source, key, tag, metadata and registration checks.
3. Run the focused assembly/export/installation guard tests. Update historical
   installed-stage assertions only after preserving their historical fixtures.
4. Record canonical hashes, source-stage counts/exclusions and remaining gates
   in the source README/checklist/ledger. Database, full CLDF and browser gates
   remain deferred; this route never invokes them.
