# SIL Bhumij 2015 manual ingestion

The authoritative manual ledgers cover all items 1--210 on physical PDF
pp.34--76 / printed pp.29--71: 3,780 cells, comprising 3,690 attestations and
ninety explicit source blanks, producing 3,876 responses and 2,100 target form
candidates.
Every cell was keyed while viewing the 400-dpi renders and visually matched
before acceptance; the Unicode text layer supplied or verified no reading and
was used only to locate table structure.

Source-local extraction and manual review are complete: zero cells remain
pending or unresolved, and the importer now accepts `--stage`. No form is yet
installed. The five 1989 lists reprinted as Ho 2024 controls are exhaustively
reconciled in `ho_2024_overlap_reconciliation.tsv`: all 1,050 conceptual cells
have status parity, and the Bailey & Maggard 2015 rendering is the canonical
publication route. Shared metadata/profile/reference/build/browser gates remain
deferred until the active survey packages finish.

`list_registry.tsv` gives all ten target lists durable identities under the
Mundari ISO parent `unr`; Udala keeps the report's explicit mixed/uncertain
`Mundari? Bhumij?` label. `staged_forms.tsv` contains 2,100 target forms with
unique entry keys, `staged_audit.tsv` accounts for all 3,780 source cells, and
`profile_inventory.tsv` confirms that `conversion/sil-bhumij.txt` covers all
53 characters used in target forms. The source-local bibliography record is
`reference.bib`.
