# Bhumij 2015 integration status

Source-local extraction and overlap reconciliation are complete. No shared
registry, conversion profile, bibliography, generated CLDF, or browser database
has yet been changed for this source.

The five 1989 locality lists republished as Ho 2024 controls are the same
elicitation events: locality, date, speaker initials/sex/age, and recorder all
match. `overlap_registry.tsv` assigns stable list identities and
`ho_2024_overlap_reconciliation.tsv` accounts for all 1,050 duplicated cells.
All statuses agree (1,039 attested and eleven blank); 221 representations are
Unicode-exact after removal of Ho's similarity label and 818 differ because the
two publications use different diplomatic transcription conventions. Install
the Bailey & Maggard 2015 forms and retain the Ho 2024 versions audit-only.

Before installation, define stable source/dialect identities for the other five
target lists, retain eight comparison lists audit-only, add a source-specific
sound profile and bibliography entry, then run focused tests, the consolidated
full build/audit, generated-diff review, and representative browser QA.

Those source-local artifacts now exist. `list_registry.tsv` assigns all ten
targets to ISO `unr` with distinct durable dialect IDs; this follows Glottolog's
placement of Bhumij under Mundari. Udala remains a distinct mixed-labelled
route because the source itself prints `Mundari? Bhumij?`. The eight comparison
lists remain audit-only.

`staged_forms.tsv` has 2,100 forms from 2,054 attested target cells, expanding
the source's separately numbered variants. `staged_audit.tsv` accounts for all
3,780 cells: 2,054 attested target cells, 46 target blanks, 1,636 attested
controls, and 44 control blanks. Similarity-group numbers remain in notes only
and are never interpreted as etymological cognates. `conversion/sil-bhumij.txt`
covers all 53 source characters used by staged targets, and `reference.bib`
contains the source-local bibliography record.

Shared registration, installation, consolidated build/audit, generated-diff
review, graph review, and representative browser QA remain deferred until the
other active survey packages finish.
