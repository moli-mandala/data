# Source completeness ledger (2026-09-25 ingestion batch)

The accompanying TSV enumerates **72 installed `20260925-*.yaml` packages**
(70 at the checkpoint before the new complete-source directive, plus two
complete-section candidates added afterward). The live status counts should
be read directly from its `completion_status` column: agents are closing
partials and classifying unverified packages concurrently. A passing bounded importer test is not evidence that the
underlying source lect's lexical section was exhausted. The `scope_flags`
column is an automated README search cue, not a scholarly determination;
41 packages have at least one cue. Packages without a cue also need review.

A reviewer must identify the complete relevant lexical section, account for
all printed/archived records (including blanks, controls and holds), and
reconcile any rows already installed by a pilot. Mark `complete_source_stage`
only when the full section is accounted for and source-side extraction,
identity, sound, reference and focused checks pass. Mark `confirmed_partial`
when a README or source inspection identifies remaining lexical material;
fill in the exact pages/columns/items in `remaining_scope_or_exclusion`.
Distinct grammar, paradigms, narratives, and other lects may be excluded only
with an explicit reason. A source-stage completion does not waive the user's
separate prohibition on database generation or the deferred full pipeline.

This ledger covers the newly dated packages, including its initial 70-package checkpoint.
`20260925-prior-source-completeness-ledger.tsv` separately enumerates the
198 earlier YAML packages. Its status counts likewise change during this
review; most earlier YAML files do not
declare an importer command, so their source directory and completeness
cannot be inferred automatically. The user has explicitly required us to
close all partial ingestions; these two inventories are a starting audit,
not a declaration that any unverified package is complete.


## Literal transcription recovery correction (2026-09-26)

Exhaustive cell accounting alone does not establish completed extraction. A
package with broad typography placeholders remains `confirmed_partial` when
readable source forms have not been recovered. Preserve legible source marks
literally when their phonological interpretation is uncertain; reserve holds
for specific unresolved source readings or scope/semantic ambiguity. A passing
audit sampled only from accepted rows does not validate excluded rows. Rohru,
Baghi and North Jubbal were reopened on this evidence; their accepted rows and
passing accepted-only audits remain valid subject to documented corrections.
