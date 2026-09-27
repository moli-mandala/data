# Full Bhalesi source-stage recovery

The installed source stage covers Bailey 1908, Part III, the complete Bhalesi chapter on
printed pp. 68–75, together with every explicit Bhalesi lexical comparison
located on pp. 28 and 53–54. The shared introduction's isolated grammatical
endings on p. 54 and introductory p. iii are retained as audit evidence.
The original PDF has 358 pages; chapter PDF pages are 182–189. The original
is retained at `tmp/pdfs/bailey-sainji/bailey1908.pdf` with SHA256
`953f5da5ee9bc341cdf7eb558920e824dc6f15f7473a8c0d5c8134c401ad60b5`.

The installed source stage has **435 rows from 436 editorial source units**: 410 target
units, seven other-lect comparison controls, and nineteen isolated morphology
units. Every one of the 34 glossary lines and all 22 numbered sentences are
accounted for. The remaining target material comprises full paradigms,
directly glossed grammatical examples, and explicit shared comparisons.
Twenty-five alternate answers expand their source units into additional rows.
All sixteen former pilot keys survive. Existing cross-source rows are not
removed or rewritten.

The source checklist is active, including the historical survey/comparative
table and source-comparison addenda. `whole-source-scope-20260926.json` gives
the complete census. `shared-context-audit-20260926.json` records the broader
search, explicit target attribution, other-lect paragraph controls, geography,
and historical linguistic claims. The first chapter reading is preserved;
`full-reviewed.jsonl` and the second-reading correction log are authoritative
for the proposal. All eight chapter pages and additional target passages
received a second visual reading from the original. OCR text was used only
to locate references, never accepted as lexical transcription.

Source spelling is retained, including variation between repeated forms,
vowel length, breves, stacked nasal marks, retroflex dots, and underlined
consonant groups. Printed stem abbreviations are expanded only against their
explicit source stems; the audit retains source shorthand and first-reading
provenance. Unlabeled six-position verb paradigms retain their printed order
without invented person assignments. The author's general future-person
summary is preserved as an attributed statement. Agent case is represented
by the established `erg` tag with the original label in Notes; imperfect
indicative uses `ipfv`, not a speculative progressive analysis. Whole
sentences retain their punctuation and are not split into invented lemmas.

The book's contrastive type is meaningful. Per-answer Notes and audit spans
record italic `eu` within roman words, or roman `eu` within italic words,
as the author's shortened vowel. The long continuous-bar `e͞u` is distinct
from separately barred `ēū`. No IPA is invented. The literal display profile
maps `ṅ` to house-style `ŋ` and removes terminal sentence punctuation only
from normalized Form; Original retains the literal source text. The marked c in
`kaṇčā` has a source-specific profile-policy exception because the source gives
no secure phonetic interpretation of its additional mark; the global č rule
is unchanged.

Six source units carry typed uncertainty: four compact nasal i length marks,
the initial i length in the sister form provisionally read `bīnyi`, and the
Unicode attachment of the tilde over the woman's continuous-bar `eu` span.
For the last item, the visible span bar, separate tilde, dot above n, and
underlined sh are all preserved; only single-character attachment is uncertain.
These are explicit uncertain readings, not undisclosed omissions. The woman
attestation's derivative LSI/Zoller overlap remains documented without
claiming an independent field record or dropping Bailey's attestation.

Nine focused staging and installed tests pass, covering full accounting, preserved keys, explicit
variation and typography, grammar, shared-source scope, all profile graphemes,
registered tags, real citation parsing, and scoped parsing of all 435 rows.
The independent sample is reproducible with `sample_full.py`, which checks
all frozen input hashes and excludes earlier samples. The earlier chapter-only
sample was interrupted before a verdict when shared comparisons were found;
its selection and freeze manifest are preserved transparently. The expanded
pass 2 found one sampled error; the resulting consonant-class review corrected
three forms, recorded in `doubled-consonant-class-review-20260926.json`. The fresh disjoint pass 3 sample passed all twenty units; the root reviewer also
verified all eleven shared-source additions directly from the original.

The exact independently reviewed CSV, audit and profile are installed. Bibliography,
YAML scope, importer, profile registration, source checklist, and completeness
ledger are reconciled. The former bounded pilot is preserved under `legacy-pilot/`. The existing `bhal` language
identity remains unchanged. Bailey locates Bhales valley a few miles east of
Bhadrawah town, in eastern Jammu proper; he gives no exact elicitation site.

No database or full CLDF build, full test suite, compiled identity/graph
verification, or browser QA has been run. Those gates remain explicitly
deferred under the user's no-build/no-remote instruction. This proposal is
not a claim of application-visible ingestion completion.

Regenerate the exact reviewed source stage from the data root with
`.venv/bin/python data/other/forms/raw_data/bailey_bhalesi_1908/import_source.py --check-pdf --install`.
Installation verifies the independent audit hashes and writes only source files.
