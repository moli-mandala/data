# LSI 1916 Śōrāchōlī: complete source stage

The active ingestion checklist covers the survey/table, grammatical paradigm, and historical page-image addenda. The full source is the dedicated chapter on printed602–609 plus every Śōrāchōlī column cell on odd printed629–645. Adjacent Barari601 and Kirni610 are controls. The original998-page `tmp/pdfs/lsi-v9-4/LSI-V9-4.pdf` has SHA256 `ef007663270b3a0ef5ba26804404e4db1281fb5eb239614cadf69b20b3d5395f`; printed page+16 gives its one-based PDF page. No remote resources or database build were used.

The installed source stage accounts for683 units and yields728 forms:

| Section | Source units | Installed forms |
| --- | ---: | ---: |
| Unusual words602 |27|27|
| Grammar602–604 |108|172|
| Comparative table629–645 |241|292|
| Aligned specimen607–609 |307|237|

All units received a complete second page-image reading. Corrections are recorded in `full-table-second-review-20260926.json`, `table-final-a-review-20260926.json`, `grammar-list-second-review-20260926.json`, and `specimen-second-review-20260926.json`. The four reviewed TSVs are authoritative; the alignment TXT files and old `transcription.tsv` retain earlier readings as evidence and are not importer inputs.

The native-script605–606 passage matches the complete Roman/English607–609 sequence by narrative blocks; see `specimen-witness-reconciliation-20260926.json`. This is a source-version reconciliation, not a diplomatic native-script transcription. No separate duplicate lexical output is inferred from the native presentation. The307 aligned surface attestations retain individual locators;70 exact same-form/same-gloss repeats reuse one entry with every locator. No casefolding or cross-section merging is performed.

Three table cells22/174/201 are explicitly blank. Eight grammar units are source control forms, counterfactual spellings, or bound endings and remain inventory-only. Unusual-word head8 is fully represented through its explicitly translated construction `khāyŏ chhĕkṇū`; its bare head is not assigned the whole phrase's meaning. The27 unusual heads are all accounted for. Earlier generic typography/phrase holds and speculative Zoller-derived exclusions are superseded; similar later-source forms do not prove dependence.

Three units carry specific uncertainty: the capital-U nasal/length shape in table75, the final i/ī of `Bāṭhṇī` in table130, and the small raised mark beside ṭ in unusual-word27 `uṭī`. Independent root typography review supports ī in `chhēwṛī` at128/130 and plain i in `chhiṭē` at228. Literal page-specific spelling differences remain. The source leaves `āsū sū` adjacent without a comma in table156; that sequence is retained, while explicit comma-separated predicates receive the shared printed subject. Source grammar and explicit paradigm person/number/case/tense are structured without reconstructing lemmata or relations.

The canonical language is existing `Shoracholi`. The printed locale is the Thakurāte of Rawain, Keonthal State, east of Barari Pargana. No narrower consultant locality, coordinate, or dialect split is invented. The conservative profile preserves printed length, nasalization, breve, and underdot distinctions in normalized form, lowercases capitals, applies house w→v and ṅ→ŋ, and removes editorial parentheses/question marks/full stops while retaining Original. It does not claim reconstructed phonemic IPA.

`preview_full_source.py` writes only `full-preview.csv` and its audit. `import_source.py --install --check-pdf` reproduces the final canonical source and audit after independent review. The existing11 entry keys are preserved. Full database/CLDF build, compiled reference/graph validation, full suite, and browser QA remain explicitly deferred under the user's no-build/no-remote instructions. Fresh independent audit pass2 (seed2026092662) passed0/20 material errors after correcting table177 to literal Piṭda and reviewing final-a marks across all table pages. Canonical installation reproduces728 rows; four focused tests include complete parser/profile/reference-locator checks and pass. Source metadata validates. The source stage is complete; full ingestion remains subject to the deferred build gates.
