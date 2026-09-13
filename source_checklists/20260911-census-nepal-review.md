# Census and Nepal survey ingestion review — 2026-09-11

The mandatory source-ingestion checklist is active. Applicable addenda: survey wordlists/comparative tables, plus external dataset for the archived Tharu XLSX. This is an ingestion of lexical attestations, not an etymologisation. No browser refresh or deployment was requested.

## Scope and acquisition

Six sources are represented by `data/other/forms/raw_data/census_nepal.py`, its adjacent `census_nepal_2026/` acquisition manifest, source-cell snapshots, mappings, image transcriptions, and per-cell audits. Public original PDFs/workbook are cached but ignored by Git; hashes, exact download URLs, source cells, and extraction code remain reproducible. No explicit open redistribution licence was found; the installed material consists of attributed lexical facts. The Tharu workbook was recovered from the Internet Archive's 2015-11-09 capture of SIL's download. Its canonical SIL catalogue dates creation to 2011–2013 and labels it an unreviewed draft.

Danuwar uses Regmi and Thakur's correctly titled January 2016 edition. A second government-hosted PDF has Darai frontmatter but the same Danuwar body; it is excluded as a duplicate/mislabelled acquisition. Existing Shackelford Danuwar records represent different sites. Tharu overlaps the existing Kochila import; exact lect/gloss/transcription matches add citations to existing records. The workbook and publication use different numbering after item 239; matching their numerical IDs would be wrong. Western Tharu survey and Proto-Tharu sources already present are separate attestations.

| Source | Raw cells | New rows | Excluded cells | Reused readings |
|---|---:|---:|---:|---:|
| Tamil Nadu | 5,988 | 5,995 | 106 | 0 |
| Uttar Pradesh | 5,988 | 5,931 | 170 | 0 |
| Bihar | 3,500 | 4,177 | 117 | 0 |
| Sikkim II (Nepali) | 1,500 | 1,729 | 8 | 0 |
| Danuwar | 1,050 | 1,051 | 0 | 0 |
| Tharu | 2,360 | 1,520 | 118 | 761 |

Total: **20,403 new rows**, 761 exact republished readings with additional citations. Tamil Nadu and Uttar Pradesh omit item 94 in print: another 12 absent cells each are explicitly audited. Alternative readings account for row expansion. Tharu's excluded-from-similarity flag does not suppress an otherwise valid lexical attestation. Its lexical similarity group numbers remain source-local annotations, not inferred ancestors.

## Language and transcription decisions

Five canonical language registrations: Katkari, Paniya, Saurashtra, Kurmali, and Maliyad. All 47 source lects/sites receive language-qualified dialect registrations. Source-locality descriptions are retained; coordinates remain blank where precise evidence was not obtained. Maliyad remains the source's named South Dravidian language without an asserted Glottocode or forced identification. Chetti Bhasha is provisionally under Kannada following its chapter's classification. Kathodi follows its chapter's Katkari/Indo-Aryan identification despite the contradictory table heading. Erode Urali remains distinct as a dialect from the existing Idukki sample, with provisional mapping flagged. Mainpuri maps to Braj; Giharo and Pratapgarhi to Awadhi; Gurumukhi to Punjabi. Danuwar applies the existing Done/East Danuwar distinction provisionally by site.

Census PDFs needed structural cross-reference repair before pdfplumber could read their otherwise selectable text. Numeric row and column boundaries are preserved. All 124 Sikkim and 24 Tamil Nadu cells containing image fragments have explicit manual transcriptions from 220-dpi page crops. No OCR software contributed. Faint Tamil Nadu image glyphs retain transcription review markers.

Three explicit sound profiles preserve Original and convert only documented display conventions. Census IPA maps ordinary IPA symbols to house notation. Census ASCII maps documented vowels, retroflex stops and aspiration; ambiguous R/M (conflicting definitions in Bihar's symbol table) remain literal. Danuwar's apical diacritics are preserved rather than silently read as dental or retroflex. There is no independently supplied phonemic layer to duplicate into Phonemic. Source blanks, X/underscore placeholders, Excel #VALUE!, and unrecoverable empty readings remain audit exclusions.

The Census comparative tables demonstrably contain source gloss mismatches (including shifted Paniya cells), internally inconsistent phonetic notation, and wrapped words/phrases. These source pairings remain `uncertain` with typed reasons. Clear continuation fragments and reviewed examples are joined; ambiguous longer boundaries remain spaces with transcription review markers. **This is a conservative, review-flagged import, not a claim that the published tables are linguistically clean.**

Parenthetical gender/case labels are structured into tags, English qualifiers remain in glosses, and residual unclear abbreviations remain notes. Parentheses grouping alternative forms never produce punctuation-only rows. No ancestry is inferred; generated forms remain unlinked.

## Audit and validation

The initial seeded 20-cell comparison for each source found wrap-boundary issues in Tamil Nadu and Bihar and confirmed the Danuwar, Sikkim and Tharu cell readings. Follow-up edge-case inspection caught punctuation-only alternatives and embedded English annotations; regression tests cover both. A fresh sample uses seed 20260914. Residual source inconsistencies and unresolved wrap boundaries are retained explicitly; no 0/20 linguistic-cleanliness claim is made for these malformed Census tables.

Build, full-suite comparison, persistent-ID checks, reference checks and final sample results are recorded below after validation. Baseline before this batch: 746,336 forms; 31 pre-existing full-suite failures (1,773 passed, 18 skipped). The complete generation pipeline previously succeeded before its final manual-survey-etymology gate failed two existing assertions.

Browser database construction/app inspection: inapplicable until a refresh is requested. No commit, push or deployment is included.

### Full-build inspection

The seven generation stages completed and produced **766,739 forms** (+20,403). All 746,336 previous IDs and non-citation content are unchanged. No citation was lost. There are 742 existing records receiving the 761 workbook citations (some records have multiple matched readings). Existing curated Kochila graph links remain intact. Edges and alignments are exactly unchanged; source keys and ID aliases each gain 20,403 rows. The identity registry updates 742 existing citation-bearing fingerprints without changing their IDs. All six references have inclusion/provenance/editor metadata. The new sources introduce no replacement-character conversion errors.

`make all` ends at its pre-existing manual-etymology gate: the same two failures as the baseline. The final focused run passes **39/39** tests, including new-source, existing Kochila, profile and dialect checks.

Concept-index regeneration replaced 2,669 links and added 21,481 (net +18,812). This was investigated: pysem builds a set of candidate matches and sorts only by similarity/POS/frequency, leaving equally ranked concepts dependent on Python hash order. Isolated runs with hash seeds 1 and 7 map `praise` and `stable` to different concepts without any source-data change. This confirms that the heuristic mapper can change old concept links even without lexical-content changes; no claim is made that every changed concept assignment is an improvement. No manual concept substitutions were applied.

Fresh seed 20260914 review: 20 cells per source inspected. No shifted-column, lost-cell, punctuation-only-form or annotation-as-form error remains in those samples. Tamil Nadu and Bihar still exhibit the explicitly flagged boundary/letter-spacing ambiguities documented in `sample-review.json`; these are not represented as a clean 0/20 transcription audit.

### Representative compiled records

These IDs are ready for the next browser refresh; this task did not refresh the browser database.

- mitchell-eichentopf2013tharu: `f_ihqakdazvihxy` — KochilaTharu **āŋ**, “body”.
- census2023tamilnadu: `f_agbpbw4v3ow5c` — Tamil **kārru**, “air”.
- census2023uttarpradesh: `f_d7s4fyi2fi3am` — H **vayʊ**, “air”.
- census2020bihar: `f_alw7eiunokh2i` — H **hawa**, “air”.
- census2012sikkim2: `f_v5bxqcsusohjm` — N **hawa**, “air”.
- regmi-thakur2016danuwar: `f_6rbicd5xihi4s` — DewasDoneDanuwar **jiu**, “body”.

Full re-extraction from the original PDFs and XLSX reproduced all six installed CSV files byte for byte (`reextraction-check.json`).

### Final validation result and remaining gates

- Full suite: **1,790 passed, 18 skipped, 33 failed** in 500 seconds. Failure-set comparison identifies exactly the same 31 pre-existing failures plus two test assumptions introduced by citation reuse. Those two were corrected (multiple primary/supplemental citations and already-curated reused graph nodes); the final affected test suite passes 39/39. The complete suite was not rerun after these test-only corrections.
- All seven data generation stages pass; the final `make all` test gate still has the two baseline manual-etymology failures. Therefore the checklist's fully clean build/full-suite gate remains open; this batch is installed, not certified as a fully clean ingestion.
- Checklist artefacts are current (`audit_source_ingestions.py --check`, exit 0). The retrospective checker continues to show its pre-existing workspace-wide full-pipeline gate as incomplete.
- Full original-document re-extraction and offline snapshot-only reproduction both yield byte-identical installed CSVs.
- No newly inferred inherited/borrowed/variant graph edges. Unlinked records and source-local lexical similarity annotations are retained. Source ambiguity remains typed and reviewable, particularly the Census tables' wording and transcription.
- Logs and exact old/new comparisons are under `source_checklists/audits/20260911-census-*`. No browser rebuild, publication, commit or push was performed.
