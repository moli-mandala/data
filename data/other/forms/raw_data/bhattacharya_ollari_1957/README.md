# Bhattacharya 1957, Ollari: acquisition and OCR scaffold

Status: **not installed**. The source is selected for underrepresented Central
Dravidian coverage, but this directory is not a completed ingestion.

The current compiled data contains 60 Ollari Gadaba rows; the manual inputs
also contain 60, all from DravLex. The canonical language is `OllariGadaba`.
The 1957 monograph provides a full comparative vocabulary on printed pp.48–77,
PDF pp.59–88. Its bibliography is printed p.78 / PDF p.89. Front matter describes
fieldwork in 1951 and 1955. Printed p.8 / PDF p.19 identifies the collection villages as Lamptapuṭ,
Munḍagaṛ and Koṭri. Use the base language: the vocabulary provides no
per-entry village assignment or uniform named dialect, and no point
coordinates have been invented. Introduction pp.1–8 were visually read.

The Tamil Digital Library record provides one PDF twice, not two editions.
The 93-page scan has **zero native text characters**. Its title page says
Memoir No.3, 1956, with publication imprint 1957; preserve both dates correctly.
The exact PDF URL, SHA-256, catalogue URL, scan date and retrieval date are pinned
in `manifest.json`. No open-data licence is asserted; the PDF is kept in scratch.

A newer alternative was checked: Mendem Bapuji's 2019 Hyderabad thesis URL
redirects to a missing `/chamo/` file and returns HTTP 404. The publisher's
2025 Bapuji–Mohanty revised grammar is verified bibliographically, but no public
full text was found. It was not purchased. Do not describe either as inspected
lexical content. The older scan is an accessible primary source in its own right.

## OCR is evidence, not transcription

`ocr_source.py` verifies the pinned PDF and processes the complete 30-page
vocabulary with pdfplumber at 300 dpi and Tesseract 5.5.2, English, PSM 3,
`OMP_THREAD_LIMIT=1`. It retains raw text, word-coordinate TSV and page hashes.
One page and one OCR thread run at a time. No CSV is emitted. All 30 pages
are now present: 10,094 OCR word tokens, with every text/TSV hash verified.
Twenty focused acquisition/evidence-regression tests pass; they do not certify the OCR
transcription or establish a headword count.

The pilot on printed pp.48–49 recovers reading order and English paragraphs,
but loses phonetic vowel length/diacritics and misreads grammatical labels.
Those are known systematic OCR defects: this output **must not be installed**
without a reviewed transcription layer. Uppercase headwords and italic compared
forms must not be folded together. Comparative forms and source borrowing claims
need their own analysis; no ancestry links may be inferred just from similarity.

Reproduce from the data repository:

```sh
.venv/bin/python data/other/forms/raw_data/bhattacharya_ollari_1957/ocr_source.py PATH_TO_PINNED_PDF
```

## Boundary and transcription pilot

`segment.py` retains word boxes and proposes **657 candidate boundaries**, not a
verified inventory. `candidate-entries.jsonl` is reproducible from the pinned
OCR. The margin heuristic is sensitive to scan skew and outliers; eleven missed boundaries have been repaired by visually documented overrides
(one each on pp.50, 53 and 57, eight on p.69). A visually checked gutter at
x=1300 on skewed p.52 recovers four additional right-column headwords and
removes their text from unrelated left-column entries. Printed p.69 has 34 physical headwords after repair.
`boundary-overrides.json` retains these corrections independently of the OCR;
its headword labels on p.69 are provisional, not certified transcription.
`manual-lines.json` additionally restores GŌLER-/GŌLEN- on p.60, entirely
absent from OCR. Its text is manually transcribed; coordinates are approximate
reading-order locators, not invented OCR boxes. Its stable key is explicitly
manual and its reviewed record has empty raw_ocr plus manual_scan_evidence.
A second manual recovery supplies NAGUP-/NAGUT- on p.63: OCR retained
its grammatical label and gloss but omitted both headwords. The reviewed
record keeps the surviving OCR separate from the manual headword evidence.
Review every remaining page before treating this inventory as complete. Physical keys use printed page,
column and the frozen OCR line's top coordinate.

`reviewed-p48.json` through `reviewed-p77.json` contain **657 visually reviewed
headword records** across thirty pages (pp.48–77). These preserve lexical headwords, POS, glosses,
inflection notation, explicit alternates and selected cross-references. Two
homonymous ĀM entries remain separate, and AṚUP-/AṚUT- and ĀYA/AYA retain their
alternation. The close crop of p.49 confirms ASAṚ, including its retroflex mark.
These are 657 physical records, not 657 final lemma rows. Comparative paragraphs
remain unreviewed OCR; no comparative phonetic transcription is certified and
no graph edges have been installed. No exclusions or final unresolved-case
count has been established. This is not the fresh randomized 0/20 audit. Three typed uncertainties remain
in this pilot: the E/F reading in the p.52 household-member headword, and an
exclamation-like example symbol under ESEL on p.53, and a possible dot above
N in the p.54 tear headword (provisional KAṄĪR). The provisional headwords
are marked `uncertain-head-character`, rather than certified transcription.
The p.53 ORG-/OṚG- distinction and long-vowel homonyms are retained.
Printed homonym numbers in KARKE¹/² and KĀKAL¹/² are separate metadata;
they are preserved in the source head but excluded from phonetic forms.
Each retains its physical record key and its own gloss and morphology.
Entry-specific c=ts and j=dz notes on p.60 are also retained. The source
POS sb. on TĀRG- (to swallow) is preserved with an editorial note, not silently
corrected to a verb. Entry-specific j=z notes for KERIJ and GĀNJA KOR are structured as pronunciation
overrides while retaining source j. The kor/kōr length contrast and gã·ti
nasal-vowel length spelling are retained separately from gāṭi.

Printed pp.9–11 explain the initial transcription decisions: macrons represent
vowel length; the raised middle dot represents length of a nasalized vowel;
the source prints ṅ for the velar nasal because of typography. Preserve these
source symbols before applying a reviewed house profile. Small-cap headword
case is recorded separately from lowercase transcribed forms. Verb hyphens,
retroflex marks, spaces, and suffix versus full-form inflection distinctions
are retained. Remaining phonology still needs review; locality evidence is recorded in the manifest.

## Open gates

The mandatory dictionary, comparative-source and OCR addenda apply. Website/API
and upstream-CLDF addenda are inapplicable: the website is only the scan host.

- Establish exact headword/alternate/inflection counts, page continuity and scope.
- Finish reading phonology and comparative conventions (abbreviation keys and
  introductory locality now inspected).
- Visually transcribe phonetic forms; preserve raw OCR, boxes and corrections.
- Parse senses, POS, inflection, variants, donor labels and comparative prose;
  decide which compared attestations belong in structured comparison records.
- Complete a per-record audit and fresh seeded 0/20 source-to-output review.
- Register source settings, bibliography, sound profile and justified dialects;
  generate/review/install rich CSV with stable physical entry and child keys.
- Run focused checks, then consolidate with Turi/Asuri for the required full
  pipeline, full suite, ID/graph/reference/concept checks and browser QA.

The first visual lexical pass covers all thirty vocabulary pages. Final
extraction review, installation and integration gates remain open. No final source row count or completed lexical audit is
claimed yet. There are no representative app entries for this source because
it is not installed; full-build and browser gates remain deferred.

Pages 66–67 preserve pal/pāl and pinḍe/pinde contrasts, numbered PUN- homonyms,
and PUL as a cross-reference-only record with blank POS and gloss. PUNED’s
plural ev has no printed hyphen; its morphology notation remains unresolved.
This is separate from the three earlier typed reading uncertainties.

Pages 68–69 add 64 records. The PUL cross-reference now identifies the
physical BER PUL entry while retaining its printed target ber-pul. PODU
has no printed POS label; POYTA/POYTAN preserves separate postpositional
and nominal senses. Open ɔ, the BANJI j=z note, and BĀBU homonym numbers
are retained. The source spelling pumkin is retained with an editorial note.

Pages 70–73 add 80 records, including two explicitly labelled causatives
linked to their source verbs. MĀL daughter and wine remain separate physical
entries. Nasal length in MÃ·JIK, optional segments, VAṬ-/VAT, and the printed
MAGINḌ/MAGGINḌ spelling difference are retained. The tentative MARDIL/tree
comparison does not create ancestry. Source uncertainty in palate (tongue ?)
and the printed jyaiṣṭha calendar correspondence are preserved.

The final four pages add 89 records. All 657 candidate keys now have a
visually transcribed lexical record; `transcription-coverage.json` records
page counts, blanks and unresolved cases. Candidate coverage is not an
independent audit of completeness. Comparative prose and final source-to-output
audits remain pending. Explicit compounds are recorded for later graph review.
The source’s SANḌUP- open / SANDUP- make to grow contrast and SALÑIḌ retroflex
ḍ are preserved; source cross-references retain their printed spellings.

## Grammar review and resolved readings

`grammar-review.json` records visual review of pp.12–21, completing the
phonology chapter and checking noun-number/gender/case conventions. The tear
headword is now kanīr, confirmed both by the vocabulary close-up and the
explicit kanīr : kanīr-til pair in §13(iv), p.18. The previous provisional ṅ
reading is retained in review history. PUNED’s abbreviated plural ev is resolved
as punev from the explicit pair in §13(ii), p.18; its expansion replaces ed
rather than appending ev. Two reading uncertainties remain: ULFṬE’s F/E and
the example symbol in ese! senan. Both retain typed flags.

Sandhi and comparative correspondences will not be applied as unconditional
profile substitutions. Full profile implementation and output audit remain
open. Browser database work is user-triggered under the current checklist;
no refresh has been requested or run.

## Reproducible lexical expansion draft

`expand_lexical_units.py --output <directory>` generates 880 draft units:
657 physical headwords, 123 printed alternates, and 100 inflected forms.
The checked draft and per-record expansion audit are under `draft/`; every
input file hash is pinned in its summary. Units retain physical provenance,
source POS/gloss, typed uncertainty and entry-specific pronunciation notes.
Printed full forms override suffix concatenation; ending replacement for
puned → punev is checked against the grammar evidence.

Of 102 inflection annotations, two suffix scopes remain pending: āya/aya
with -v, and mar/marin with -kil. No plural child is guessed for either.
This script does not install CSV, assign final tags, resolve comparisons,
or assert graph relations for alternates/inflections. Those steps and the
fresh final audit remain open. Twenty focused checks pass.

## Sound conversion draft

`conversion/bhattacharya-ollari.txt` preserves the source’s vowel quality,
length, retroflex consonants and word boundaries, maps ṅ to house ŋ, and
maps ã·/õ· to nasalized macron vowels. `convert_draft.py --output <directory>`
applies reviewed pronunciation annotations per entry before tokenization.
This distinction matters: jir in jir er- has j=dz, whereas tanḍ jir has j=z.
Neither pronunciation is a global replacement. Original source forms remain
separate in the output. Affricates ts/dz become house ʦ/ʣ.

All 880 draft units convert without replacement characters; 77 display forms
change and 13 units carry scoped pronunciation notes. NFC/NFD input produces
identical results. Twenty-one focused tests pass. Pipeline routing remains
pending because it must preserve Original while honoring those local notes;
this standalone conversion does not establish compiled CLDF correctness.

## Pipeline support for source pronunciation

The generic source settings now support `transcription.input: phonemic` with
fallback to the original Form for blank pronunciation, and
`transcription.preserve_hyphens: true`. Parser tests show Original remains
unchanged while entry-specific pronunciation drives display conversion and
verb hyphens survive. Ollari should supply Phonemic only where the author’s
pronunciation annotation supplies a distinct layer; other rows leave it blank.
Existing defaults and special legacy conversion routes are unchanged.
The Ollari YAML/CSV are still pending; this is pipeline support, not installation.

## Rich import preview

`preview_import.py --output <directory>` produces the 15-column CSV and
proposed YAML settings under `preview/`, plus a physical-record audit and a
row audit. It regenerates from review records rather than trusting a stale
expansion snapshot. The 880 rows include 123 proposed variant links and 110
proposed derivation links (100 inflections plus 10 explicitly labelled source
derivations). These graph proposals still require final review.

Source POS, number, oblique-stem and explicit gender labels become canonical
tags. Only the 13 rows with source pronunciation annotations have a distinct
Phonemic value. Raw OCR and unresolved prose remain in the record audit;
comparison/usage sidecar publication is still pending. Three rows inherit
the two typed reading uncertainties, including the uncertain headword’s plural.
PUL’s cross-reference-only record retains its blank gloss.

All 880 preview rows pass through the actual parser without conversion errors,
with source Original, Phonemic and expected display Form checked row by row.
Twenty-five focused tests pass across acquisition/expansion and parser settings.
This is not compiled CLDF survival: final bibliography, comparison review,
fresh audit, installation and the complete pipeline remain open.

## Seeded lexical sample

`sample_audit.py --seed 2026092101 --output <file>` reproduces the lexical
sample. Optional `--pdf <pinned PDF> --render-dir <scratch directory>` produces
scan crops at 180 dpi after verifying the source hash. The checked sample and
manual results are in `audits/`. Twenty physical entries (27 expanded rows)
were visually compared for headword, gloss, POS/gender, alternates, inflection
and display conversion: 0/20 material lexical errors. The sample pins review
files and preview hashes. It does not certify comparative prose, complete
sidecar preservation, compiled graph or full ingestion. A fresh final audit
will be required after remaining importer changes. Twenty-six focused tests
pass across source acquisition and parser settings.

## Bibliography and comparison attribution

`bibliography-review.json` transcribes the 13 listed works on printed p.78
(PDF 89), preserving multi-part/edition dates. Kittel 1894 and Trench 1919–1921
match existing registry keys. Gundert’s printed initial F. conflicts with H.
in the existing candidate, and the Tamil Lexicon date span 1924–39 differs
from the registry’s 1924–1936; those candidates remain unresolved.

The bibliography explicitly attributes Dorli, Koya, Konda, Poya, Savara and
many Kurukh items to the author’s field notes. Language abbreviations therefore
cannot be blindly mapped to one published dictionary. Entry-level references
still need review. `source-reference.bib` is the validated main-reference
preview, explicitly marked not installed and retaining series year 1956 versus
publication imprint 1957. It has not been added to the canonical registry.
Twenty-seven focused tests pass across source evidence and parser settings.

## Comparative prose: first reviewed page

`comparison-reviewed-p48.json` holds the visual prose review for all 13
physical records on printed p.48 (PDF 59): eight comparison passages, one
usage comparison, one explicit source derivation and two literal-gloss notes.
Three entries have no additional prose. The review preserves printed language
labels and source relation cues, including cf., id. and etc. Column closeups
confirm Halbi tapa-tapi, Naik. īr aḍ-, and Kur./Brah. abbā.

This separate review layer pins the lexical-record hash and leaves the earlier
lexical sample reproducible. The lexical records' old comparison-pending flags
remain until the final merge of these review layers. Comparative forms retain
the author's transcription, including the page's warning that some Tamil
words reflect pronunciation. No auxiliary-publication attribution or graph
endpoint is inferred from a language abbreviation.

644 physical records on pp.49–77 still need prose review. This page has not
yet been installed as a published entry-text sidecar.

## Comparative prose: pp.49–50

The next two page reviews add 43 physical records, bringing coverage to 56/657
records across pp.48–50. The 50 prose passages comprise 33 comparisons, two
usage parallels, two source derivations, two literal-gloss notes, seven internal
cross-references, three source etymologies and one source attribution.

Source IA. attribution at ĀDIBAR, De. ɔlɔp < Sk. alpa at ƆLƆKEN, and the
Dravidian analysis at IYĀNḌ remain explicit source claims awaiting endpoint
review. ĀL's demonstrative/locative proposal retains the author's question
mark. The scan's Tamil ñanṭu/ñenṭu/nanṭu spellings in the crab comparison
retain plain n rather than being regularized. No lexical CSV or published
sidecars changed. Remaining prose review covers 601 physical records.

## Comparative prose: pp.51–52

Two further visual reviews add 47 physical records. Coverage is now 103/657
records on pp.48–52, containing 85 ordered prose passages. Page 51 preserves
the explicit `lw.` label at ISKUL without inventing its donor, the author's
question mark on ĪL's locative analysis, and comparative reconstructions and
causatives under IṚG- without turning them into Ollari ancestry claims.

Two comparative-transcription uncertainties are explicitly held from matching:
the Tamil dotted/underlined consonants at p.49 AṚ-, and l diacritics at p.52 US-.
These are additional comparative-prose uncertainties, separate from the two
previous lexical reading flags. Full-page and enlarged-column visual review,
key coverage, lexical-evidence hashes and NFC checks were performed. Remaining
prose review covers 554 records; canonical sidecar installation is pending.

## Comparative prose: pp.53–54

Coverage now includes 144/657 physical records on pp.48–54 and 123 ordered
prose passages. The two latest pages add 41 records and preserve cross-column
continuations at OKUṬ and KAṆ. Underlined k in the Kurukh/Brahui comparison
ḵhan is retained as source typography, without house-profile conversion.
The ESEL example retains its previously recorded uncertain exclamation-like
mark, bringing prose-layer uncertainty records to three (one overlaps an
existing lexical uncertainty).

KAṄAR explicitly cites S. C. Roy, The Mundas and their Country (1912), p.400.
This entry-level citation is absent from the 13-item bibliography and has been
added to the bibliography review's separate entry-level reference list. Its
registry resolution remains pending. Historical source commentary is retained
as attributed prose; it does not establish a donor or ancestry relationship.
513 records' prose still needs review.

## Comparative prose: pp.55–56

Added visual review for 34 records, bringing coverage to 178/657 records and
156 ordered passages. KAR- retains the Hindi/Bengali/Parji usage parallel;
KARKE¹ retains the source's calendar note. KĀRUP's comparison continues across
columns and is preserved as one passage followed by its cross-reference.
KĀYP-/KĀYT- prints Pa., which remains unresolved rather than silently becoming
Pj. Underlines visibly spanning kh are represented on both characters; the
p.54 KAṆ comparison markup was corrected consistently with a review history.
479 physical records' prose remains to review.

## Comparative prose: pp.57–58

Coverage now reaches 212/657 records and 189 ordered passages. The new pages
add 34 physical records. KIRK- explicitly cites The Parji Language, Vocabulary,
kelay-, now attached to bibliography-review item 4. KĒṬI/KĒṬIN's loanword
label occurs on the comparative Sanskrit form; KONḌKE's Oriya qualifier occurs
inside the Kui comparison. Both scopes are retained for graph review.

Cross-column continuations of KUYUG and KĒY- are preserved. Closeups confirm
the plain d of Pj./Naik. kēd- and retain the source's length punctuation.
445 records still require prose review; reference and entry-text publication
remain pending.

## Comparative prose: pp.59–60

The latest 50 records bring coverage to 262/657 and 220 ordered passages.
GŌTI ILENḌ explicitly compares only its gōti component with Sanskrit jñāti;
CŌKA cites a Bengali-from-Persian derivation within its comparative prose.
These scopes are preserved without automatically assigning whole-entry
ancestry or donor edges. The ber-goṭa example and cock-crowing literal gloss
are retained in their own semantic types.

One tiny mark above p in the Gondi comparison at CIPṚA is provisionally
transcribed and explicitly held for rechecking. Four prose reading flags
now remain (one overlaps an existing lexical flag). No lexical-preview rows
changed. 395 physical records still need prose review.

## Comparative prose: pp.61–62

Added 56 physical records; cumulative coverage is 318/657 records and 259
ordered passages. TANḌ JIR's component analysis jir < sir is preserved
separately from the Tamil comparison. The Bengali comparative form at
ḌEBRI KI remains labelled dialectal without inventing a named variety.
TĪN's honey, bee and honey-bee comparanda retain their distinct gloss scopes.
Closeups checked uncommon consonant marks and the Savara tæ·ne notation.
339 records' prose still needs review; canonical integration remains pending.

## Comparative prose: pp.63–64

Added 40 physical records, bringing coverage to 358/657 and 292 ordered
passages. DASRE and DIYĀLI retain the source's festival notes; DIGAL/DIGEL
retains its example and source derivation. NIRḌIN's Parji comparison preserves
both next-year and last-year meanings. NĪR's Sanskrit loanword label and NEY's
Telugu/Savara compound analysis retain their comparative scope.

Two fine marks in the plough comparison remain provisional under one typed
reading flag. Five prose reading-uncertainty records now remain. Key coverage,
lexical-evidence hashes and NFC checks pass; 299 physical records' prose and
canonical reference/sidecar integration are still pending.

## Comparative prose: pp.65–66

Added 44 physical records, bringing coverage to 402/657 and 322 ordered
passages. Numbered PANḌ- and PAR- homonyms retain their individual comparison
lists. The two PĀṬE entries (song/beam) remain separate. PIDIR distinguishes
nominal name from verbal to name in its comparanda, and PIRĀNḌ retains the
source's contrast between Ollari next year and Parji last year. PAT-'s
cross-column comparison is complete. 255 records' prose remains to review.

## Comparative prose: pp.67–68

Reviewed 52 physical records against the scan, bringing prose coverage to
454/657 records and 352 ordered passages. PUL retains its cross-reference
without an invented gloss. PERNONḌ comparisons remain attached to the big
adjective rather than the elder-brother noun; PĒN comparative plurals remain
scoped to Parji/Konda. PUṬKAL preserves distinct nest, ant-hill and white-ant
meanings. The POṄOR cross-column comparison is complete.

Ordered key coverage, lexical-evidence hashes and NFC checks pass. Five
earlier prose reading flags remain. No lexical preview rows or graph edges
changed. Outstanding: 203 records’ prose, reference/sidecar integration, final
graph review and complete audit, canonical installation, and full build/full
suite in an authorized environment. Browser refresh remains unrequested.

## Comparative prose: pp.69–70

Previous turn classified as progress. Visually reviewed 50 records against
scan columns and closeups; coverage is now 504/657 records and 394 ordered
passages. BĀNI retains the author’s question mark on its Sanskrit comparison.
MAGINḌ SINḌ preserves component meaning, literal gloss and suffix analysis
separately from comparative forms. Savara plurals and Parji stems retain
their comparative scope. Closeups resolved fine consonant and length marks
in the navel comparisons, including boḍḍu, buṭṭī and pokkuṟ.

Ordered page/key coverage, lexical-evidence hashes and NFC checks pass. Five
earlier prose reading flags remain; lexical-preview rows are unchanged and
no comparison edges were inferred. Remaining: 153 records’ prose, reference/
sidecar integration, final graph review and complete audit, canonical
installation, full build and full suite in an authorized environment. Browser
refresh remains unrequested.

## Comparative prose: pp.71–74

Previous turn classified as progress. Reviewed 80 records against scan
columns and closeups. Prose coverage reaches 584/657 records and 453 ordered
passages. MARDIL preserves the author’s probable tree connection. Sanskrit
loanword labels at MĪN/MĒGE retain their comparative scope. The LSI qualifier
on the Kui numeral is recorded for reference resolution. Bow/highland
homonyms remain separate; comparative plurals are not Ollari paradigms.

Fine marks were checked at 450 dpi. One unresolved mark in the Kurukh hare
comparison is typed explicitly, bringing prose reading flags to six. Ordered
key coverage, lexical-evidence hashes and NFC checks pass. No lexical-preview
rows or graph edges changed. Remaining: 73 records’ prose (pp.75–77), reference
and sidecar integration, final graph review and complete audit, canonical
installation, full build and full suite in an authorized environment. Browser
refresh remains unrequested.

## Comparative prose: pp.75–77 and complete first-pass coverage

Previous turn classified as progress. Reviewed the final 73 vocabulary
records against scan columns and selected 450 dpi closeups. All 657 physical
records now have first-pass prose review, preserving 507 ordered passages.
VĒLE explicitly distinguishes the Kolarian sun-word group; SEY- and SOYUP-
retain Munda comparisons without invented ancestry. SAVKOL and SIREL retain
component analyses. SIṬ- and SIR homonyms remain separate, and comparison
plurals and literal glosses retain their source scope.

Six earlier reading flags remain. Lexical-evidence hashes, ordered keys and
NFC checks pass. Added corpus-wide prose coverage and scope regressions.
Lexical-preview rows remain unchanged; no comparison graph edges emitted.
Outstanding: resolve or type final uncertainties, reference/sidecar
integration, final graph review and complete audit, canonical installation,
full build and full suite in an authorized environment. Browser refresh
remains unrequested. Complete prose coverage is not completed ingestion.

Validation: 28 focused acquisition/preview/prose and transcription-settings
tests pass (0.83 s); scoped whitespace check passes. Full pipeline and suite
remain outstanding.

## Enlarged-scan review of prose uncertainties

Previous turn classified as progress. Rechecked all six provisional prose
readings against 450 dpi scan crops. Four are resolved: aḻu replaces aḷu on
p.49, umiḻ/ugiḷ are confirmed on p.52, cipṛe replaces ciṗre on p.60, and
ñeṅṅal/nāyerụ are confirmed on p.63. The exclamation-like mark in ESEL and
the fine mark in the Kurukh hare comparison remain explicitly uncertain
and unavailable for endpoint matching. Before/after evidence and decisions
are saved in audits/prose-reading-review.json.

All 657 prose records and 507 passages remain accounted for. Lexical evidence
and preview rows are unchanged. The 29 focused tests pass (0.82 s), including
correction-history and residual-uncertainty checks. Scoped whitespace check
passes. Reference and prose-sidecar integration, final graph review/audit,
canonical installation, full build and full suite remain outstanding.

## Source-keyed prose preview

Previous turn classified as progress. Added prose_preview.py to reproducibly
export 507 typed text passages on 465 physical headwords, with a full 657-record
audit and two retained reading flags. The preview cites Bhattacharya with
page/column locators; auxiliary reference resolution remains pending. Prose
is not automatically copied onto generated variants or inflections.

Added generic optional Entry_Key resolution for raw entry-text sidecars in
make_cldf.py via entry_text_sources.py. It streams emitted forms, preserves
legacy Form_ID sidecars, and rejects missing/ambiguous/conflicting targets.
The 33 focused tests pass (0.87 s), including actual parser attachment for
all 507 passages after reversing form order, correction history, source
coverage, and negative target checks. Scoped whitespace checks pass.

Canonical installation remains pending reference resolution, graph review,
and final source-to-output audit. Full build and full suite remain required
in an authorized environment; browser refresh is unrequested.

## Explicit auxiliary citations

Previous turn classified as progress. Resolved three explicit prose citations
in explicit-reference-resolution.json: Roy 1912 at p.400, Burrow/Bhattacharya
1953 at Vocabulary, kelay-, and the existing LSI key at volume IV. Two new
auxiliary records remain in auxiliary-references.bib for preview; no canonical
registry edits yet. Roy’s publisher/printer distinction is left unasserted,
with the catalog discrepancy recorded. These are indirect citations through
Bhattacharya, not independently collated lexical attestations.

The prose preview now adds auxiliary citations only on those three passages.
LSI is explicitly scoped to the Kui numeral, and language labels alone do not
receive book citations. Bibliography-only works retain their identification
notes pending final reference reconciliation. All 507 passages and 657 source
records remain accounted for. The 34 focused tests pass; full compilation,
formatted references, final graph/audit and installation remain outstanding.

## Internal cross-reference review

Previous turn classified as progress. Reviewed all 51 cross-reference,
derivation, analysis and etymology passages in internal-relationship-review.json.
Of 34 cross-references, 32 have navigation targets selected using printed
heads, directional cues and glosses. BĀR- = pār- is an explicit-equivalence
variant candidate. MUTAM SIKAṬ’s reference to sikaṭ differs from the SĪKAṬ
headword and remains a candidate pending a source check. Homonym selections
exclude ĀM we and PERNONḌ elder brother. PUL remains a cross-reference without
an invented gloss or variant relation.

The remaining 17 analysis passages include one existing preview derivation,
three explicitly uncertain analyses kept unlinked, and 13 analyses needing
endpoint/relation review. Candidate navigation does not create ancestry. No
new graph edges installed. The 35 focused tests pass, including exhaustive
claim coverage, target evidence, homonym protection and claim-text fidelity.
Reference reconciliation, final graph integration, source-to-output audit,
canonical installation and full build/suite remain outstanding.

## Combined integration preview

Previous turn classified as progress. Added reviewed-relations.json and
integration_preview.py, combining lexical and prose outputs in a separate
integration-preview directory. The source’s explicit BĀR- = pār- equivalence
is now represented by Variant_Of_Key targeting the distinct to sing entry.
The combined preview has 880 rows, 124 variant links, 110 derivation links,
and 507 prose passages. The earlier lexical-only audit remains reproducible.

The actual parser preserves the new variant key and matching gloss. Tests
reject stale headwords/glosses, conflicting relationships, missing endpoints
and changed source evidence. The 36 focused tests pass. No canonical data or
compiled graph was regenerated; final graph serialization, remaining source
analyses/references, a fresh integrated audit, installation and full build/
suite remain outstanding.

## Source-analysis dispositions and pronoun inflection

Previous turn classified as progress. Reviewed all 17 source analyses and
recorded individual decisions in source-analysis-dispositions.json, including
current compiled candidate snapshots. Preserved donor chains, uncertain
claims and component scopes; unsupported endpoints remain unlinked in prose.
The Sanskrit kakṣa candidates require sense review rather than a spelling-only
match. Printed kal stone was not silently changed to Ollari kanḍ.

ŌR is explicitly the plural of ōnḍ/on- and had no duplicate expanded row.
The combined preview now links the existing ŌR headword to ōnḍ as an
inflectional derived relation and adds pl. It retains 880 rows, 124 variants
and now 111 derivations. The actual parser preserves the relation and tag.
All 36 focused tests pass; the 17 analysis decisions exactly cover their
reviewed source passages. Full graph serialization, remaining reference and
suffix decisions, final integrated audit, installation and full build/suite
remain outstanding.

## Final source-stage status (25 September 2026)

The earlier progress notes above are historical. The complete printed
vocabulary on pp. 48--77 is now installed in
`data/other/forms/20260921-bhattacharya-ollari.csv`: all 657 physical
headwords are accounted for, producing 880 lexical rows. The source-keyed
entry-text sidecar has 509 comparative prose passages on 466 headwords;
comparative-language words remain prose, without unsupported graph edges.
Both canonical files are byte-identical to their integrated preview outputs.
The source, two auxiliary citations, and settings are registered. Both
ambiguous suffix scopes were reviewed and retained unexpanded; three lexical
readings and two prose entries retain recorded uncertainty.

The final seeded source-to-output audit reviewed 20 physical records, 24
emitted lexical rows, and 16 prose passages against the scan, with zero
material errors (`audits/integration-results-2026092105.json`). The focused
Ollari and entry-text tests pass (39 tests), and source metadata validates.
Full database/graph compilation, the full test suite, and browser QA are
deferred under the user's explicit no-database-build and no-remote-work
instructions; this is a source-stage completion, not a full build claim.
