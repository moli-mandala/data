# Diversity-oriented source ingestion, 2026-09-21

Goal remains active: find and ingest additional Indo-Aryan, Dravidian and/or
Munda sources prioritizing language diversity and sparse coverage.

This continuation made progress: a new source-local Turi package was installed
and checked. There was no earlier goal-turn record available in this context
to classify; no stale process/lock was treated as a running job.

## Installed this pass: Peterson et al. 2024, Odisha Turi

See [complete source report](../data/other/forms/raw_data/peterson_turi_2024/README.md)
for provenance, exclusions, review decisions, paths, commands and deferred gates.
The mandatory ingestion checklist and survey-wordlist addendum apply. PDF
extraction used the native text layer, so OCR addendum is inapplicable. External
CLDF/API, control-language exclusion and printed etymon matching are inapplicable.

- 275 source records → 224 attested responses → 246 installed source rows.
- 51 explicit unelicited cells excluded; all preserved in the audit.
- 0 ancestry/borrowing/variant links. 106 responses have source IA commentary;
  comments do not identify donor nodes and are not converted into graph edges.
- Existing language Turi, one new named Odisha dialect, no invented coordinates.
- One transcription residual (redundant nasal mark, item 58), flagged; one
  explicitly tentative borrowing comment (item 206), flagged separately.
- Reproducible glyph-baseline extractor; frozen raw glyph and page evidence;
  seeded visual audit 0/20 plus 18 edge cases. Bibliography/YAML/profile installed.
- 12 focused source/profile/dialect tests pass; metadata validation passes.
- The global sound-profile policy check reports a pre-existing missing rule in
  `sil-pahari-pothwari`; the new profile has no violations.
- Compiled survival test was run separately and fails because the current CLDF
  has no source keys for this new source. This is an open integration gate, not
  a passing result. Full build, full suite, references, IDs, graph, concepts,
  conversion error report, generated-diff review and browser/app checks remain
  deferred under the laptop resource policy. Configured CI has no dispatch
  route for uncommitted inputs and does not perform a full pipeline build.
- No commit, push, release, browser DB refresh, or heavy local build performed.

## Next discovery priorities

### Continuation: Asuri selected; complete-source extraction underway

The preceding Turi turn is classified as **progress**: it changed source inputs,
added tests, and established integration as outstanding. Its package still
exists in the current worktree. The next turn verified pending-input coverage:
Asuri still has 29 source rows, so the CUJ dictionary was selected over the
20-word Birhor web sample and the larger Ho/Mundari/Santali comparative survey.

The university's linked draft PDF is now pinned and a reproducible extractor
accounts for **2,005 physical entry blocks**. A non-installing candidate parser
records 1,825 articles, 180 cross-reference entries and 2,107 sense slots.
Six focused extraction tests pass. Native font defects, full-width alphabet
bands and bottom-line truncation were found and corrected; one source glyph
and the missing-transcription/gloss classifications remain under review.
**No Asuri forms are installed yet.** See the [stage report](../data/other/forms/raw_data/cuj_asur_2020/README.md)
for precise counts, reproduction, source-version decisions, remaining work and
all deferred ingestion gates. Do not mistake the candidate snapshot for a
finished importer or a fresh 0/20 lexical audit.

Streaming counts in current compiled data show Asuri 29, Gorum (`go`) 107,
Birhor 323, Gutob (`gu`) 392, and Koda 620. These are selection evidence from
the current compiled snapshot, not counts of all pending source inputs.

1. Asuri: the Central University of Jharkhand endangered-language centre's
   publication page lists an Asur grammar/semantic vocabulary and a 2020
   Asur–Hindi–English dictionary with over 2,000 words. Inspect availability,
   licensing, actual coverage and duplication before ingestion:
   https://cujcel1220.wixsite.com/endangeredlanguages/publications
2. Birhor: Visva-Bharati's Centre for Endangered Languages publishes a sample
   wordlist. Inspect the actual records and overlap before selecting it:
   https://cfelvb.in/language_data_new.php?lang=49
3. Kobayashi et al. (2003) Kherwarian survey was considered but is lower priority
   for this objective: its 12 lists cover Ho, Mundari and Santali varieties,
   rather than the sparsest base languages. Do not select it merely for volume.

These are discovery leads, not verified ingestions or claims that the full
goal is satisfied. Continue broader IA/Dravidian discovery alongside the
underrepresented Munda targets; consolidate source changes into one full build
when an authorized suitable runner becomes available.

### Asuri continuation: candidate audit and reference accounting

Corrected double pre-base-i reordering and the embedded-font dotted-circle
mapping. Current extraction: 2,005 entries; 1,805 articles, 200 cross-references,
1,714 IPA heads and 2,107 sense slots. A complete mechanical audit classifies
all entries, including 84 bare and 50 transcribed undefined heads. It records
515 unique exact reference targets and two unmatched targets; installs no edges.
Eight focused tests pass. Seed 20260924 yields 0/20 material errors within the
limited head/IPA/English/sense review; rich annotations, tags, relations and
normalization are not yet certified. Eight dotted-circle spellings and one
unknown CID remain flagged. Source installation, sound profile, bibliography,
complete rich-import audit and all integration/full-build/browser gates remain
open. This continuation is progress, not a completed ingestion.

### Asuri continuation: transcription and rich importer proposal

This turn is progress: a source-specific transcription profile and 15-column
proposal importer now exist. Proposal: 2,106 rows, one corrupt-head exclusion,
177 variant links, two explicit ordered component edges, and 23 unresolved
relationships. Twelve focused tests pass, including complete proposal symbol
coverage, NFC/NFD equivalence and cycle detection. Full-sized reference sense
numbers are now distinct from homonym labels; 539 exact targets and two missing
targets are accounted for. Four embedded subentries were inspected visually;
three Hindi-only definitions were preserved with explicit translations and
both locators. Source CSV/YAML/bibliography installation and a fresh complete
rich-output sample remain pending. Full build, full tests, compiled metadata,
IDs/graph and browser QA are still deferred. No production changes were made.

### Asuri continuation: source inputs installed, full integration pending

This turn is progress. Installed 2,106 rows from 2,005 physical records; excluded
one undecoded native-only head with full audit evidence. Registered the source
YAML/profile route and bibliography, reusing base Asuri without inventing a
uniform dialect or coordinates. Corrected target-relative parenthetical scope,
reference hyphen spacing and inclusive/exclusive person codes. Accepted 185
variant links and two ordered component edges. Sixteen unresolved relationships
(15 ambiguous targets/senses and one missing target) remain unlinked/uncertain.
Native-only rows: 290; blank glosses: 352; uncertain rows: 47. Typed reasons and
all emitted rows are retained in the per-record installation audit.

Fresh rich-output seed 20260926: 0/20 material errors. All dotted-circle and
superscript-stop cases visually checked; source spellings retained, not guessed.
Fourteen source tests plus six dialect tests pass; metadata validation passes;
source profile passes scoped policy and every installed form tokenizes in NFC
and NFD. Actual parse_file retains 2,106 rows, distinct keys, phonemic/source
layers and graph keys with no conversion errors. Bibliography parses correctly.
The compiled-survival test explicitly fails: 0/2,106 expected keys in current
CLDF. Full build, full suite, formatted references, stable IDs, compiled graph,
concepts, deduplication, generated-diff review and browser QA remain deferred.
No heavy local job, commit, push, release or browser refresh was performed.

Asuri source preparation has reached installation; this does not complete the
broader diversity goal. Continue source discovery for underrepresented IA and
Dravidian languages while consolidating the required full integration run.

### Next selection: Ollari Gadaba comparative vocabulary

The preceding turn was progress: Asuri inputs were installed, with full gates
still explicitly open. Coverage was rechecked from both current CLDF and raw
manual inputs. Ollari Gadaba has 60 in each, from DravLex; Ravula has 100,
Pattapu 101, and Belari 108 manual rows. Palu Kurumba has only four compiled
rows, but no accessible substantial lexical source was found in this pass.
Its 2019 Kapp grammar/dictionary is a verified publisher lead, not an ingest.

Selected Bhattacharya (1957), *Ollari: A Dravidian Speech*, pp.48–77. Recovered
the 93-page public Tamil Digital Library scan after inspecting its catalogue
HTML; both listed downloads point to the same PDF. The complete scan has no
native text. A reproducible, one-thread 300-dpi Tesseract scaffold now covers all 30
vocabulary pages (10,094 OCR word tokens; two acquisition checks pass), with the dictionary/comparative/OCR addenda active. Pilot OCR loses
phonetic diacritics and is explicitly non-installable pending visual review.
The pinned acquisition package is
`data/other/forms/raw_data/bhattacharya_ollari_1957/`.

Bapuji's 2019 Hyderabad thesis was investigated first; its public URL redirects
to a missing file (confirmed HTTP 404). The 2025 revised grammar exists at
https://lincom-shop.eu/LWM-518-A-Descriptive-Grammar-of-Ollari-Gadaba/en but no
public full text was found or purchased. The selected primary scan is at
https://tamildigitallibrary.in/assets/docs/uploads/primary_files/book/TVA_BOK_0042223/TVA_BOK_0042223_Ollari_a_Dravidian_speech.pdf

Configured CI was rechecked: it triggers only on main push or PR and runs tests,
not a complete data pipeline; no suitable authorized dispatch route appeared.
No publication or new infrastructure was used to offload outstanding gates.

### Ollari transcription pilot

Read printed phonology pp.9–11 and retained source macrons, retroflex marks,
ṅ and raised-dot nasal-vowel length conventions. Boundary extraction proposes
640 records but is not complete: the margin heuristic misses entries around
BER PUL / BERE / BELE on p.69. The raw OCR and boxes remain pinned.

Visually transcribed 35 physical headword records on pp.48–49, including
inflections, two separate ĀM homonyms, causative alternates and mother variants.
ASAṚ was checked in a close crop to preserve its retroflex final consonant.
Comparative prose remains explicitly unreviewed OCR. Four focused tests pass
for acquisition integrity, physical evidence attachment and reviewed lexical
contrasts. No final lemma count, exclusions total or randomized audit is claimed.

Remaining: complete boundary/transcription review, source locality, comparative
records, rich parsing, profile/settings/bibliography, per-record and fresh sample
audits, installation, and all full integration/browser gates. No Ollari app
entry exists yet; no heavy local run, commit, push or publication was performed.

### Ollari continuation: pp.50–51 and source locality

Added 44 visually transcribed headword records, bringing the four-page total
to 79. Preserved the two INḌI homonyms, open ɔ, long ī pronouns, alternate
causatives and inflection stems. A 300-dpi crop confirms all three retroflex
ḍ marks in UNḌUP-/UNḌUT-/UNḌUK-. Nine explicit boundary overrides recover
ɔssa on p.50 and eight missed p.69 entries. Page 69 now has 34 boundaries;
the global 649 candidates remain unverified as a complete source count.

Read introduction pp.1–8. Printed p.8 / PDF p.19 specifies collection at
Lamptapuṭ, Munḍagaṛ and Koṭri in 1951/1955. Retain base OllariGadaba and
source-level locality provenance; no per-entry village, uniform dialect
or coordinates are justified. Five focused checks pass, including exact
boundary reproduction and attachment of all 79 reviewed records to raw OCR.
Comparative prose remains unreviewed; exclusions and final counts, remaining
transcription, audits, installation and full integration/browser gates are open.

### Ollari continuation: pp.52–53

Added 45 visually reviewed physical headword records (124 total across
pp.48–53). Corrected the skewed p.52 gutter: four right-column headwords
had been interleaved into left-column paragraphs. Added a visual boundary
override for OṚG- on p.53, preserving the separate ORG- entry. There are now
654 global candidates, still not a verified inventory. Six focused checks pass.

Two typed pilot uncertainties remain: E/F in the household-member headword
on p.52 and an exclamation-like symbol in an ESEL example on p.53. These are
recorded explicitly, not silently normalized. Comparative prose remains OCR,
and no final lemma count, exclusions count or fresh random audit is claimed.
The source remains uninstalled; remaining transcription, profile/metadata,
comparative review, audit and full integration/browser gates remain open.

### Ollari continuation: pp.54–55

Added 36 visually reviewed physical records (160 total on pp.48–55).
Numbered KARKE and KĀKAL homonyms remain separate records; their superscript
labels are metadata, not phonetic characters. Preserved the printed KAṬ-/KAT-
contrast and kaṇul inflection. One additional possible diacritic in the
tear headword is explicitly uncertain, bringing typed pilot uncertainties
to three. Seven focused tests pass. The global boundary count remains 654
unverified candidates; these two pages required no additional overrides.

Comparative prose remains unreviewed OCR. Remaining transcription, uncertainty
resolution, comparative parsing, final audit, installation, full build and
browser QA are open. No final row/exclusion count or completed ingestion claimed.

### Ollari continuation: pp.56–57

Added 37 reviewed physical records, reaching 197 on pp.48–57. Recovered the
missed KUYUG thigh entry and its plural kuyugul by visual boundary override.
Kept KI hand/or as distinct records, preserved kã·j- nasal-vowel length
notation, and retained the parenthesized nīr context for KĀKOR separately.
The global inventory is now 655 unverified candidates; eight focused checks
pass. The three earlier typed uncertainties remain open.

No source installation or final exclusions/count audit is claimed. Remaining
transcription, comparative review, profile/metadata, fresh random audit, full
build and browser QA remain required. No heavy local run or publication occurred.

### Ollari continuation: pp.58–59

Added 37 reviewed physical records, reaching 234 on pp.48–59. Preserved
entry-specific j=z notes as pronunciation overrides, the kor/kōr fowl/horn
length contrast, and gã·ti joint versus gāṭi many. Raw comparative passages
remain attached and explicitly unreviewed. These pages require no additional
boundary overrides; the global 655 candidates remain an unverified inventory.
Nine focused checks pass; the three earlier typed uncertainties remain open.

Remaining transcription, comparative review, profile/metadata, fresh random
audit, installation, full build and browser QA remain required. No final source
row/exclusion count or completed ingestion is claimed.

### Ollari continuation: pp.60–61

Added 59 visually reviewed physical records (293 total across pp.48–61).
Recovered GŌLER-/GŌLEN- to abuse, absent entirely from OCR, using a clearly
labelled manual scan transcription with approximate reading-order locator
and no OCR word boxes. Its raw_ocr remains empty. The inventory now has 656
candidates, still unverified globally. Ten focused checks pass.

Preserved c=ts and j=dz entry-specific notes, retroflex and nasal symbols,
and the anomalous printed sb. label for TĀRG- to swallow with an editorial
note. Three earlier typed reading uncertainties remain open. Comparative
prose remains unreviewed. Remaining transcription, comparative parsing,
profile/metadata, final audits, installation, full build and browser QA are
required; no final row/exclusion count or completed ingestion is claimed.

### Ollari continuation: pp.62–63

Added 45 reviewed physical records, reaching 338 on pp.48–63. Restored
NAGUP-/NAGUT- headwords manually while retaining the surviving OCR label
and gloss separately. Preserved TUÑ(G)- optional-segment notation, the
tite/tīte bitter/bird contrast, source cross-references, and the missing
printed hyphen in TŌṬP. Eleven focused checks pass. The global 657 candidates
remain unverified; three earlier typed reading uncertainties remain open.

Remaining transcription, comparative review, profile/metadata, fresh sample
audits, installation, full integration and browser QA are still required.
No final row/exclusion count or completed ingestion is claimed.

### Ollari continuation: pp.64–65

Added 41 visually reviewed records, reaching 379 across pp.48–65. Preserved
PANḌ-¹/² and PAR-¹/² as distinct physical entries with homonym numbers outside
phonetic forms. Retained gender-marked NIYAṬE/NIYAṬONḌ and the source nasal
contrasts. No new boundary repair was needed; 657 global candidates remain
unverified. Twelve focused tests pass; three earlier reading uncertainties
remain open. Comparative prose remains unreviewed OCR.

Remaining transcription, comparison parsing, profile/metadata, final audits,
installation, full integration and browser QA remain required. No final
row/exclusion count or completed ingestion is claimed.

### Ollari continuation: pp.66–67

Added 45 visually transcribed physical records, reaching 424 on pp.48–67.
Preserved pal/pāl, pinḍe/pinde, optional g in PAṚṄ(G)-, numbered PUN-
homonyms and the two PERNONḌ entries. PUL remains a cross-reference-only
record with no invented POS or gloss. The printed plural ev under PUNED
is retained with unresolved morphological notation, separate from the three
open reading uncertainties. No boundary changes or exclusions in this batch.
Thirteen focused checks pass; this is not a fresh source-to-output audit.

Remaining ten vocabulary pages, comparative review, profile/metadata, fresh
sample audit, installation, full integration and browser QA remain open.
There are no representative app entries for this uninstalled source yet.

### Ollari continuation: pp.68–69

Added 64 visually transcribed physical records, reaching 488 on pp.48–69.
Linked PUL’s printed ber-pul cross-reference to the physical BER PUL entry,
without inventing a PUL gloss or ancestry. Preserved the missing POS on
PODU, separate POYTA/POYTAN senses, open ɔ spellings, BANJI j=z and numbered
BĀBU homonyms. No new boundary repairs or exclusions. Three reading
uncertainties and one morphology notation remain open. Fourteen focused
evidence checks pass, not a fresh source-to-output audit.

Eight vocabulary pages remain, along with comparison review, profile and
metadata, final audits, installation, full integration and browser QA.
No final lemma count or completed ingestion is claimed.

### Ollari continuation: pp.70–73

Added 80 visually transcribed physical records, reaching 568 on pp.48–73.
Preserved two explicit causative links, distinct MĀL daughter/wine entries,
nasal-vowel length, optional segments, retroflex contrasts, the printed
MAGINḌ/MAGGINḌ variation and the source’s palate (tongue ?) uncertainty.
The tentative medicine/tree comparison remains a note without an ancestry
edge. No boundary repairs or exclusions in this batch. Sixteen focused
checks pass. Three earlier reading uncertainties and one morphology
notation remain open; this is not the fresh final source-to-output audit.

Four vocabulary pages remain; comparative review, metadata/profile, final
audits, installation, full integration and browser QA are still required.
No final lemma count or completed ingestion is claimed.

### Ollari continuation: pp.74–77 and complete first lexical pass

Added 89 physical records, reaching 657 across all thirty vocabulary pages.
Every candidate key has one review record; the new transcription coverage
ledger lists page counts, blank fields and unresolved cases. This establishes
coverage of the candidate scaffold, not an independent no-omissions audit.
Preserved SANḌUP-/SANDUP-, numbered SIṬ-/SIR homonyms, optional retroflex ṭ,
entry-specific j=z, explicit compound analyses and printed cross-reference
variants. Close-up review corrected SALÑIḌ’s final retroflex ḍ. No exclusions
in this batch. Eighteen focused checks pass. Three earlier typed reading
uncertainties and one morphology notation remain open.

Comparative prose, remaining phonology, per-record and fresh seeded audits,
profile/metadata, final row expansion, installation, full integration and
browser QA remain required. This source is not yet installed; no final lemma
count or completed ingestion is claimed.

### Ollari continuation: grammar cross-check and uncertainty repairs

Visually read pp.12–21: remaining phonology plus noun number, gender and case.
Resolved provisional KAṄĪR to KANĪR using a vocabulary close-up and the
independent printed kanīr : kanīr-til pair, p.18 §13(iv); preserved the earlier
reading in review history. Resolved PUNED plural ev to punev using the explicit
pair p.18 §13(ii), retaining raw ev and structured ending replacement. Two
reading uncertainties remain; no morphology notation remains unresolved.
Recorded evidence in grammar-review.json and updated the coverage ledger.
All 657 physical records remain present; 18 focused checks pass.

Profile/importer, comparative review, final audits, metadata, installation and
full data integration remain open. Browser refresh is conditional on a user
request under the current checklist and has not been requested or performed.

### Ollari continuation: reproducible lexical-unit expansion

Added expand_lexical_units.py and draft output with a complete physical-record
audit. 657 headwords + 123 printed alternates + 100 inflected forms produce
880 draft units. Of 102 printed inflection annotations, two remain pending
because suffix scope over alternative stems is unresolved: āya/aya -v and
mar/marin -kil. No guessed plural is emitted for those records. Stable child
keys preserve provenance, uncertainty and local pronunciation notes.
Twenty focused checks pass, including exact reproduction, complete audit
accounting, plural replacement and rejection of unexpanded notation.

Two reading uncertainties and two suffix scopes remain open. This is an
intermediate expansion, not an installed CSV or final lemma count. Comparison
review, profile conversion, tags/graph interpretation, final audits, source
metadata, installation and full integration remain required.

### Ollari continuation: source-aware sound conversion draft

Rechecked the printed phonology pp.9–11. Added the sound profile and a scoped
conversion step that keeps source forms separate. All 880 draft units convert
without replacement characters: 77 changed display forms and 13 units with
entry-specific pronunciation. Correctly distinguishes jir er- (j=dz) from
tanḍ jir (j=z), and retains c/j elsewhere. Nasal length, ṅ→ŋ, vowel quality,
retroflex contrasts and word boundaries have focused NFC/NFD checks.
Twenty-one focused tests pass. Pipeline routing for local pronunciation is
still required; no installed CSV or compiled survival is claimed.

Two reading uncertainties and two suffix scopes remain. Comparative review,
final audits, grammar/graph modeling, metadata, installation and full data
integration remain open.

### Ollari continuation: generic pipeline pronunciation settings

Added source-configurable generic conversion input and hyphen preservation.
The parser now supports converting a nonblank authorial pronunciation field
while retaining source spelling in Original, with source-Form fallback when
pronunciation is blank. Separate setting retains meaningful boundary hyphens.
No Ollari-specific branch was added. Parser tests cover differently pronounced
jir entries, unchanged Original/Phonemic, blank fallback and legacy defaults.
24 focused tests pass (21 acquisition/expansion + 3 settings/parser cases);
source metadata validation and two existing focused sound-profile checks pass.

Ollari settings and CSV installation remain pending, as do grammar/graph
modeling, comparative review, final audits and full integration. Shared-parser
full-suite validation must run with the required full build on an authorized
runner; it has not been substituted with these focused checks.

### Ollari continuation: rich CSV preview and corpus parser check

Added a reproducible 15-column CSV preview and proposed settings, regenerating
from authoritative review records. 657 records yield 880 preview rows with
123 variant links and 110 derivation proposals (100 inflections plus 10
explicit source derivations). Canonical POS/number/gender/oblique tags are
separated from glosses. Thirteen authorial pronunciation rows preserve
source spellings separately; three rows retain typed reading uncertainty.
PUL remains the one blank-gloss cross-reference. Two suffix scopes remain held.

All 880 rows pass actual parse_file conversion with zero errors and exact
row-wise checks of Original, Phonemic and display Form; 25 focused tests pass.
No compiled survival or installation is claimed. Comparison/usage sidecars,
reference resolution, final graph review, fresh source audit, bibliography,
installation and complete data pipeline/full suite remain required.

### Ollari continuation: seeded visual lexical audit

Seed 2026092101 sampled 20 physical entries, covering 27 expanded preview
rows. Compared scans against source and display forms, glosses, POS/gender,
printed alternates and inflections: 0/20 material lexical errors. Sample
selection and crop rendering are reproducible; review input and preview hashes
are pinned beside manual per-entry outcomes. Twenty-six focused tests pass.

This lexical-only sample does not close the final audit gate: comparative
prose/reference review, sidecar preservation and compiled graph remain outside
its scope. A fresh final sample remains required after importer completion.
Two reading uncertainties and two suffix scopes remain open; no installation
or full-build completion is claimed.

### Ollari continuation: bibliography and comparison provenance

Visually transcribed all 13 entries on bibliography p.78 and its explicit
field-note attribution. Matched Kittel 1894 and Trench’s two-volume Gondi
reference to existing keys. Preserved unresolved Gundert initial and Tamil
Lexicon date discrepancies rather than silently equating them. Language
abbreviations do not uniquely imply a bibliography source: several comparison
languages are explicitly attributed to the author’s field notes.

Added a validated main-reference BibTeX preview with 1957 imprint/1956 series
year, exact scan URL, provenance and OCR/editor credit, explicitly not installed.
Twenty-seven focused tests pass. Remaining auxiliary/entry-level citation
resolution, comparative prose, final audits, installation and full data
integration are still open.

### Ollari comparative prose, p.48

Visually reviewed all 13 records on printed p.48/PDF 59 against full page and
enlarged columns. Saved eight comparison passages, one usage parallel, one
explicit duplication derivation, and two literal-gloss notes in a separate
source-local review layer; three records have no prose. Correctly retained
Halbi tapa-tapi, Naik. īr aḍ-, and Kur./Brah. abbā, with author language labels
and cf./id./etc. unchanged in meaning. Key coverage and NFC checks pass.
No comparison graph edges or auxiliary references were guessed.

Remaining: 644 physical records' prose, sidecar publication, auxiliary
reference resolution, final graph review and fresh complete audit, canonical
installation, then required full build/CLDF survival gates on an authorized
runner. User-triggered browser refresh has not been requested. The 880-row
lexical preview and previous 0/20 lexical-only sample remain unchanged.

### Ollari comparative prose, pp.49–50

Previous goal turn classified as progress: p.48 visual prose review was saved.
Continued under dictionary, OCR-heavy and comparative-source addenda. Added
visual prose reviews for all 43 records on pp.49–50 (PDF 60–61), using enlarged
columns and 400-dpi closeups for difficult glyphs. Combined review now covers
56 physical records and 50 passages. All records retain immutable keys and
hash-pinned lexical evidence; page/key coverage and NFC validation pass.

Preserved the author's uncertain ĀL derivation, IA. attribution at ĀDIBAR,
De./Sanskrit analysis at ƆLƆKEN, and Dravidian analysis at IYĀNḌ. Cross-references
remain separate from ancestry; no guessed graph edges or auxiliary citation
keys were introduced. No lexical rows excluded or changed this turn.
601 records still require prose review; sidecar/reference integration, graph
review, final complete audit, canonical installation and full build/suite
remain outstanding. Browser refresh is not requested.

### Ollari comparative prose, pp.51–52

Previous goal turn: progress, with authoritative page-review files for pp.49–50.
Verified current manifest, then reviewed all 47 records on pp.51–52/PDF 62–63
against scan columns and enlarged glyph crops. Combined coverage now comprises
103 physical records and 85 passages: 60 comparisons, two usage parallels,
three source derivations, two literal glosses, 13 cross-references, three
source etymologies and two source attributions.

Retained ISKUL's explicit loanword label with unspecified donor and ĪL's
question-marked locative analysis. Added two typed comparative-transcription
uncertainties for fine dotted/underlined consonants (p.49 AṚ-, p.52 US-); no
matching or ancestry claims were made from them. No lexical rows changed or
were excluded. Source-key coverage, evidence hashes and NFC checks passed.
The earlier lexical-only audit is still reproducible; it does not certify
these comparative passages.

554 records' prose remains, along with canonical reference/sidecar integration,
final graph review and full audit, installation and full build/suite on an
authorized runner. Browser refresh remains unrequested. No heavy local work
or publication was performed.

### Ollari comparative prose, pp.53–54

Previous goal turn classified as progress. Revalidated manifest and reviewed
41 records on pp.53–54/PDF 64–65 against scan columns and closeups. Combined
coverage: 144 physical records, 123 prose passages. Preserved the cross-column
OKUṬ and KAṆ comparisons and original underlined k typography. ESEL's example
retains its existing reading uncertainty rather than silently emending it.

Found an explicit entry-level reference absent from the printed bibliography:
Roy, The Mundas and their Country, 1912, p.400 (KAṄAR). Added it to the
reference-resolution audit after a title/author registry search found no
match. Canonical reference creation remains pending. No comparative forms
were promoted into ancestry edges. Key coverage, lexical evidence hashes,
NFC and scoped whitespace checks passed; lexical preview remains unchanged.

513 prose records, canonical integration, final graph/audit work and full
build/suite remain outstanding. Browser refresh remains unrequested.

### Ollari comparative prose, pp.55–56

Previous turn was progress: p.53–54 prose and the additional Roy citation were
saved. Revalidated manifest before reviewing 34 further records on printed
pp.55–56/PDF 66–67, using full columns and closeups. Combined coverage is
178/657 records and 156 passages. Preserved KAR-'s usage parallel, KARKE¹'s
calendar note, KĀRUP's cross-column comparison, and unresolved printed Pa.
Underlined kh spans were preserved on both characters, with a documented
correction to the earlier p.54 markup. No lexical output or graph changed.

Page/key completeness, unchanged lexical-evidence hashes, NFC and scoped
whitespace checks passed. Three previously identified prose reading flags
remain, plus the unresolved Pa. abbreviation. Remaining work includes 479
records' prose, reference/sidecar integration, graph review, final audit,
canonical installation, and full build/suite in an authorized environment.
Browser refresh is unrequested; no heavy local work was run.

### Ollari comparative prose, pp.57–58

Previous turn classified as progress; current manifest revalidated at 178
records before reviewing pp.57–58/PDF 68–69. Added 34 visually checked records,
bringing coverage to 212/657 and 189 passages. Preserved cross-column entries
and explicit headword locator into The Parji Language (Vocabulary, kelay-),
attaching it to bibliography item 4 for canonical reference resolution.

Loanword and language qualifiers scoped to comparative forms remain scoped
there, without becoming Ollari donor edges. Key coverage, lexical hashes and
NFC checks passed; closeups corrected Pj./Naik. kēd- to plain d. No lexical
rows or graph edges changed. 445 prose records remain, plus reference/sidecar
integration, final graph/audit work, installation and full-build/full-suite
validation. Browser refresh is unrequested.

### Ollari comparative prose, pp.59–60

Previous turn classified as progress. Revalidated authoritative manifest, then
visually reviewed all 50 records on pp.59–60/PDF 70–71. Combined coverage is
262 physical records and 220 ordered passages. Preserved compound-component
comparison scope at GŌTI ILENḌ, the Bengali/Persian comparative derivation at
CŌKA, GOṬA's example, and the literal gloss at KOR-GOṬNA BELE. Added one typed
comparative reading uncertainty for the tiny mark over p at CIPṚA.

Key coverage, lexical-record hashes, NFC and scoped whitespace checks pass.
The lexical preview and prior lexical-only sample remain unchanged. No graph
edges were inferred. Remaining: 395 records' prose, auxiliary reference and
sidecar integration, final graph review and complete audit, installation and
full build/suite in an authorized environment. Browser refresh unrequested.

### Ollari comparative prose, pp.61–62

Previous turn classified as progress. Revalidated manifest and visually
reviewed 56 records on pp.61–62/PDF 72–73 using scan columns and closeups.
Coverage now reaches 318/657 physical records and 259 passages. Preserved
compound-component analysis at TANḌ JIR, the unspecified Bengali dialectal
label at ḌEBRI KI, and distinct honey/bee/honey-bee gloss scopes at TĪN.

Key coverage, lexical-evidence hashes, NFC and scoped whitespace checks pass.
No lexical preview rows or graph edges changed; four earlier prose reading
flags remain. Outstanding: 339 records' prose, reference/sidecar integration,
final graph review and complete audit, canonical installation and full build/
suite in an authorized environment. Browser refresh remains unrequested.

### Ollari comparative prose, pp.63–64

Previous turn classified as progress. Revalidated current manifest, then
reviewed all 40 records on pp.63–64/PDF 74–75 against columns and closeups.
Coverage now reaches 358/657 records and 292 passages. Preserved festival
notes, the DIGAL/DIGEL example and derivation, NIRḌIN's two comparative time
meanings, and the comparison-specific scopes of NĪR and NEY analyses. Added
a typed reading flag for fine marks in the plough comparison.

Key coverage, unchanged lexical-evidence hashes, NFC and scoped whitespace
checks pass. No lexical CSV rows or graph edges changed. Remaining: 299
records' prose, reference/sidecar integration, final graph review and complete
audit, canonical installation, and full-build/full-suite validation on an
authorized runner. Browser refresh remains unrequested.

### Ollari comparative prose, pp.65–66

Previous turn classified as progress. Revalidated authoritative manifest and
reviewed all 44 records on pp.65–66/PDF 76–77 against scan columns. Cumulative
coverage is 402/657 physical records and 322 ordered passages. Comparisons
remain attached to individual numbered PANḌ-/PAR- homonyms and separate PĀṬE
song/beam entries. Preserved noun/verb and contrasting temporal meanings in
PIDIR and PIRĀNḌ. No unsupported graph relations were generated.

Page/key completeness, lexical-evidence hashes, NFC and scoped whitespace
checks pass. Five earlier prose reading flags remain; no lexical rows changed.
Outstanding: 255 records' prose, reference/sidecar integration, final graph
review and complete audit, canonical installation, full build and full suite
in an authorized environment. Browser refresh remains unrequested.

### Ollari comparative prose, pp.67–68

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

### Ollari comparative prose, pp.69–70

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

### Ollari comparative prose, pp.71–74

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

### Ollari comparative prose, pp.75–77

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

### Ollari prose uncertainty disposition

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

### Ollari prose preview and build attachment

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

### Ollari explicit reference resolution

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

### Ollari internal relationship review

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

### Ollari explicit variant integration preview

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

### Ollari source-analysis and plural integration

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

### Ollari canonical-input installation

Installed 880 lexical rows, 509 prose passages and 12 bibliography records after
the fresh 0/20 acceptance audit (24 expanded rows, 16 passages). The reproducible
installer verifies audited hashes and is idempotent; source YAML uses append
order 32. All 42 source/sidecar/transcription-input tests pass and source metadata
validation passes. The two initial sound-profile failures were checker issues,
now resolved: routing uses source YAML, and the global length policy respects
the Asuri profile’s explicit preservation of native-script colons. The earlier
diagnosis of missing Asuri length rules was incorrect. No imported forms or
conversion profiles changed.

Full make all, full suite, compiled identity/graph/deduplication, error report and
formatted-reference verification remain deferred under the resource policy.
Existing CI runs on push/PR, tests only, and has no uncommitted-input dispatch.
No commits, pushes, heavy builds or browser refresh were performed.

### Post-installation focused validation

82 tests passed across sound profiles, dialects, Ollari acquisition, prose
sidecars, transcription input, CUJ Asur and Peterson Turi. Two compiled-source
survival tests were explicitly deselected because the full build is pending;
they remain mandatory gates. Updated the stale Ollari manifest test to assert
installed inputs and pending full validation. All 12 new canonical bibliography
records format successfully with the make_refs Pybtex plain/Markdown formatter.
This smoke check does not replace compiled-reference verification.

### Next source: Das (1987), Kodagu Yerava comparison table

This continuation is progress: acquired the official 193-page census scan,
pinned its SHA256, preserved 13 OCR/context evidence pages, and visually
reviewed the complete printed pp.65–66 table. The publication metadata gives
1987; 1981 is the census year. Package:
`data/other/forms/raw_data/das_yerava_1987/`. Survey-wordlist and OCR addenda
are active. All 42 prompt groups / 84 language cells are accounted for, with
90 comma-separated responses and no lexical-cell exclusions. Three acquisition
tests pass; all 14 snapshot artifacts reproduce byte-for-byte.

Current pending manual inputs have Ravula 100 rows (DravLex only), Paniya 499,
Belari 108 and Pattapu 101. Panjiri Yerava maps to existing Ravula; Pani Yerava
to existing Paniya, supported by source p.143 and Glottolog. Dialects will
preserve the Kodagu varieties without invented village coordinates. A surprising
three-label kinship brace was checked at 300 dpi and retained literally.
Multiple OCR confusions were corrected only in reviewed.tsv, preserving raw OCR.
No ancestry or variant claim follows merely from co-equivalents.

Not installed: importer/audit, transcription profile decision, dialect registry,
bibliography/YAML and fresh acceptance audit are next. Full build, full suite,
compiled identity/graph/deduplication/reference and error-diff gates are still
required on a suitable authorized runner. Browser refresh remains unrequested.
Gorum lexicon discovery found a university link but it was inaccessible; no
Gorum data was acquired or claimed installed. No commit, push or release.

### Das Yerava source-input installation

This continuation is progress. Installed the complete table: 42 prompt groups,
84 language cells -> 90 rows (48 Ravula, 42 Paniya); zero exclusions, unreadable
forms or graph links. Registered ravula_panjiri_kodagu and paniya_pani_kodagu,
without invented coordinates. Source spelling is preserved by explicit YAML
convert:false; Phonemic is blank because this source supplies no phonemic
analysis or usable symbol key. Six co-equivalents are separate keyed responses,
not inferred variant edges. The unusual kinship brace remains literal with a
source note. No grammatical labels occur in the source table.

The importer provides a non-installing preview, per-cell audit, deterministic
sample and hash-gated --install. Repeat installation is idempotent. Seed
2026092106 passed 0/20 material errors against page images; prior complete-table
review covers edge cases. 25 source/dialect/profile tests passed; source metadata
validation passed (198 files, 195 citation keys). The bibliography formats with
Pybtex. The actual parse_file preserves all 90 keys and printed forms with zero
conversion errors. One compiled-survival test was separately run and fails:
0/90 new source keys exist in current CLDF. This is an explicit open gate.

Full pipeline/suite, compiled identities, deduplication, graph, concepts,
references and generated/error diffs remain deferred under the resource policy;
no suitable authorized runner has yet been established. Browser refresh and
representative app views are inapplicable without the user-triggered refresh.
No commit, push, heavy local build or publication occurred. See the source
package README for all files, commands, exact edition and editorial policy.

### Indo-Aryan discovery: Darai provenance mismatch

Current streaming coverage check found Darai 32 compiled/manual rows and Chakma
one row. Both are currently classified as Other in the registry; a future source
integration must review their Indo-Aryan clade rather than copy that fallback.
A promising 2015 Darai survey file was downloaded from Tribhuvan University,
but direct inspection exposed a title/body mismatch: Darai title and opening
acknowledgements, Danuwar chapters and a 210-item appendix with Danuwar survey
districts. No forms were installed from it. The PDF hash, exact conflicting
pages and alternative-source assessment are preserved under
`source_checklists/evidence/darai-survey-mismatch/`. This provenance finding is
progress: it prevents 1,050 cells from being incorrectly assigned to Darai.

The SIL 1973 vocabulary catalogue describes 17 pages of Darai forms without
English glosses; its 1971 English key is a required companion, and the vocabulary
PDF returned 403. Dhakal's 2011 university dissertation is a verified alternative
catalogue lead; acquisition and lexical-scope assessment continue. None of these
discovery leads constitutes an installed source or closes outstanding full-build
gates for the previously installed packages.

Dhakal (2011) acquisition subsequently succeeded: 486-page PDF pinned at SHA256
6ca12ee82a22a16b0394b08e319afd5a94f075111fb8a994ee10d59da234bebf.
Title and contents match the Darai dissertation. It has a native text layer
with legacy phonetic private-use glyphs. Contents show lexical phonology tables,
kinship discussion, numerals, verb paradigms and grammatical morphemes, but no
standalone comprehensive vocabulary appendix. Contents evidence and acquisition
status are saved in the same discovery review. Next step: assess a principled
lexical scope and decode fonts, or recover the complete 1973 vocabulary with
its English key. No new forms were installed in this discovery turn.

### Dhakal Darai phonology chapter: native acquisition and decoding

This continuation is progress: selected complete chapter 3 (printed pp.42–76,
PDF pp.64–98) for systematic lexical extraction, preserving all 40,425 native
glyphs over 35 pages with coordinates and hashes. Package:
`data/other/forms/raw_data/dhakal_darai_2011/`. Font decoding covers every
private-use glyph in the chapter; three acquisition/decoding tests pass.
Original PDF content order fixes two visually confirmed nasal-placement errors
that ordinary positional text extraction introduces. No OCR is needed.
No lexical records are yet parsed, accepted or installed. Chapter scope and
all deferred metadata/audit/profile/full-integration gates are recorded in the
package README. The previous source-mismatch discovery turn is progress,
not a blocked turn. No full build, commit, push or browser refresh performed.

### Darai chapter lexical inventory

This continuation is progress: a reproducible glyph-anchored inventory now
accounts for 495 quoted spans, with 444 slash-delimited form candidates and
51 explicitly triaged context cases (32 bare-form candidates, 18 interlinear
sentence examples pending scope review, one metalinguistic exclusion). An
additional malformed-quote lexical example is preserved separately so the
scanner does not silently omit it. Six acquisition/inventory tests pass,
including quote/apostrophe handling, multiline table glosses, repeated source
attestations and contradictory source gloss preservation. 477 potential lexical
occurrences are accounted for, but acceptance, full structural completeness,
prose-only shared glosses, grammar, references, transcription and canonical
installation remain pending. No forms installed this turn; full integration
gates for earlier sources remain open. No heavy job or publication performed.

### Darai contextual decisions and proposal

Progress: visual review of pp.43, 56 and 73–75 resolved seven prose additions,
including a shared-gloss pronunciation form and explicitly described suffixes.
The reproducible proposal now has 485 lexical candidates and 17 exclusions
(16 multiword clauses, one metalinguistic term), accounting for all 495 quoted
spans plus seven additions. Two one-word negative inflections are retained.
Four explicit source-local variant claims have resolvable candidate keys;
two Nepali-loan claims remain prose/tags without invented donor edges. Seven
focused acquisition/inventory/proposal checks pass. No Darai rows installed:
grammatical labels, transcription and metadata, auxiliary citations, complete
chapter review and fresh acceptance audit remain pending, followed by the
previously recorded full integration gates. No heavy job or publication.

### Darai grammar and source diacritics

Progress: parsed explicit source grammatical labels into structured tags for
30 of 485 candidates, retaining raw labels in the audit. Four purely grammatical
glosses are intentionally blank. Added prospective, non-past, classifier and
high-honorific to both shared registries, supported by preserved thesis context
pages; prospective is not collapsed into future. Three retroflex-dot placements
were visually corrected with original extracted strings retained. Sixteen
focused Darai/shared grammar checks pass. No canonical Darai installation yet;
transcription profile, metadata/references, complete structural review, fresh
acceptance audit and full compiled integration gates remain open.


### Darai transcription review

Added source-specific profile and reproducible 485-row transcription preview,
with literal originals, grammar, variant targets and loan prose. Source
quality contrast ə/a survives; no vowel length is invented. Visually verified
p.51 alveolar affricates map c/dz → ʦ/ʣ and glides j/w → y/v. Two internal
spaces normalize only in house forms; source originals remain available.
17 focused Darai/shared grammar tests pass. Still not installed or accepted:
whole-chapter completeness, bibliography/dialect/classification, fresh seeded
audit and full integration gates remain open. No browser refresh requested.


### Dhakal Darai source installation

Completed native-text plus visual review of all 35 chapter pages; accounted for
495 quoted spans and 7 explicit prose additions: 485 rows installed, 17 excluded.
Fresh seed 2026092107: 0/20 material errors. Four variants, two unlinked donor
attributions, six typed uncertainties; no inherited claims inferred. Original
and phonemic source notation survive actual parser conversion with no errors.
Registered Chitwan and explicitly attested Pidrahani (6 acoustic words), with
blank dialect coordinates. Corrected Darai base clade to Bihari. Added thesis
and six auxiliary bibliography records; two printed Kotapish 1973 citations
retain an explicit bibliographic ambiguity. 22 focused tests plus 20 shared
sound-profile/dialect tests passed; repeated install is stable.

Full build/full suite and compiled verification remain open; streaming source
key check confirms 0/485 currently compiled. Added a compiled-survival test
that must pass after the full build. Browser refresh remains unrequested.

### Remote build resource failure and scheduled retry

The frozen 24,316-file snapshot passed hashes and read all five new source CSVs. The combined `make_cldf.py` phase terminated with exit 137; its driver process disappeared and the dedicated validation window closed. The login user cgroup has memory.max=1,610,612,736 bytes and reported four OOM kills. No source-parser error was reported before termination. Full build, full suite and compiled survival remain unpassed.

A streaming interrupted-output audit is live (PID 3948946, ~32 MB RSS); restoration is restricted to audited changed files with verified baseline hashes. Packaging and CPU-only `sbatch` submission are queued behind that audit in `main:jambu-checks`. The retry requests one CPU, 16 GiB, eight hours on the existing sc-freecpu partition. Scripts and runner metadata are under `tmp/diversity-validation/`; shared staging is `/nlp/scr/aryaman/jambu-validation/20260921-diversity-batch`. The batch executes all validation gates sequentially in node-local storage and archives results for review. Submission has not yet been verified.

Source discovery separately verified Pulin Bayan Chakma, *Chakma Dictionary (Chakma–English)*, first edition 1993, Arts & Culture Department, Chakma Autonomous District Council, Kamalnagar. The 619-page scan has no text layer on sampled pages, and printed entries provide Bengali-script headwords, Roman transcription and English definitions. Discovery evidence remains in `tmp/pdfs/chakma-discovery/`; it is not part of the frozen batch or an installed source.

The interrupted snapshot audit completed with zero changed and zero missing files; the evidence is preserved in `tmp/diversity-validation/interrupted-output-inventory.json`. The 2.7 GiB snapshot archive was staged successfully. Scheduler submission required the existing `nlp` account (confirmed by account associations and another existing job); submission then succeeded as job **17539388** on `sc-freecpu`, one CPU, 16 GiB, eight hours. No generated artifacts have been copied back or browser database refreshed.

### First completed pipeline output; regression failure under investigation

Job 17539388 completed all data-generating stages through `make_refs.py`, but `make all` failed its final manual-survey regression: Rajasthani source-bearing forms were 15,891 versus the test expectation 15,887 (other three tests passed). This is not yet classified as a stale test: the extra nodes are being compared with the baseline. The job archived its generated data and output inventory. Follow-up job **17539459** is running the full suite and independent source-survival/prose audits sequentially on `visionlab18`, worktree `/tmp/jambu-audits-aryaman-lqn3un`. Full ingestion validation remains pending, with no generated outputs copied into the shared local checkout.

The Rajasthani comparison found **15,891 before and 15,891 after**, with zero added or removed source-bearing IDs (`rajasthani-count-review.json` in the audit worktree). Therefore the failing 15,887 expectation was stale before this ingestion. The exact expectation in `tests/test_manual_survey_etymologies.py` was corrected to 15,891; accepted-link floors, source-owned link checks and graph-status assertions remain intact. This test-only correction still requires execution against the built output.

### Compiled survival passed; full-suite collection repairs

Job 17539459 confirmed all **3,807/3,807** new source keys and compiled nodes, zero lexical/citation/tag/graph problems, and zero existing aliases retargeted. Ollari prose passed **509/509 blocks across 466 entries**, with no missing or extra blocks. Reports were recovered from the structurally valid temporary archive after tar reported its root directory changed during a concurrent read-only diagnostic report write; `tar -tf` passed. Reports are in `tmp/diversity-validation/reports-17539459/`.

Full pytest stopped in collection because Bhumij and Noira both named their source-local module `test_preintegration_contract.py`. Bhumij was renamed to `test_bhumij_preintegration_contract.py` and its helper loaded under a unique module name. Focused execution then exposed Noira metadata stale after an existing conversion-profile edit: only the profile hash had been updated in `source_manifest.json`, leaving its outer hash and generated profile inventory stale. The outer hash was reconciled, the existing audit generator refreshed inventory/manifest, and assertions were updated for the three already-present long-vowel sequences and their longest-match consumption of all length marks. No lexical rows or conversion mappings were changed. Both source-local suites now pass **13 tests**. Pinned source PDFs are included in the remote repair package so the full suite retains those checks. The full-suite rerun also includes the corrected Rajasthani count.

### User-directed execution boundary

The user explicitly instructed: no remote work, and no database builds unless directly requested. Continue source research, extraction and source-level checks locally. Job 17539512 was cancelled at user request; do not restart it. Its full-suite result remains incomplete. Build-dependent gates are now explicitly deferred by the user, not passed. Existing downloaded generated artifacts remain scratch-only. Hash verification of the nine changed outputs covered by output-inventory.json passed locally; entry-texts.csv is also present. No canonical compiled files have been replaced.

### Chakma source preparation, local only

Prepared Niranjan Chakma (2010), *The Chakma Vocabulary & Terminology*,
TTAADC Language Wing, ISBN 978-93-82172-08-6. The 110-page public scan is
pinned by hash; no open licence is inferred. The main three-column wordlist
covers printed pp.19–46 / PDF21–48. Later lexical, geographic-variation and
comparison chapters remain pending; Brok-Skad is not silently mapped to Bru.

The new `data/other/forms/raw_data/niranjan_chakma_2010/` package preserves
2,298 OCR lines, page/word coordinates, extraction settings and a provisional
731-record review preview. Fixed-column clipping was detected and a second
pass uses per-page crops. Fourteen missing aligned OCR cells were visually
confirmed to contain text (eight Chakma, six Bengali); they remain explicitly
unreviewed. Ninety-two unpaired native-column lines are preserved. No Chakma
forms are installed, no acceptance audit has passed, and source transcription,
metadata and complete structural review remain open. All work was local,
one OCR thread at a time, with no database/data build or remote commands.

Focused preparation checks passed: **3 tests** for coordinate preservation,
review-output reproducibility and explicit retention of OCR omissions and the
wrapped botanical gloss. These do not certify transcription quality.

### Chakma OCR recovery and first visual transcription review

The official Bengali `tessdata_best` model was downloaded to local scratch
and pinned by SHA256. Its embedded English fallback produced Latin guesses;
explicitly disabling that fallback recovered all 14 previously missing aligned
cells. Current evidence: 2,313 raw lines, the same 758 English anchors and 731
candidates, 93 retained unpaired native lines, zero missing candidate cells.
Earlier v2 evidence remains in the source package. This is an alignment result,
not a claim of accurate spelling.

The first printed page now has 27 manually reviewed headword readings in
`visual-review.jsonl`, of which seven retain explicit glyph uncertainties.
The 14 recovered cells also have visual readings, with three conjunct/letter
uncertainties. The grammar introduction (printed p.5) explicitly discusses
problems representing Chakma in Bengali letters; source-script preservation
with blank Phonemic is therefore the proposed route. Publisher location alone
will not be used to infer a Tripura dialect. Four focused evidence tests pass.
No source CSV installation, database build or remote execution occurred.

### Chakma whole-row recovery and expanded visual review

Visual review now covers **244 entries on printed pp.19–27**: 203 have no
remaining glyph flags and 41 retain explicit transcription uncertainty. Three
English-gloss discrepancies are recorded separately from the printed wording
(Worm/warm, Spout/sprout, Torm/thorn). A complete source row, “Dancing hall”
on printed p.26, was absent from whole-column English OCR despite both native
cells being present. It was restored from visual evidence and isolated-cell OCR
with exact page coordinates. The draft therefore has **732 candidates**, not
731; 91 native-column lines remain unpaired. This demonstrates why alignment
alone is not a completeness audit. Six focused evidence tests pass. Review of
remaining pages, uncertain glyphs, complete metadata and installation remain
open. No database build or remote execution occurred.

### Chakma review through printed page 35

Reviewed **463 headwords across printed pp.19–35**, including 85 explicit
letter/conjunct uncertainties and four separately recorded gloss discrepancies.
A second complete row, “Linguistic” on printed p.35, was missing from English
column OCR. Visual evidence plus isolated-cell OCR restored it with exact page
coordinates; there are now **733 candidates** and 89 unpaired native lines.
Both separate printed occurrences of শুলোনি “Pain” on p.33 remain source records;
no source-internal deduplication was applied. Seven focused evidence tests pass,
including whole-row recovery, review provenance and repeated-record retention.
The remaining 11 main-wordlist pages, uncertainty review, source metadata and
installation are unfinished. No database build or remote execution occurred.


### Chakma main-wordlist first review complete; acceptance still pending

All **733 candidate headwords on printed pp.19–46** now have a first visual
review: 589 have no remaining glyph flags and **144 retain explicit transcription
uncertainty**. The four gloss discrepancies remain separate from printed wording.
An OCR-added comma in the shoulder entry was removed after full-page inspection;
its uncertain virama readings remain flagged. All 89 unpaired native OCR lines
have a separate provenance-preserving exclusion ledger: 31 section-heading
fragments, 30 column headings and 28 page footers. No lexical row is excluded by
that ledger. Two previously recovered whole-row omissions remain in the 733.

**Eight focused preparation tests pass (0.21 seconds)**, including full visual
review coverage and preservation of every excluded raw OCR line. This is not a
fresh seeded lexical acceptance audit. Uncertain readings, later book sections,
metadata, source-preservation settings and source CSV installation remain open.
Per the user's standing instructions, no remote execution or database build was
performed. Compiled integration and browser QA remain explicitly deferred.


### Chakma second glyph review, printed pp.19–28

Reviewed enlarged crops for **45 flagged entries** from PDF21–30. Sixteen
readings are resolved, including separate k/t in the morning-star term, the
nasal conjunct in sunrise/sunset, the repeated ladder spelling, and explicit
viramas in waterfall, ventilator, sickle and lamp-stand terms. The other 29
remain explicitly uncertain; no inferred phonemic readings were added.
The append-only second-review ledger records previous spellings and notes,
crop evidence, decisions and current readings. Across the full 733 candidates,
**128 transcription uncertainties remain** and 605 have no current glyph flag.
Nine focused evidence tests pass (0.23 seconds), including preservation of
second-review history and unresolved flags. Fresh acceptance audit, remaining
book sections, metadata and installation remain pending. No remote work or
database build was performed.


### Chakma second glyph review through printed page 38

Reviewed 59 further flagged readings on PDF31–40 against enlarged scan crops.
Twenty-one were resolved; 38 retained uncertainty. Corrections include the e
vowel in the drama term (গ্যেন), and previously omitted explicit viramas in
জিল্গাত and আল্গা জিল. Source-specific ধ in পুধি and independent vowel sequences
remain unchanged rather than being normalized to Bengali cognates. The second
review now covers **104 of the original 144 flagged entries**: 37 resolved and
67 retained. Forty flags still await their second review. Across all 733 rows,
**107 remain uncertain** and 626 have no current glyph flag. Nine focused checks
pass (0.22 seconds). This is still source preparation, not lexical acceptance
or completed ingestion. No database build or remote execution occurred.


### Chakma second pass complete; first fresh audit fails

The second glyph review now covers all 144 originally flagged records: 53
resolved, 91 still uncertain (642 of 733 total candidates without current glyph
flags). It also reopened the shoulder entry's punctuation: the earlier claim
that its comma was purely OCR-added is not established by the enlarged scan.
The reading remains provisional and must not be split into alternatives yet.

The reproducible `audit_sample.py` command sampled 20 of all 733 candidates with
seed **2026092108** and preserved review/candidate snapshots plus the review
file hash. Visual comparison found **2/20 material errors**, not a pass:
extra ব in the mid-day headword (corrected to দিবুচ্যা), and an OCR-added bar
before Sclera (removed from reviewed gloss, retained in raw English anchor).
All English glosses were screened for comparable unexpected punctuation; no
other such glyph was found. The failed sample and its source evidence remain
preserved; a fresh sample is required. Ten focused evidence tests pass (0.23s),
including both regression cases. Source installation and metadata remain open.
No database build or remote execution occurred.


### Chakma fresh transcription sample and multiple-form scope

Fresh seed **2026092109** sampled 20 current records and found **0/20 new material
errors**, including one already flagged Poinciana spelling that remains
unresolved. This is a transcription sample, not proof of all spellings or an
installed-output audit. The previous 2/20 failed sample remains preserved.
Visual review of all 12 comma/slash cells supports 11 independent coequivalent
pairs; no variant relationships are asserted. One eye-ball expression has
unresolved shared-prefix scope, explicitly retained in `multiple-form-review.json`.
No repeated prefix is inferred. Source importer, metadata and installation
remain pending; no database build or remote execution occurred.


### Chakma source-level draft importer

Added a deterministic draft importer and froze all 733 physical source-cell
identities. It emits **742 proposed rows**: 733 source records minus two
withheld records plus 11 additional coequivalents. The damaged warm headword
and unresolved eye-ball slash scope stay fully represented in the per-record
`draft-audit.jsonl`. **97 draft rows carry uncertainty flags**; script is
preserved in Form/Native, Phonemic stays blank, and no relationships are
invented. Three explicit English-gloss corrections retain printed evidence
and typed gloss uncertainty. Thirteen focused tests pass (0.27s), including
accounting, alternative expansion, uncertainty, source-script preservation,
and stable keys under review reorder/spelling corrections. Bibliography,
settings, language metadata, inventory and importer-output audit remain open.
No canonical source CSV was installed; no DB build or remote work occurred.


### Chakma preservation settings, bibliography and character inventory

Prepared source-local bibliography and YAML settings, both still outside the
installed source directory. The YAML disables conversion and implicit alternate
splitting and protects source-record identities. A complete reproducible
55-character inventory records counts and example keys: Bengali-script symbols
plus spaces, hyphens and apostrophes. No phonological mapping is asserted.
The bibliography parses and records the pinned scan, imprint, scope, editorial
attribution and licence limits. Glottolog confirms Chakma / chak1266 / ccp;
source-local dialect and registry clade review remain explicit. Fourteen focused
checks pass (0.32s). Installation and importer-output acceptance audit remain
open; no database build or remote execution occurred.


### Chakma main-wordlist source installation (not a database build)

Source-output seed **2026092110** found **0/20 new material errors**, with one
existing uncertain spelling; the installed CSV reproduces the audited sample.
Installed **742 source rows** from 733 cells (two withheld, 11 extra coequivalents),
97 tagged uncertain, zero relationship edges. Registered source YAML and
bibliography, with Bengali-script preservation and no phonemic conversion.
Corrected Chakma's registry clade Other → Eastern using the official Chakma
Autonomous District Council language page, corroborated by ASJP's Glottolog
lineage. Retained quality-C coordinates; no publisher-based dialect invented.
Fifteen focused tests pass (0.96s), including a source-only `parse_file` check:
742 rows survive parsing, zero conversions, exact source spellings preserved.
The full data pipeline was not run, so compiled survival, graph integration,
full suite and browser gates are not claimed. Remaining book chapters, residual
uncertainties and the two withheld cells remain open. No remote execution or
DB build occurred. Goal remains active.


### Diversity selection: Gutob and Gorum survey supplement

Returned to breadth after the Chakma main-wordlist source installation. Current
compiled streaming counts show Gorum 107 and Gutob 392 (snapshot evidence only,
not proof of all pending source coverage). Belari's Lindgren source is already
present; Palu Kurumba requires identity/Attappady overlap review. A Gorum lexicon
university link remains unavailable, so no acquisition is claimed from it.

A better available target is the already pinned JLSR 2022-004 Appendix B:
Tikrapada Gutob and Kinumun Parenga Parja were fully extracted but excluded as
controls in the Bonda/Didayi package. Created a separate supplemental preparation
package with 420 raw source cells and exact upstream/PDF hashes. Initial parsing:
eight disqualified cells, six missing responses, 448 response segments before
review. Five cells repeat an identical spelling under multiple similarity-group
numbers (six redundant segments); these must not become duplicate variants.
The original package remains unchanged. Visual review, response-scope decisions,
metadata, profile, importer and installation remain open. No DB build or remote
execution occurred.


### Gutob/Gorum response structure and paired prompts

Visually inspected six source pages and recorded **78/420 cells** for
site attribution, response boundaries and prompt context (not final glyph-level
acceptance). Confirmed repeated Parenga ear and both month transcriptions under
different similarity numbers. The response policy preserves those labels while
collapsing exact same-cell repetitions; distinct responses remain separate
attestations without variant or cognacy edges. Paired prompts 182–201 retain the
complete English wording with typed grammatical-scope uncertainty: neither
response order nor similarity codes justify an inferred imperative/past label.
Pronoun prompts will preserve distinctions across item IDs. Page continuations
not yet viewed were excluded from the reviewed ledger. All recorded cells were
checked against exact source raw responses. Supplement installation, remaining
visual review, metadata/profile and acceptance audit are still open. No remote
execution or database build occurred.


### Gutob/Gorum reproducible draft parser

Added a non-installing parser with an exhaustive 420-cell audit. It proposes
442 rows after eight source-disqualified cells, six unanswered cells and six
same-cell duplicate segments are accounted for. Every repeated similarity code
and original response position remains auditable. Fifty-seven paired-prompt
responses carry grammatical-scope uncertainty; no invented tense assignment or
variant edge. Pronoun features are structured and distinct prompt identities
are retained even when spellings coincide. Three focused tests pass (0.04s).
Remaining visual review, source-local dialect metadata, sound-profile coverage,
seeded acceptance and installation remain open. No remote work or DB build.


### Gutob/Gorum profile coverage and printed corruption

The inherited profile left unknown symbols in 20 rows. Prepared a separate
candidate profile retaining q, ø, ɕ and half-ring marks, with established ñ and
length conventions. Removing inherited digit pass-through exposed Gutob item83:
PDF32 visibly prints `b18 ti̪`, also present in neighbouring Bonda lists. This
is now explicitly withheld as source corruption, not silently repaired.
Updated draft count is **441**, with 420 cells fully audited, one corrupt
withholding, eight disqualifications, six missing responses and six duplicate
segments accounted for. All 441 draft forms tokenize without unknown output;
four focused tests pass (0.19s), including NFC/NFD normalized equivalence and
retention of unusual symbols. Profile acceptance, remaining visual review and
source installation remain open. No remote work or database build occurred.


### Gutob/Gorum glyph review, PDF22–24

Added a reproducible target-cell crop generator with pinned-PDF and per-site
anchor-count checks. Visually compared 42 response cells on printed17–19 with
the native-text ledger, including diacritics, multi-response punctuation and
similarity labels; no transcription changes found. q and ø are visibly present
and remain preserved. Saved exact crop locators and per-cell decisions.
Disqualified headings fall outside these crops and were not falsely counted
as reviewed. Remaining pages, metadata, profile acceptance and installation
remain open. No remote execution or database build occurred.


### Gutob/Gorum glyph review through PDF34

Reviewed 140 additional target cells on PDF25–34, bringing the glyph-review
ledger to **182/420 cells**. No transcription changes were needed.
Confirmed Gorum item69's explicit no-entry marker and Gutob item83's printed
corruption (still withheld). Gorum item62 repeats ʋolɐ with group labels 2 and4;
its one-attestation treatment is now visually confirmed. Multiline response
extent and diacritics were checked, with exact page crops retained. Disqualified
headings still require full-page confirmation outside these target crops.
Remaining pages, dialect metadata and acceptance remain open; no remote work
or database build occurred.

### Gutob/Gorum local review checkpoint

User boundaries reaffirmed: no remote execution and no database build without a direct request. Continued only local source review. Recorded 156 additional target-cell visual checks for PDF35–45, bringing the glyph-review ledger to 338/420 source cells. No transcription changes arose. All five same-cell repetition cases now have visual confirmation; six redundant segments remain audit-only. The crop locator handles PDF37's merged KinumunParengaParja label while still enforcing anchor counts. Four focused parser/profile tests pass (0.20s). Draft remains 441 proposed rows, none installed. Remaining: PDF21 and PDF46–50 response review, disqualified headings, metadata/profile acceptance, fresh output audit, and source-file installation. Compiled/database/browser gates remain deferred by the user's explicit instruction.

### Gutob/Gorum complete visual review and site registration

Progress: all 420 target source cells now have visual-review records. Final pages and eight source-disqualified cells were checked against rendered PDF pages; no transcription corrections were needed. Registered Tikrapada Gutob under gu and Kinumun Parenga Parja under go, with blank coordinates and site-level Glottocodes rather than borrowing base-language points. Importer now emits the registered dialect tags; draft audit regenerated. Source methodology supplies no justification for inferring length from unmarked i/u or for erasing tense/lax vowel distinctions, so the source-local profile preserves those qualities and converts explicit length only. Difficult-example outputs were refreshed. Nine focused importer/profile/registry checks pass (0.18s). Source rows remain uninstalled pending profile routing and a fresh seeded output audit. No remote commands, full data build, or browser database build were run.

### Gutob/Gorum source files installed

Installed 441 source rows after a fresh seeded 20-entry lightweight-parser-versus-scan audit found 0 material errors (seed2026092111). All420 source cells have visual review. Two village dialects are registered without unverified coordinates. Profile routes gu/go specifically while preserving the existing gt/re route; a single canonical bibliography key covers both packages. Source-key deduplication preserves prompt distinctions. Updated source inclusion/provenance metadata. Six redundant segments are audit-only; eight cells disqualified, six unanswered, one source-corrupt salt response withheld;57 emitted rows carry typed grammatical-scope uncertainty. No ancestry/variant claims inferred. All16 focused supplement/original-package tests pass (0.68s). CSV installation and small parser tests only: no remote execution, full data build, reference regeneration, full suite, database refresh or app QA. These compiled gates remain explicitly deferred. The broader diversity goal remains active.

### Next diversity source: Birhor Living Dictionary, pinned inventory

Previous turn was progress: Gutob/Gorum source inputs installed and tested. Current source search found a much richer Birhor dictionary than the earlier20-word lead. Current compiled Birhor count323 is only selection evidence, not a total of pending source inputs. No Birhor-specific dictionary YAML/bibliography/package was found. Acquired the public Living Dictionaries snapshot locally; read-only integrity passes; pinned SHA and revision in birhor_living_2026/source-manifest.json. Inventory freezes2746 entries/senses with stable IDs, IPA/native forms, all lexical annotations and91 auxiliary reference records. Media and speaker/account metadata excluded from lexical export. Archived2021 CSV/PDF and live source distinguished; PDF download returned a challenge rather than a PDF and was not used. Reuse basis requires resolution before installation (current terms/archived metadata recorded; prior Kharia permission is not silently generalized). Parser, sound profile, annotation review, seeded audit and tests remain. No installed Birhor rows yet; no remote execution or database build.

### Birhor sense and annotation preparation

Progress: added a reproducible source-local sense preparer and checked3,097 candidate meanings against the2,746-entry topology.259 entries have multiple numbered meanings; one other has only a leading1. Grammar markers are parsed per meaning, leaving botanical author parentheses and semantic qualifiers intact. Preserved23 headword/IPA discrepancies as separate source fields, flagged four optional-segment cases and two botanical scope cases. A focused test caught one missing English gloss; added typed uncertainty and regression coverage rather than inventing a translation.29 unique entries currently need review. Ten morphological strings, one interlinear string,98 notes and90 source-attributed entries remain preserved for further mapping. Two focused preparation tests pass (0.09s); no rows installed. Reuse basis, source references, profile and seeded acceptance audit remain open. No remote work or builds.

### Birhor provisional source rows and profile

Prepared3,10015-column draft rows from2,743 eligible entries; all2,746 entries are audited. Three withheld entries: two unresolved botanical headwords and one absent English gloss. Four single-segment optional spellings produce six additional variant readings with resolvable keys.25 emitted rows preserve typed headword/IPA disagreement. Raw headword, IPA and native script remain distinct. Source notes and10 morphological/one interlinear analyses are preserved, and91 auxiliary records classified as89 botanical references, one ethnobotanical work and one excluded photo credit. Candidate profile covers all3,100 forms in NFC/NFD; four focused checks pass(0.51s). No installation, remote work or builds. Fresh acceptance audit, bibliography/pipeline integration and Birhor reuse basis remain open.

### Birhor reviewed draft; reuse clarification pending

Fresh20-record audit(seed2026092201) against pinned upstream JSON found0 material errors. Actual parser tested onall3,100 draft rows with temporary in-memory settings:3100 conversions, no conversion errors or lost source fields. Added source-anchored pronoun/person-number decisions for five entries; six focused tests pass(1.09s). Prepared three bibliography records; every draft citation resolves and all format successfully without running make_refs or changing generated references. Main corpus files remain unmodified for Birhor. Asked whether the existing Living Dictionaries permission covers Birhor; local record names Kharia, so do not silently extend it. Source installation awaits that clarification; broad source-discovery goal remains active and independent work is possible. No remote commands or database/data builds.

### Dravidian discovery and Pattapu primary-source completion

While Birhor reuse clarification remains pending, continued independent discovery. Found a registry duplication relevant to prioritization: PaluKurumba and Kurumba both carry atta1243; the former's4compiled rows understate coverage because the latter has463 manual Palakkad-survey rows. No registry migration made. KIRTADS2017 grammar is indexed with substantial Attappady vocabulary, but direct official download returned404; retained as a lead, not an extraction.

Selected ISO request2013-020 Pattapu: pinned actual9-page PDF,210 numbered wordlist prompts onpages6–8. Lindgren thesisPDF28/printed27 explicitly cites IRA(2013), so current101Pattapu rows are a derived selection; this work extends primary-source coverage and must reconcile overlap rather than claim independent attestations. Rendered page6 reveals visible IPA glyphs missing in native text. Added reproducible210-cell locator scaffold, allmarked unreviewed/non-installable, and source manifest. No lexical readings accepted yet; visual transcription/profile/overlap review remain. Survey addendum active. No remote execution or database/data builds.

### Pattapu embedded-font recovery

Diagnosed missing text: the PDF's Cambria ToUnicode maps15 visible letters/diacritics to spaces. A reproducible source-local decoder now recovers their Unicode from the embedded TrueType cmap with Identity CIDToGIDMap, preserving genuine spaces and original raw extraction. Recovered all210 numbered scaffold cells without OCR or cognate-based guesses. One glyphCID1345 lacks a Unicode mapping, appears in9cells, and remains explicitly U+E000; enlarged rendering shows an arc near affricates but its exact identity/attachment is not asserted. First10cells have glyph-review records. Two tests pass(0.37s), including actual pinned-PDF reproduction. Full visual review, profile interpretation and101-row Lindgren overlap reconciliation remain; no source rows installed or builds run.

### Pattapu local review continued under explicit execution limits

Reaffirmed no remote execution and no database/data builds without a direct request. Compared items11–42 with a scale3 PDF rendering, bringing the glyph-review ledger to42/210 cells. Preserved item11’s two responses and concept-distinct identical arm/elbow forms. No transcription corrections inferred. Checked embedded MATH variant constructions; no relationship resolves CID1345, so nine readings retain explicit uncertainty. Source remains a preparation package, not an installed ingest. Full visual review, sound-profile decisions and Lindgren overlap reconciliation remain open.

### Pattapu visual pass complete; overlap discrepancies exposed

Previous turn was progress (42 glyph comparisons). Completed visual comparison of all210 prompts.199 retain recovered readings pending profile acceptance, nine have unresolved arcs, item69 needs vertical-stroke notation review, and item73 is blank. Fixed a systematic footer leak by detecting the full-page footer before column cropping; original raw scaffold preserved. Reconciled all101 Lindgren-derived rows to primary prompt candidates:93 ordinary correspondences and eight semantic/grammatical exceptions. In particular, derived cold contains primary chili form; second-person gender contrasts replace primary formal/informal prompts; dual/exclusive and unmarked/inclusive analyses differ; third-person remoteness/animacy is not explicit in primary prompts. Existing entries unchanged, and none treated as independent attestations. Four focused tests pass(0.31s). Source installation, profile/grammar decisions, overlap resolution and seeded output audit remain. No remote commands, full builds, database refresh or publication.

### Pattapu lexical draft and grammar preparation

Previous turn was progress (complete visual ledger, footer repair and overlap audit). Added reproducible draft importer:201 proposed rows from199 prompts; ten transcription cases withheld and one unanswered prompt excluded, all210 audited. Breast/snake each produce two source responses without inferred variant edges.104 proposed prompts extend beyond the existing derived selection. Pronoun grammar follows primary formal/informal and dual labels; full elicited clauses remain intact, imperative prompts retain their forms, and existing Lindgren rows remain unchanged. Recorded44-symbol inventory. Non-wordlist request pages supply no transcription key; sound-profile acceptance remains open. Five focused tests pass(0.35s). No installation, remote work, full pipeline, database build or publication. Remaining profile/overlap decisions, dialect/reference integration and seeded output acceptance audit precede source installation.

### Pattapu profile and source-row audit

Previous turn was progress (201-row draft and44-symbol inventory). Added conservative profile covering all201 forms in NFC/NFD. Preserve stress, superscripts, dental marks and tie placement; four tie-bearing rows explicitly uncertain. Actual lightweight row parser converts201/201 with no errors and intact Originals/glosses. Seed2026092202 output audit compared20 source responses/prompts/locators:0 material errors. Prepared local bibliography and Ethamukkala dialect proposal with blank coordinates; no global registration yet. Six focused tests pass(0.42s). Global overlap integration remains before installation. No remote execution, full pipeline, generated-reference refresh or database build.

### Pattapu overlap integration, derivative annotations

Previous turn was progress (profile,201-row parser check, bibliography/locality drafts). Retained distinct primary and derived transcription/analysis with explicit shared provenance. Annotated eight conflicting Lindgren rows with uncertainty and primary-source notes, preserving forms, glosses, phonemic fields, citations, stable keys and source cognate sets. Added source-local review sidecar and importer support with expected-form/gloss assertions so regeneration preserves annotations. Updated derivative audits and bibliography provenance note.17 focused source/preparation tests pass(2.72s), including full reproduction of the three small source input files; compiled test excluded. No full data/database build, remote work or publication. Primary201-row installation and registry integration remain next.

### Pattapu source inputs installed; compiled gates deferred

Previous turn was progress (repeatable derivative annotations). Installed201 primary rows from199/210 prompts, including104 prompts beyond the derived selection. Ten transcription cases withheld, one unanswered, four emitted tie-placement readings uncertain. Breast/snake responses add two rows. Profile/settings(append_order37), bibliography and Ethamukkala dialect registered; coordinates blank. Metadata CLI passes; actual row parser converts201/201 with Originals, glosses, keys and dialect tags intact.19 source tests and13 profile/registry tests pass(eight overlap:24 distinct checks). House alphabet checked; bibliography parsed/formatted in memory. Prior0/20 lexical output audit remains applicable; added dialect checked exhaustively. Full CLDF build, compiled identity/graph/reference checks, full suite, database refresh and app QA remain deferred by user. No remote work, database build, commit, push or publication. Broader diversity goal remains active.

### Next-source selection: Belari primary appendix

Previous turn was progress (Pattapu source inputs installed). Compared current source inputs with streamed compiled counts, avoiding false zero-coverage claims caused by source aliases. Existing compiled snapshot: Kudiya397, Manda539, Koda620, Mahali984, Malto1511, Naiki518, Naikri689, Pengo1015, KolBangladesh303, Belari109; counts are selection evidence, not fresh builds. CIIL Kudiya2017 transcription is a57-page lead, but repository terms prohibit systematic compilation without permission; no lexical download/import. Manda2009 CIIL entry is catalogue material, full dictionary not verified.

Selected the already cached Bhat1971 Belari appendix (PDF126–130/printed119–123). Existing Bhat import excludes it; Lindgren's108 source rows derive from it, so primary completion and overlap reconciliation are required. Thesis printed27 mentions114 comparative concepts; do not equate difference with six words. Pinned130-page PDF SHAa16d98b086813a80a1f86a0ae9da69e12575c99f87202fa1253caf17614f4ab9, saved five-page raw-text scaffold and source manifest. First appendix page viewed; OCR corruption means no scaffold form accepted. Glossary/comparative addenda active. All lexical/paradigm counting, glyph review, profile, overlap and focused validation remain; no source rows yet. No remote execution or database build.

### Belari appendix structural review and first36 readings

Previous turn was progress (source selection and pinned scaffold). Viewed allfive appendix pages. Recorded191 target form occurrences across lexical lists, paradigms, pronouns and explicit suffixes, with repeated forms counted as source occurrences; seven Tulu table cells are controls. Preserved author's uncertainty about borrowing/inheritance and past/perfect interpretation; no graph claims inferred. Saved36 first-pass lexical readings for printed119 with immutable section/column/row keys and raw-page provenance. Two barred-i/cluster readings require closer review; give-to-I/II versusIII-person restrictions remain lexical semantics, not agreement tags. No accepted source rows or registry edits. Remaining155 occurrence transcriptions, second glyph pass, grammar/scope review,108-row overlap and profile/output checks. No remote work or database/data builds.

### Belari first-pass transcription complete

Previous turn was progress (191-unit structure and36 candidates). Reviewed enlarged PDF127–130 scans and completed all191 target candidate occurrences with exact page/section/column/row keys. Retained distinct syncretic paradigm cells, original brave/bravo first-person ordering, literal chilly gloss with typed ambiguity, source we without inclusive/exclusive enrichment, and recipient restrictions in give meanings. Barred vowels in suffix/italic material explicitly await second glyph review. Two preparation tests pass; no accepted output, profile or installation yet. Remaining second visual pass, grammar/morphology handling,108-row derivative reconciliation and source validation. No remote work or database/data build.

### Belari second glyph pass and grammatical scope

Previous turn was progress (191 first-pass readings). Inspected higher-magnification crops and recorded43 second-pass glyph reviews. Corrected one neuter-plural past reading battigo→battɨgo in a separate ledger, preserving original candidate text. Confirmed sigɨṇi/saṭrɨ and preserved literal suffix-table vowel differences. Added reproducible192-analysis preparation for191 occurrences; explicit feminine-singular/plural you expands to two analyses. Cell-specific gender/number scopes retained; source concessive/assertive headings preserved in notes; temporal since gloss not mislabeled conditional. Four focused tests pass(0.06s).148 glyph reviews, derivative overlap, profile and acceptance audit remain before installation. No remote work or database/data builds.

### Belari full second glyph review

Previous turn was progress (43 glyph checks and192 grammar proposals). Completed remaining148 source checks from full enlarged pages and specific italic-example crops. Ten first-pass corrections now recorded separately; three readings remain uncertain (first-person n/ṇ, italic past-form vowel, source ay.i dot). Literal chilly semantic ambiguity adds a fourth uncertain analysis. All192 proposals retain review/uncertainty metadata; source forms and original candidates remain auditable. Four focused tests pass(0.07s). Remaining derivative overlap, duplicate/root relationships, sound profile, rich rows and seeded output audit before installation. No remote work or database/data builds.

### Belari derivative and paradigm reconciliation

Previous turn was progress (complete191-glyph review). Accounted for all108 Lindgren Belari rows with reproducible reviewed correspondences:89 ordinary overlaps, seven derived stems, three unlocated entries(coast/hand/tree), two grammar and two transcription differences, one gloss ambiguity, two recipient-scope generalizations and two unsupported inclusive/exclusive analyses. No primary matches fabricated; existing derivative rows unchanged. Recorded45 source-explicit paradigm-to-bar-root variant proposals, with no historical ancestry/borrowing edges. Source contact and proto-sound commentary distinguished from graph claims. Six focused tests pass(0.08s). Next: sound profile, rich rows/commentary, derivative annotations and source-output acceptance checks. No remote work, graph output changes or database/data build.

### Belari draft source emission and Unicode checks

Prepared 192 draft rows from 191 reviewed source occurrences, preserving 45 explicit
paradigm links, four uncertainty flags and 14 historical-commentary notes without
inventing historical graph edges. The source-local profile preserves the printed
ay.i dot and distinguishes affricate j from glide y. A focused test exposed missing
decomposed-diacritic coverage; explicit NFD spellings now produce identical output
to NFC for every draft form. Seven focused tests pass (0.20s); the source-local
profile policy audit reports no violations. The row parser accepts all 192 rows
and preserves Original, Gloss, Notes, Etymology, Variant_Of_Key and Tags.

The seed-2026092203 sample is prepared but its acceptance audit remains pending.
Draft CSV and parser output are in workspace tmp only; canonical source installation
and derivative annotations remain pending. No remote work or database/data build.

### Belari source-output acceptance and source-file integration

The preceding goal turn made progress: 192 draft rows, Unicode coverage and parser
checks. Seed 2026092203 now has a completed 20-row visual source-output audit with
zero material errors (output-audit-2026092203.json). All five printed pages were
consulted; spelling, glosses, grammar, locators and parent relationships checked.

Installed 192 rows and their source-local profile/settings. Reused bhat1971koraga
without duplicating its registry entry; its existing owner routes Belari first and
retains the Koraga fallback. Extended the canonical bibliography's inclusion and
provenance fields to cover appendix8. No new dialect/location is asserted: the
source has no named village suitable for a more specific attestation.

Eight focused tests pass (0.49s), including parsing the installed CSV with global
metadata, preserving all reviewed fields and checking the shared reference. Source
metadata validation passes (203 files,198 keys). The rows retain four uncertainties,
45 explicit paradigm links,14 historical-commentary entries and no inferred
historical edges. Seven Tulu control cells remain excluded. Lindgren annotations
remain pending; these sources share provenance, not independent field evidence.
Full pipeline/graph/reference output, full-suite and browser checks remain deferred
under the user's no-build constraint. No remote execution or database build.

### Belari derivative annotation integration

The prior goal turn made progress by installing 192 source rows and completing the
source-output audit. Applied 19 reviewed comparison notes to existing Lindgren
Belari records, with seven typed uncertainties. Seven derived stems, three
unlocated entries and two recipient-scope generalizations receive provenance notes
without automatically becoming transcription errors. Original forms, glosses,
phonemic fields, citations, cognate claims and keys are unchanged; no primary
locator is fabricated for coast, hand or tree. The source importer loads both
Pattapu and Belari annotation sidecars and reproduces installed CSVs exactly.
Eleven focused IA/Dravidian tests pass (compiled test excluded), and the added
Belari distinction test passes separately. Full data/database and browser gates
remain explicitly deferred. No remote execution or database build.

### New Koda/Mahali discovery evidence

Inspected the public Visva-Bharati 2024 dictionary front matter and sample entries locally:380 PDF pages, source-reported2452 entries across52 domains. Indian community material could complement existing Bangladesh lists. Discovery record: data/other/forms/raw_data/cfel_koda_mahali_2024_discovery.json. Sample IPA is Bangla, not target-language transcription; target Bengali-script text extraction has missing/reordered glyphs. Introduction explicitly includes coined terminology. No lexical import started; inspect online language-pair data before selecting extraction strategy. Official SPPEL Manda page describes documentation as ongoing without a lexical download.

### Koda/Mahali online schema discovery

Previous turn was progress: Belari annotations and a new dictionary lead. Inspected
the publisher dictionary page and bounded live searches (limit5). Unlike the PDF
sample, online Koda/Mahali records contain target IPA, native form, record ID and
concept ID. Water has two source concepts in different domains; Koda IPA differs
between them, so forms must not be collapsed automatically. Reverse English
responses supply glosses but omit concept IDs. Empty searches return0; the old
table endpoint returns no data. Complete inventory and exact gloss alignment are
therefore still unproven. Saved endpoint, queries, hashes and findings in the
existing discovery JSON. No full harvest or lexical installation performed.
Public pages were fetched and examined locally; no remote execution or DB build.

### Mahali primary edition selected and extracted

Previous turn made progress by exposing API fields and limitations. Publisher publication
index yielded a complete Mahali-headed2024 dictionary with aligned IPA/English labels,
and a Koda2022 XPS whose inspected lexical pages are image-based. Selected Mahali;
recorded edition and coinage caveats, SHA256 and387-page topology. Reproducible local
raw extraction completed. First structural pass finds2450 multiline IPA/grammar
boundaries against2451 reported entries;2441 Description headings. Counts remain
unreconciled and no lexical rows accepted. Font mapping damages native-script text;
IPA/English extraction must be audited separately. No remote execution or DB build.

### Mahali complete candidate topology

The previous turn made progress by selecting and extracting the Mahali edition.
All2451 reported entries now have source-page/item keys and raw candidate blocks.
The missing boundary was Fig on printed380:item2: the source itself prints
-ɖũmbɔr/ without an opening slash, verified against a140dpi page render. The
exception is recorded rather than silently repairing raw source punctuation.
Ten number entries lack Description paragraphs; they remain explicit source
records. Multiline grammar labels are captured across line boundaries.

prepare_candidates.py is reproducible from the pinned page cache and emits no
installed CSV. Candidates retain source grammar, IPA, English label and damaged
native text separately.1176 native candidates contain NULs; none of the extracted
IPA candidates do. Absence of NULs is not proof of phonetic accuracy. Native glyph
recovery, full symbol inventory, semantic-domain and compound-label scope, visual
sample audit and source integration remain. Three focused tests pass (0.07s).
No remote execution, full pipeline, database build or browser refresh.

### Mahali font and transcription inventory

Previous turn was progress:2451 candidates and count reconciliation. Inspected the
embedded NirmalaUI font:251 CIDs map to U+0000, none has a reverse mapping in the
embedded cmap, and shaping tables are absent. This direct recovery method cannot
restore native text; no character mappings invented. fontTools4.65.0 was installed
only in temporary workspace storage for this inspection.

Recorded53 raw IPA symbols and28 entries requiring targeted review. Three high
resolution crops confirm unusual source marks below an opening slash (ripe,
skeleton) or below m (maternal aunt). Preserve source forms with typed transcription
uncertainty; do not silently reinterpret these as regular phonological features.
The review ledger records all three without overwriting raw candidates.25 flagged
entries remain for targeted review, in addition to the fresh random source audit,
native recovery/withholding decisions, grammar/domain scope and profile integration.
Four focused preparation tests pass. No installed Mahali rows, remote execution or
DB build. All compiled gates remain deferred.

### Mahali targeted glyph review and geometric correction

Previous turn was progress: font limitation and IPA inventory. Reviewed all28 flagged
entries in source crops. Found a systematic layout error: elevated affricate ties
were detached, reordered or excluded by default line clustering. New reproducible
extract_positioned.py anchors Latin combining glyphs to overlapping base glyphs on
the same line, using adjacent source glyph order to resolve shared boundaries;
it does not alter Bengali shaping marks. All1377 combining glyphs have geometric
anchors. Prepared2451 positioned candidates separately, changing204 IPA candidates
and retaining211 ties inside IPA. Original candidates/page extraction are unchanged.
Literal question marks, internal dots and unusual combining marks remain flagged;
no phonological substitution inferred. Delimiter-attached marks stay in raw blocks
and issues rather than being forced onto initial consonants. Five focused tests
pass. Fresh source-output audit, native strategy, profile/grammar and integration
remain. No installed rows, remote execution or database build.

### Mahali positioned extraction audit and domains

The previous turn made progress by fixing elevated combining-glyph layout. A fresh
seed2026092204 sample of20 entries matches the printed IPA, grammar, English labels
and entry boundaries with0 material extraction errors. This does not certify Native
or normalized output. Source Grain on306:item2 denotes a weight unit; preserve this
disambiguation. English hyphenated line wraps require joining without a new space.

The contents table contains50 lexical domains, numbered3–52 after two introductory
sections; retain the introduction's reported52 separately. map_domains.py matches
each heading uniquely on its declared page, then assigns2451 keys by heading offset.
This handles mid-page transitions (82:item6 remains a numeral,82:item7 begins
causatives;306:item2 remains a measurement). Six focused tests pass. Raw/positioned
candidates stay distinct; no installed rows or DB build. Next: grammar and profile,
field-level native treatment, normalized draft and output acceptance.

### Mahali analysis and source-local sound profile

Previous turn made progress: fresh0/20 extraction audit and50-domain mapping.
Prepared2451 analysis proposals preserving180 component-label chains in notes
with multiword-expression instead of wrongly assigning all components as whole
phrase POS. Simple POS, source-domain numerals/ordinals/causatives/compounds are
structured. Nineteen transcription/grammar uncertainties retain typed issues.
Grain(306:item2) is disambiguated as a weight unit from its source definition;
English hyphenated line wraps retain hyphens without inserted spaces.

Prepared a59-rule source-local IPA profile;2451/2451 forms tokenize without
replacement characters and NFC/NFD inputs agree. Dental marks, æ/ɔ, unusual
combining marks, literal question marks, glottal stops and dots remain distinct;
IPA glide j maps to y, affricates to c/j. Policy audit has no violations. Eight
focused tests pass(0.41s). Native remains explicitly pending, with raw forms in
the audit. Local Tesseract includes Bengali models, enabling a small crop pilot
before deciding whether further recovery is viable. No OCR run, installed CSV,
remote execution or data/database build this turn.

### Mahali native-script recovery pilot

Previous turn was progress:2451 grammar/profile proposals and8 focused tests.
Ran a sequential, one-thread, ten-headword Bengali OCR pilot at400dpi, preserving
crop coordinates and raw outputs. Initial crops admitted next-line glyphs;
reduced bottom boundary from18 to13 points and reran separately. Results improve
but still contain incorrect letters and terminal signs, plus one blank. The pilot
is rejected for unattended Native-field installation; OCR has not populated rows.

A bounded publisher API Ring query returned two semantic records. Mahali record11360,
concept183, /aŋʈi/, Adornments domain agrees with printed10:item1; its Unicode আঙটি
was visually verified and recorded as one native recovery. The other Ring hit in
General has different IPA and is not merged. One match is not proof that editions
are interchangeable. Next: constrained publisher/printed correspondence or another
reviewed recovery method, preserving per-record evidence. No full OCR, installed
Mahali CSV, remote execution or database build.

### Native recovery review: first publisher comparison batch

Reviewed 21 printed headwords against three local 300-dpi contact sheets and
cached publisher Unicode results. The native recovery ledger now records 21
accepted readings, including the previous Ring decision. Each decision retains
an API record/concept ID, response hash and visual evidence. Comparison keeps
vowel quality, dental marks, question marks and glottal stops distinct; it does
not establish general equivalence between the printed and online editions.
Shirt and Dress remain separate records despite identical forms; the noun and
verb glossed Ring also remain separate. Printed terminal virama and the spelling
চুরি are retained without modernizing the source.

Analysis proposals now consume the explicit review ledger: 21 accepted Native
values and 2,430 pending values, with raw extraction preserved for all 2,451.
Nine focused preparation tests pass. No source CSV installed yet; native recovery,
source-output audit and metadata integration remain pending. All database builds,
full pipelines and remote execution remain deferred under the user's explicit
instructions. This is source preparation progress, not completed ingestion.

### Mahali metadata and existing-source review

Frontmatter pp. 2–3 confirms the 2024 first edition, editors, publisher and ISBN;
pp. 6–8 supplies contributor roles and the West Bengal context. No collection
village or named variety is specified. `metadata-review.json` records canonical
Mahali mapping, contributors as provenance, and no invented dialect or coordinates.
The introduction explicitly permits coinages, so individual entries are not
uniformly labelled traditional vocabulary.

Existing source inputs contain 908 Mahali survey rows across three Bangladesh
villages, 68 Pinnow rows and eight Zoller rows. This is an inventory, not a claim
that lexical overlap has been resolved. The new dictionary remains a separate
source attestation. Source registration and normalized output audit are pending;
no database build or remote work was performed.

### Native recovery: second bounded comparison batch

Thirty additional public searches were cached locally, yielding 31 unreviewed
printed correspondences (including two meanings of Pant). All 31 primary native
headwords were visually checked in five 300-dpi contact sheets and recorded with
crop coordinates, publisher IDs and raw response hashes. Accepted Native readings
now total 52; 2,399 remain pending. Nine focused tests pass (0.40s).

The API supplies an additional word for Shoe (চাটকি) and Bindi (টিকেঃ) that is not
printed as an alternate in the corresponding headword. These remain API evidence
only, not new printed-source variants. Lingerie has repeated spaces in the API;
its accepted native reading uses single word separators, with the raw value kept.

`render_native_review.py` reproduces bounded visual-review sheets from the pinned
PDF and currently unreviewed proposals, recording crop coordinates and resolution.
This helper does not accept readings or install rows. Review artifacts for this
batch are in the local `tmp/pdfs/cfel-koda-mahali/native-batch-2/` directory.
No remote execution or database build was performed. Source installation and
normalized output audit remain pending.

### Native recovery: third bounded comparison batch

Fifty further public searches were cached locally. The cache now contains 100
searches and the correspondence file covers 104 printed entries; two new matches
outside this batch (Axe p47:e6 and Farmer p320:e2) await visual review. Fifty
headwords on pp. 19–27 were reviewed against eight local 300-dpi contact sheets,
with crop coordinates recorded. Accepted native readings total 102; 2,349 remain
pending. Fourteen records have nonblank API-only alternates; these are retained
in the review ledger without expanding the printed edition. A whitespace-only
alternate for Bull is not a lexical variant.

The public downloader now makes at most three attempts for transient network or
selected HTTP errors, with one- and two-second backoff. Permanent HTTP errors
fail immediately. Tests exercise recovery, exhausted retries and permanent-error
behavior without network access. All ten focused tests pass (0.44s).
No source rows installed, database built, or remote commands executed. Native
recovery and normalized output audit remain the next source-preparation gates.

### Native recovery: fourth bounded comparison batch

Forty additional public searches bring the local snapshot to 140 searches and
145 corresponding printed entries. Reviewed 43 primary native headwords against
seven 300-dpi sheets, including the previously pending Axe and Farmer entries.
All 145 correspondences currently cached have explicit visual decisions; 2,306
of the dictionary's 2,451 native readings remain pending.

Thirteen entries in this batch have online-only alternative words; these remain
comparison evidence. The two Chick entries retain their different printed virama
placement (সিম্ হপন versus সিম হপন্), separate keys and domains. Jackal and Fox
also remain distinct records despite identical primary forms. Goat retains the
publisher's final non-joiner after its visible virama, with the Unicode detail
recorded in the decision.

The focused suite now reconciles every accepted native reading with its saved
publisher record ID, response hash, matching printed IPA/domain, and visual
review evidence. Eleven tests pass (0.42s). No source rows installed, remote
commands executed or database built; remaining native recovery and source-output
review are still required.

### Native recovery: fifth batch and review CSV

Forty more public searches bring the snapshot to 180 searches, covering 190
printed entries. Forty-five primary headwords were visually reviewed in seven
300-dpi sheets. Accepted readings total 190; 2,261 native fields remain pending.
Eleven entries in this batch have online-only alternate words. Repeated Pen,
Paper, Computer, Notebook and Glass entries retain their printed keys/domains;
Kettle and Tea Pot likewise remain distinct. Publisher non-joiners are retained
as native-script shaping controls, with explicit evidence notes.

`prepare_draft.py` now writes `review-draft.csv` in the 15-column source schema
and `draft-audit.jsonl` with all 2,451 entry keys, page/item citations, native
review statuses, typed issues, source domains and graph decisions. This stays
inside the raw source package, outside the installed CSV directory. It is a
review artifact: pending Native values are blank and explicitly accounted for;
source settings/bibliography registration and normalized output audit remain
required. The raw IPA drives the planned source profile; no invented historical
relations or duplicated Phonemic values are emitted.

Twelve focused tests pass (0.44s), including exact draft reproduction, schema,
source record accounting and absence of invented relations. No remote execution,
source installation or database build occurred.

### Review-draft output audit (seed 2026092205)

Visually audited 20 randomly selected records from the 190 entries whose native
readings have been reviewed. Four local contact sheets show the printed lexical
blocks; the audit records page/item crops, source IPA, house-profile output,
native script, gloss, tags and citations. No material extraction/conversion
errors found (0/20). Page 307 entry 6 describes glass as the material used in
windows and doors; its draft gloss is now “Glass (material)”, retaining the raw
source label “Glass”.

`output-sample-2026092205.json` freezes the ordered 190-key population and sample,
so later recovery does not silently change the audited set. `audit_draft.py`
reproduces selection and asserts current draft/profile values agree with the
review. Twelve focused tests pass, and the audit command passes.

Limitations: this audit covers only the reviewed-native subset and happened to
sample noun labels exclusively. It does not pass the whole-source output audit,
component-label grammar checks, installed routing, compiled survival or browser
gates. Native recovery remains at 190 accepted / 2,261 pending. No database build
or remote execution occurred.

### Native recovery: sixth comparison batch

Forty additional searches bring the public snapshot to 220 cached queries and
234 printed correspondences. Forty-four primary headwords were visually checked
in seven 300-dpi review sheets, retaining crop coordinates, publisher IDs and
response hashes. All currently cached correspondences have explicit native
review decisions; 234 readings accepted and 2,217 pending overall.

The Chair entries preserve printed ট্যান্ডার মাচি (p48:e1) versus ট্যাঁন্ডার মাচি
(p132:e8), including the nasalization difference. Light noun/adjective matches
remain separate by key and domain. Small Stool has one API-only alternate, which
is not inserted into the printed source. Source-specific pots and baskets remain
separate records rather than being merged by their English head noun.

Regenerated the review-only analysis, CSV and per-entry audit. Twelve focused
tests pass (0.42s); the frozen 20-entry output audit still reproduces. Native
recovery, whole-source output audit and source registration remain pending.
No remote execution, source installation or database build occurred.

### Native recovery: seventh comparison batch

Forty additional queries bring the public snapshot to 260 searches and 281
printed correspondences. Reviewed 47 primary native headwords in seven local
300-dpi sheets, recording crop coordinates, publisher IDs and response hashes.
Accepted readings total 281; 2,170 remain pending.

Six records have online-only alternate words; these remain comparison evidence.
Printed Picture forms ফট / ফটো and Bench forms বেন্‌চি / বিঞ্চি remain separate,
as do the two different Kitchen headwords. The Lamp Stand API contains repeated
spaces, reduced to one word separator with the original retained in its audit.
Three printed Book entries retain separate source keys rather than collapsing.

Regenerated analysis, review CSV and audit. Twelve focused tests pass (0.40s)
and the frozen subset output audit reproduces. Native recovery, whole-source
output audit and source registration remain pending. No source installation,
remote execution or database build occurred.

### Native recovery: eighth comparison batch

Forty additional public searches bring the snapshot to 300 queries and 323
printed correspondences. Reviewed 42 primary native headwords in six local
300-dpi sheets. Accepted readings total 323; 2,128 remain pending. Every decision
retains crop coordinates, publisher record IDs and response hashes.

Six entries have online-only alternate words, retained as comparison evidence.
The two Duster forms (ল্যাতে and বোড মুছোআঃ) remain separate. Side Pillow সিতান্‌
and Pillow সিতেন্‌ retain their printed vowel distinction. String and Thread,
and the two Candle attestations, keep their separate printed keys despite
matching native forms. Tongs preserves its printed final khanda-ta.

Regenerated review analysis, CSV and audit. Twelve focused tests pass (0.42s)
and the frozen 20-entry subset output audit reproduces. Whole-source native
recovery, output audit and source registration remain pending. No source
installation, remote execution or database build occurred.

### Native recovery: ninth comparison batch

Forty additional queries bring the snapshot to 340 searches and 363 printed
correspondences. Reviewed 40 primary headwords in six local 300-dpi sheets,
including bird vocabulary and 18 cardinal-number entries. Accepted native
readings total 363; 2,088 remain pending. All reviewed cardinal entries retain
`num` and printed word boundaries; the source's compound number expressions are
not replaced by Bengali control equivalents or inferred digit-based forms.

Four bird entries have API-only alternatives, retained solely as comparison
evidence. Each native decision retains publisher IDs, response hashes and PDF
crop coordinates. Regenerated review analysis, CSV and audit. Twelve focused
tests pass (0.40s); the frozen subset output audit reproduces. Whole-source native
recovery, output audit and source registration remain pending. No source
installation, remote execution or database build occurred.

### Native recovery: tenth comparison batch

Forty additional queries bring the snapshot to 380 searches and 403 printed
correspondences. Visually reviewed forty cardinal-number headwords in six local
300-dpi sheets. No API alternate fields were present in this batch. Accepted
native readings total 403; 2,048 remain pending, with 58 reviewed cardinal entries.

The separate Four / 4 and 3 / Three source entries preserve their own keys and
glosses even where their native forms agree. Printed multiword number expressions
and numeral tags remain intact. Each acceptance records crop coordinates,
publisher IDs and response hashes.

Regenerated review analysis, CSV and audit. Twelve focused tests pass (0.41s);
the frozen subset output audit reproduces. Whole-source native recovery, output
audit and source registration remain pending. No source installation, remote
execution or database build occurred.

### Native recovery: eleventh batch; printed numeral inconsistency

Forty queries bring the snapshot to 420 searches and 443 printed correspondences.
Reviewed forty native numeral headwords in six 300-dpi sheets. Accepted native
readings total 443; 2,008 remain pending. Ten Lakh has one online-only alternate,
retained as evidence. Printed Ten / 10 preserve separate spacing (বার তি / বারতি).

A source inconsistency was verified in full lexical blocks: p78 entries6–7
print identical Mahali spelling and IPA for Thirty-four and Thirty-nine, while
the Bengali/Hindi translations and English definitions distinguish 34 and 39.
`semantic-review.jsonl` records both as a typed source-gloss inconsistency.
Both forms and meanings remain as printed, with `uncertain` and an explanatory
note; no arithmetic reconstruction or silent correction. Native transcription
acceptance does not imply that the numeric meaning is resolved. Total uncertain
analyses now 21 (previous19 plus these2).

Regenerated source review artifacts. Twelve focused tests pass (0.43s), including
the duplicate-numeral preservation check; the frozen subset audit reproduces.
Whole-source recovery, output audit and registration remain pending. No source
installation, remote execution or database build occurred.

### Native recovery: twelfth batch; cardinal section reviewed

Reviewed 31 additional headwords against five local 300-dpi sheets, bringing
accepted native readings to 474, with 1,977 pending. The cached publisher
comparison now contains 451 queries and 474 printed correspondences. Online-only
alternates for Twenty-five and One hundred remain evidence only. Source spacing,
ZWNJ shaping and separate digit/word entries are preserved.

All 129 cardinal-number native readings are now reviewed. `inventory_numerals.py`
and `numeral-inventory.json` identify four exact-IPA repeated groups (125 distinct
values). `numeral-review.json` classifies three as equivalent numeric labels and
one as the already flagged Thirty-four / Thirty-nine source conflict. This is an
exact-IPA repetition check, not exhaustive semantic or arithmetic validation.
Both conflicting records remain uncertain and uncorrected.

Regenerated only local review analysis, CSV and audit files. Twelve focused tests
pass (0.41s), and the frozen 20-record subset audit reproduces. Whole-source native
recovery, output audit and registration remain pending. No remote execution,
database build, source installation or publication occurred.

### Native recovery: thirteenth batch; causatives and celestial terms

Forty public publisher searches, performed locally, bring the cache to 491 queries
and 514 printed correspondences. Reviewed forty headwords in six 300-dpi sheets:
all 21 Causative Verb entries, all 15 Celestial Bodies and Related entries, and
the first four Classifier-Numeral-Determiner entries. Accepted native readings
now total 514; 1,937 remain pending. No online-only alternate fields occurred.

Preserved full causative expressions, including printed চুএম versus চু এম spacing,
and Bengali digits in 20 Kilogram / 250 Gram. Publisher shaping ZWNJ and raw outer
or repeated spaces remain in evidence; accepted values use NFC and single spaces.
No morphological decomposition, borrowing inference or coinage attribution was
added. Spelling acceptance does not independently establish lexical semantics.

Regenerated local review analysis, draft CSV and audit only. Twelve focused tests
pass (0.41s); the frozen twenty-record output subset reproduces. The two printed
numeral conflicts remain flagged among 21 uncertain analyses. Whole-source native
recovery, output audit and registration remain pending; database and browser
build gates remain explicitly deferred by the user. No remote execution,
database build, source installation or publication occurred.

### Native recovery: fourteenth batch; classifier phrases and sentences

Forty public publisher searches bring the local cache to 531 queries and 555
printed correspondences. Forty-one native headwords were visually reviewed in
six 300-dpi sheets: forty consecutive classifier/quantity expressions and the
separate Girl entry at p350:e2 returned by the same public query. The p91 Girl
and p350 Girl remain distinct records with distinct domains and forms. Accepted
native readings now total 555; 1,896 remain pending.

The renderer initially stopped at p89:e4 because the multiline grammar label's
union bounding box extended to the left margin. Full-page inspection confirmed
the headword itself is on one line. The crop now anchors to the grammar label's
opening-parenthesis character, retaining the guard for an actual wrapped headword.
The rerun rendered all 41 candidates and every sheet was visually reviewed.

Full sentence expressions and printed আইমা / আয়মা variants are retained, without
inferred component records or added graph relations. The original IPA and native
spellings remain independent transcriptions; matching the publisher is not an
assertion that all printed spelling/IPA correspondences are internally consistent.

Regenerated only local review analysis, CSV and audit. Twelve focused tests pass
(0.50s); the frozen twenty-record subset audit reproduces. Whole-source recovery,
output audit and registration remain pending. No remote execution, database build,
source installation or publication occurred; full-build/browser gates remain
explicitly deferred by the user.

### Native recovery: fifteenth batch; determiner and quantity expressions

Forty public searches bring the locally cached publisher comparison to 571 queries
and 596 printed correspondences. Visually reviewed 41 headwords in six 300-dpi
sheets: forty consecutive records from p92:e6 through p97:e3, plus Tree at p376:e2.
Accepted native readings now total 596; 1,855 remain pending.

The classifier Tree and the Trees-section Tree retain separate source keys even
though both print দারে. The latter publisher record additionally offers ডারে;
this is retained only in API evidence because it is absent from the printed
headword. Each girl and All books preserve publisher shaping ZWNJ. Full determiner,
quantity and sentence expressions remain intact; no inferred components,
paradigm links, number emendations or donor relationships were introduced.

Regenerated only the local review analysis, CSV and audit. Twelve focused tests
pass (0.45s); the frozen twenty-record subset audit reproduces. Whole-source
native recovery, output audit and registration remain pending. No remote
execution, database build, source installation or publication occurred. Full-build
and browser gates remain explicitly deferred by the user.

### Native recovery: sixteenth batch; quantities and fractions

Forty public searches bring the local cache to 611 queries and 636 printed
correspondences. Visually reviewed forty headwords in six 300-dpi sheets, from
p97:e4 through p101:e8 (p98:e7 was already reviewed). Accepted native readings
now total 636; 1,815 remain pending. No API alternate fields occurred in this batch.

Retained printed পিয়া / প্যায়া and বারতি / বার তি differences across expressions.
The native headword for 1/8th of the oil at p101:e7 visibly ends with a slash,
matching the publisher field; this literal source punctuation remains. Its IPA
question mark and existing uncertainty tag remain independent and unchanged.
No arithmetic reconstruction, lexical decomposition or graph relations added.

Regenerated local review analysis, CSV and audit only. Twelve focused tests pass
(0.51s), the frozen twenty-record output subset reproduces, and a focused assertion
confirms the fraction entry retains its slash, IPA question mark and uncertainty.
Whole-source native recovery, output audit and registration remain pending.
No remote execution, database build, source installation or publication occurred;
full-build/browser gates remain explicitly deferred by the user.

### Native recovery: seventeenth batch; source IPA/native mismatch

Forty public searches bring the local cache to 651 queries and 676 printed
correspondences. Reviewed forty native headwords from p102:e1 through p106:e4
in six 300-dpi sheets. Accepted native readings total 676; 1,775 remain pending.
All printed expressions and separate occurrences remain intact.

Full-block inspection at 180dpi confirmed a source inconsistency in A few days
(p102:e4): native মিঃ বার দিন accompanies IPA /ɔlpɔ kɔjekʈi d̪in/, corresponding
to the adjacent Bengali translation rather than the Mahali headword. The publisher
repeats the same pairing. `semantic-review.jsonl` records the source field mismatch;
both fields are retained without inferred replacement IPA, and the entry is
uncertain. Native acceptance does not resolve its pronunciation. There are now
22 uncertain analyses, including three explicit source inconsistencies.

Regenerated local review analysis, CSV and audit. Thirteen focused tests pass
(0.41s), including preservation and flagging of this mismatch; the frozen
20-record subset audit reproduces. Whole-source native recovery, output audit
and registration remain pending. No remote execution, database build, source
installation or publication occurred. Build/browser gates remain user-deferred.

### Native recovery: eighteenth batch; classifier section finished

Forty public searches bring the local cache to 691 queries and 718 printed
correspondences. Reviewed 42 native headwords in six 300-dpi sheets. Accepted
native readings now total 718; 1,733 remain pending. All 196 entries in the
Classifier-Numeral-Determiner section now have reviewed native readings; the
previously flagged A few days IPA/native mismatch remains unresolved.

The batch also starts Climbers and Creepers, with two additional same-label
records elsewhere in the book. Watermelon preserves তোরমুজ / তুরমুজ in its two
sections; Bean preserves বরবোটি / লাফাই. Online alternate fields for Pumpkin and
both Bean records stay in the evidence only, as absent from the printed headwords.
No botanical identity, borrowing or graph relationship was inferred from labels.

Regenerated local review analysis, CSV and audit. Thirteen focused tests pass
(0.43s); the frozen 20-record output subset reproduces. Whole-source native
recovery, output audit and registration remain pending. No remote execution,
database build, source installation or publication occurred. Build/browser gates
remain explicitly deferred by the user.

### Native recovery: nineteenth batch; colours and compound verbs

Forty public searches bring the local publisher cache to 731 queries and 759
printed correspondences. Visually reviewed 41 headwords in six 300-dpi sheets.
Accepted native readings total 759; 1,692 remain pending. Native review now covers
all 10 Climbers and Creepers, 13
Colour Terms, and 25 Compound Verb entries.

Online-only alternates for Grey, Blue and Brown remain evidence only. Orange as
a colour (কম্‌লা) and Orange as fruit (কুমলা লেবু) keep distinct forms, domains and
source keys. Compound verbs retain their complete expressions and original
parenthetical gloss qualifiers without inferred component forms or new graph
relations. Publisher ZWNJ is retained as script shaping.

Regenerated only local review analysis, CSV and audit. Thirteen focused tests pass
(0.48s); the frozen twenty-record subset audit reproduces. Whole-source native
recovery, output audit and registration remain pending. No remote execution,
database build, source installation or publication occurred; build/browser gates
remain explicitly user-deferred.

### Native recovery: twentieth batch; cultural terms and directions

Forty public searches bring the local publisher cache to 771 queries and 805
printed correspondences. Reviewed 46 native headwords in seven 300-dpi sheets.
Accepted native readings total 805; 1,646 remain pending. Native review now covers
all 17 Cultural function and 16 Direction entries.
Online-only alternates for Movie, Culture, Song, Desert, Island and seasonal Fall
remain evidence only. Same-label records in other domains remain distinct.

Full printed lexical blocks for Fall were inspected at 180dpi in fall-senses.png:
p122:e5 is falling water, p214:e2 is the verb, p362:e6 is the leaf-fall season.
Analysis glosses now clarify Fall (waterfall) and Fall (autumn), retaining verb
Fall and all three raw source labels. Notes give concise source-grounded reasons;
no copied prose definitions or inferred relations were added.

Regenerated local analysis, CSV and audit. Thirteen focused tests pass (0.45s),
the frozen twenty-record subset audit reproduces, and focused assertions confirm
the three meanings preserve their original source labels. Whole-source native
recovery, output audit and registration remain pending. No remote execution,
database build, source installation or publication occurred; build/browser gates
remain explicitly user-deferred.

### Native recovery: twenty-first batch; landscape and education

Forty public searches bring the local publisher cache to 811 queries and 847
printed correspondences. Reviewed 42 headwords in six 300-dpi sheets. Accepted
native readings now total 847; 1,604 remain pending. Native review now covers all
28 Earth and Related entries, and proceeds into Education through Inkpot.

Preserved multiword descriptions for landscape and educational concepts without
inferring borrowing or coinage status. Online alternatives for Earth, Den,
Mountain, Language, Linguistics, School uniform, Magazine, Fountain Pen, Inkpot
and sports Ground remain evidence only. Stone and Ground occurrences in other
domains remain separate records, as do the identical printed Den/Cave forms.

Regenerated only local review analysis, CSV and audit. Thirteen focused tests pass
(0.55s); the frozen twenty-record subset audit reproduces. Whole-source recovery,
output audit and registration remain pending. No remote execution, database build,
source installation or publication occurred; build/browser gates remain explicitly
user-deferred.

### Native recovery: twenty-second batch; education and Bachelor senses

Forty public searches bring the local publisher cache to 851 queries and 889
printed correspondences. Visually reviewed 42 headwords in six 300-dpi sheets,
continuing Education from Scholarship through Sociology and additionally checking
Bachelor in Kinship and Teacher in Occupation. Accepted native readings total
889; 1,562 remain pending. Online alternatives stay in the comparison evidence.

Full-block inspection at 180dpi in bachelor-senses.png confirms the education
Bachelor denotes a degree holder and the kinship Bachelor an unmarried man.
Analysis glosses now explicitly distinguish these senses while preserving both
original Bachelor labels, forms and source keys. Teacher remains independently
attested in its two sections. No inferred relations or histories added.

Regenerated local review analysis, CSV and audit. Thirteen focused tests pass
(0.40s); the frozen twenty-record subset audit reproduces. Focused assertions
confirm both clarified Bachelor glosses retain their source labels. Whole-source
native recovery, output audit and registration remain pending. No remote execution,
database build, source installation or publication occurred; build/browser gates
remain explicitly user-deferred.


### Mahali native review, batch 23

Visually checked 41 printed headwords against cached publisher Unicode, completing
Education (78 records) and Festivals and Related (12). Finance has one native
reading still pending. Native recovery now totals 930 accepted and 1,521 pending
of 2,451 draft entries; 22 existing uncertainty flags remain. The publisher cache
contains 891 queries and 930 correspondence proposals. Evidence is recorded in
native-recovery-review.jsonl and native-batch-23 render sheets with PDF crop
coordinates, publisher record IDs and response hashes. Online alternative forms
remain audit evidence only; printed phrases, shaping characters and separate
Festival attestations are preserved without inferred equivalences or relations.

Regenerated local review analysis, CSV and audit. Thirteen focused tests pass
(0.45s); the frozen 20-record audit reproduces within its original 190-record
reviewed subset. This is not a whole-source audit. Native recovery, full output
review and source registration remain pending. No remote execution or database
build occurred; build and browser gates remain explicitly user-deferred.


### Mahali native review, batch 24

Visually confirmed 37 additional printed headwords against publisher Unicode.
Finance (23), Fire and Related (13), and Fish and Related (10) now have no pending
native readings. Recovery totals 967 accepted and 1,484 pending of 2,451 entries;
22 uncertainty flags remain. The bounded public publisher cache now contains 921
queries yielding 967 correspondence proposals. Batch 24 render sheets and crop
coordinates, publisher record IDs and response hashes are recorded in the native
review ledger. Distinct Camphor and Ember forms remain separate printed records;
repeated Fish and Prawn attestations retain their section provenance. Online-only
alternatives are audit evidence, not new entries. Source domain names do not
assert zoological classification or etymological relationships.

Regenerated review analysis, CSV and audit locally. All 13 focused tests passed
(0.46s); the frozen 20-record audit reproduces within its original 190-record
population, not a whole-source completion claim. Native recovery, full output
review and source registration remain pending. No remote execution, database
build or publication occurred. Required build/browser gates remain user-deferred.


### Mahali native review, batch 25

Visually checked 42 printed headwords and accepted 41 publisher-assisted readings,
bringing the ledger to 1,008 accepted and 1,443 pending of 2,451 draft records.
Flies and Insects (18) and Flowers and Related (18) now have all native readings
reviewed. There are 961 cached public queries and 1,009 correspondence proposals.
Render sheets, crop coordinates and API provenance are recorded per accepted row.
The renderer initially refused the one-letter Housefly/Fly headwords; full-page
inspection of page 149 verified their layout, and explicit exceptions for only
those two source keys now permit their short crops. Other wrapping guards remain.

Full-page review of page 153 confirms printed Belly denotes a flower. The draft
now uses Belly (flower), preserves source_gloss Belly and does not infer species.
The body-part Belly remains unchanged. Curry at p155:e4 prints উতু /ut̪u/ whereas
the online entry supplies অতঅ /ɔt̪ɔ/. This edition difference is documented in
publisher-edition-review.jsonl; the online form is not substituted. Manual-native
recovery for Curry remains pending. Online Mustard oil alternatives remain audit
evidence only. No graph relationships are inferred.

Regenerated local review analysis, CSV and audit. All 13 focused tests pass
(0.60s), including flower/body-part gloss separation; 22 prior uncertainty flags
remain. The frozen 20-record subset audit reproduces, not a whole-source audit.
Native recovery, full output review and source registration remain pending.
No remote execution or database build occurred; build/browser gates are deferred
under the user's instruction.


### Direct printed recovery: Curry

Rechecked the full printed page 155 and accepted উতু /ut̪u/ for Curry by direct
visual transcription. The online অতঅ /ɔt̪ɔ/ is documented as an edition difference,
not used as a replacement. The native ledger explicitly distinguishes this direct
print method from publisher-assisted recovery, records the PDF identity and crop
evidence, and links the resolved edition review. Analysis notes preserve the
edition distinction. Totals are 1,009 reviewed native readings and 1,442 pending.

The evidence check now validates direct printed readings against their explicit
edition review and PDF identity while retaining strict IPA/domain/API checks for
publisher-assisted readings. A focused regression verifies that the Curry output
keeps the printed spelling and IPA and records the online difference. All 14
focused tests pass (0.44s). The frozen 20-record subset audit reproduces; this
still does not establish a whole-source output audit. No database build, remote
execution or installation occurred. Remaining native recovery, source registration
and output review continue; build/browser gates remain user-deferred.


### Mahali native review, batch 26

Accepted 30 additional native readings after visual comparison of printed pages
156–161 with publisher Unicode. Recovery totals 1,039 reviewed and 1,412 pending:
1,038 publisher-assisted readings and one direct printed transcription. The public
cache contains 991 queries yielding 1,039 proposals. Batch 26 sheets, crop boxes,
API record IDs and response hashes provide per-entry provenance. Separate rice
preparations, rice dishes and drink labels remain independent printed entries.
Online alternatives for Lemon drink, Tea and Rice Wine remain audit evidence only;
no inferred synonym, borrowing or morphological relationships have been added.

Regenerated local review analysis, CSV and audit. All 14 focused tests pass
(0.46s); 22 source uncertainty flags remain. The frozen 20-record reviewed-subset
audit reproduces, not a whole-source completion audit. Native recovery, full output
review and source registration remain pending. No remote execution or database
build occurred; required build/browser gates remain explicitly user-deferred.


### Mahali native review, batch 27

Visually reviewed 34 headwords and accepted 33 publisher-assisted readings plus
one direct printed transcription. Totals are 1,073 reviewed and 1,378 pending,
including 1,071 publisher-assisted and two direct printed readings. The public
cache now has 1,021 queries and 1,073 proposals. Per-entry crop and API provenance
is recorded in the ledger. Separate Gram and Lime attestations retain their
source domains; no equivalence or graph relationship is inferred from labels.

Full-page inspection of p164 confirms Beverage prints নুআঃ /nuaʔ/, whereas the
online record 9167 has ঞুআঃ /ɲuaʔ/. Accepted the printed spelling directly and
recorded the edition difference in publisher-edition-review.jsonl and analysis
notes. A focused regression guards against substitution of the online palatal
nasal. No source form or uncertainty mark was silently normalized away.

Regenerated local review analysis, CSV and audit. All 15 focused tests pass
(0.49s); 22 source uncertainty flags remain. The frozen 20-record subset audit
reproduces, not a whole-source completion audit. Native recovery, full output
review and source registration remain pending. No remote execution or database
build occurred; required build/browser gates remain explicitly user-deferred.


### Mahali native review, batch 28

Accepted 30 additional publisher-assisted native readings after inspecting printed
headwords on pages 167–172. Totals are 1,103 reviewed and 1,348 pending: 1,101
publisher-assisted readings and two direct printed transcriptions. The cache has
1,051 public queries and 1,103 proposals. Batch 28 render sheets and per-entry
crop/API provenance are recorded in the ledger. Gravy and Soup retain separate
printed attestations despite identical forms. Online alternatives for Raw
Vegetables, Egg, Flour, Oat, Gravy and Clove remain audit evidence only; no
unprinted variants or inferred graph relations were introduced.

Regenerated local review analysis, CSV and audit. All 15 focused tests pass
(0.48s); 22 existing uncertainty flags remain. The frozen 20-record subset audit
reproduces; whole-source native recovery, output audit and registration remain
pending. No remote execution or database build occurred. Required build/browser
gates remain user-deferred.


### Mahali native review, batch 29

Visually checked 23 headwords and accepted 22 publisher-assisted native readings.
Totals: 1,125 reviewed, 1,326 pending; 1,123 publisher-assisted readings and two
direct printed transcriptions. The cache has 1,071 queries and 1,126 proposals.
Evidence is recorded in batch 29 sheets and the per-entry native ledger. Online
alternatives remain audit-only. Food-section Taste (p173:e2, printed স্যাবেল)
has no matching publisher candidate and remains pending direct printed review;
the General-section Taste (p198:e6, চাখা) is independently accepted. Do not substitute
one for the other. Fruit and tree attestations retain separate provenance.

Regenerated local analysis, review CSV and audit. All 15 focused tests pass
(0.47s); the frozen 20-record subset audit reproduces. Existing 22 uncertainty
flags remain. Whole-source recovery, output review and source registration are
unfinished. No remote execution or database build occurred; build/browser gates
remain user-deferred.

### Direct printed recovery: Taste noun

Full-page inspection of p173 confirms the noun স্যাবেল /sæbel/. The publisher
search returns General verb চাখা /tʃakʰa/, with স্যাবেল as a secondary string;
that string is not evidence that the verb IPA or grammar applies to the noun.
Accepted the noun directly from print with PDF/crop evidence, preserved its noun
tag and recorded the search discrepancy. The separate verb retains its own form
and tag. No graph equivalence is inferred. Totals: 1,126 reviewed native readings,
1,325 pending; three readings now use direct printed transcription.

Regenerated local review analysis, CSV and audit. All 16 focused tests pass
(0.50s), including noun/verb separation for Taste. The frozen subset output audit
reproduces; full source review and registration remain pending. No remote
execution or database build occurred; build/browser gates remain user-deferred.


### Mahali native review, batch 30

Accepted 32 publisher-assisted native readings following visual inspection of
batch 30 printed headwords. Totals: 1,158 reviewed and 1,293 pending; 1,155
publisher-assisted readings and three direct printed transcriptions. The cache
contains 1,101 queries yielding 1,158 proposals. All three Palm attestations retain
separate forms and source domains. General-section phrases retain their complete
printed forms; the online Scribe alternative remains audit evidence only. No
unprinted variants or inferred morphological/etymological relations were added.

Regenerated local review analysis, CSV and audit. All 16 focused tests pass
(0.52s); 22 existing source uncertainty flags remain. The frozen 20-record subset
audit reproduces, not a whole-source audit. Remaining native recovery, output
review and registration continue. No remote execution or database build occurred;
required build/browser gates remain user-deferred.


### Mahali native review, batch 31

Accepted 31 publisher-assisted readings after visual inspection of headwords on
pages 181–185. Totals: 1,189 reviewed and 1,262 pending; 1,186 publisher-assisted
readings and three direct printed transcriptions. The public cache contains 1,131
queries yielding 1,189 proposals. Per-entry crop and API evidence is recorded in
the ledger. Distinct Listen/Hear spellings and separate Severe attestations remain
as printed. Online alternatives are evidence only; no synonyms or graph relations
are inferred. Full multiword headwords and script-shaping characters are retained.

Regenerated local analysis, review CSV and audit. All 16 focused tests pass
(0.52s); 22 existing uncertainty flags remain. The frozen 20-record subset audit
reproduces, not a whole-source completion audit. Native recovery, full output
review and source registration remain pending. No remote execution or database
build occurred; required build/browser gates remain user-deferred.


### Mahali native review, batch 32

Accepted 32 visually checked native readings; totals are 1,221 reviewed and 1,230
pending (1,218 publisher-assisted and three direct printed transcriptions). Public
cache: 1,161 queries and 1,221 proposals. Per-entry evidence is in batch 32 sheets
and the native ledger. Fast occurs as adjective (p188:e4) and adverb (p188:e5).
Both online candidates have the same native/IPA; cached grammatical categories
select IDs 12131 and 12132 respectively. Preserve the two printed records and
parts of speech. Cook also retains separate verb and occupation attestations.
Online alternatives are audit evidence only; no graph links are inferred.

Regenerated local analysis, review CSV and audit. All 16 focused tests pass
(0.54s); 22 existing uncertainty flags remain. The frozen 20-record subset audit
reproduces, not a whole-source audit. Native recovery, full output review and
source registration remain unfinished. No remote execution or database build
occurred; required build/browser gates remain user-deferred.


### Mahali native review, batch 33

Accepted 31 publisher-assisted readings after visual inspection of batch 33
headwords. Totals: 1,252 reviewed and 1,199 pending; 1,249 publisher-assisted and
three direct printed transcriptions. Cache: 1,191 queries, 1,252 proposals. The
ledger records per-entry print crops and publisher provenance. Separate Cough
attestations remain distinct, and online alternatives remain audit evidence only.
Printed phrases, nasalization and script-shaping characters are preserved; no
unprinted variants or inferred graph relations were added.

Regenerated local analysis, review CSV and audit. All 16 focused tests pass
(0.63s); 22 existing uncertainty flags remain. The frozen 20-record subset audit
reproduces. Passing halfway in native recovery is not whole-source completion:
remaining native review, full output audit and registration are unfinished. No
remote execution or database build occurred; build/browser gates remain deferred
under the user's instruction.


### Mahali native review, batch 34

Accepted 30 publisher-assisted readings after visual inspection of batch 34
headwords (pages 196–201). Totals: 1,282 reviewed, 1,169 pending; 1,279 publisher-
assisted and three direct printed transcriptions. Public cache: 1,221 queries,
1,282 proposals. Per-entry crop/API provenance is recorded in the ledger. Drive
and Operate remain separate printed records, as do Scream and Shout; no relation
is inferred from identical forms. Online alternatives remain audit evidence only.

Regenerated local analysis, review CSV and audit. All 16 focused tests pass
(0.49s); 22 existing uncertainty flags remain. The frozen 20-record subset audit
reproduces. Whole-source native recovery, output audit and registration remain
unfinished. No remote execution or database build occurred; required build/browser
gates remain explicitly user-deferred.


### Mahali native review, batch 35

Accepted 30 publisher-assisted readings after visually checking printed headwords
on pages 201–206. Totals: 1,312 reviewed and 1,139 pending; 1,309 publisher-assisted
and three direct printed transcriptions. Public cache: 1,251 queries and 1,312
proposals. Crop coordinates, rendered-sheet locations and API provenance are
recorded per entry. Full phrases remain intact; online alternatives for Add,
Unite, Book (a ticket), Lean, Confirm, Jolt and Pretend remain audit evidence only.
No inferred morphological, synonym or etymological links were added.

Regenerated local analysis, review CSV and audit. All 16 focused tests pass
(0.46s); 22 existing uncertainty flags remain. The frozen 20-record subset audit
reproduces, not a whole-source completion audit. Native recovery, full output
review and source registration remain pending. No remote execution or database
build occurred; required build/browser gates remain user-deferred.


### Mahali native review, batch 36

Accepted 30 publisher-assisted native readings after inspecting print crops from
pages 206–210. Totals: 1,342 reviewed and 1,109 pending; 1,339 publisher-assisted
and three direct printed transcriptions. Cache: 1,281 queries, 1,342 proposals.
Per-entry evidence is recorded in the native ledger and batch 36 sheets. Jump
and Bounce retain separate printed records. Online alternatives for Lame, Leap,
Travel and Practise remain audit evidence only. Complete phrases and shaping
characters remain intact; no inferred synonym or morphological links were added.

Regenerated local analysis, review CSV and audit. All 16 focused tests pass
(0.46s); 22 existing uncertainty flags remain. The frozen 20-record subset audit
reproduces, not a whole-source completion audit. Remaining native review, full
output audit and source registration are unfinished. No remote execution or
database build occurred; required build/browser gates remain user-deferred.


### Mahali batch 37 coordination

The Mahali agent visually checked 30 additional printed headwords on pages 210–215 and updated its source-local ledger to 1,372 accepted / 1,079 pending. The source-local analysis, draft, and frozen subset audit were regenerated; 16 focused tests passed (0.54s), with 22 prior uncertainty flags retained. No installed rows or database build. Full output audit, registration, build, and browser gates remain pending under the user’s constraints.


### Further diversity sources, 25 September 2026

The checklist's dictionary/glossary, online-source, PDF and OCR-specific gates
remain active as applicable. The Gaddi agent installed 254 attestations from all
221 prompts in Kumari et al. (2026), Table B.1, printed pp. 119–124. Its
source-local audit records five printed anomalies and a fresh 0/20 visual sample;
seven source tests pass and all 254 rows parse/convert. The source adds Gaddi
with precise printed page/item provenance, bibliography and a source-specific
sound profile; append order 39.

The Gtaʔ agent installed 733 conservative lowercase lexical rows from a pinned
Donegan–Stampe Munda Lexical Archive Chatterji subset. Its audit accounts for
2,063 numbered source records and two malformed unnumbered fragments. A fresh
0/20 raw-to-output review, four focused tests, and a 733/733 parse/conversion
pass; the existing DSGT bibliography entry was updated with archive provenance
and reuse conditions. Append order 40. Uppercase and unresolved source strings
remain explicitly accounted for in the source-local audit.

The Koda online-dictionary first domain was snapshotted through 60 bounded
publisher API search probes. Sixty Adornments and Costumes records were returned;
47 clear-IPA records were installed, 13 ambiguous/incomplete-IPA records withheld, and two
unrelated General-domain hits excluded. Publisher record/concept IDs, response
hashes, 13 secondary native spellings and all decisions remain in the source-local audit. A seeded 20-record
print comparison confirmed labels, native spelling and grammar against the
publisher's 2022 printed edition, while treating online IPA as a separate
dated source reading. Four focused tests pass; 47/47 rows parse/convert;
append order 41. Publisher prose and images were not republished.

The Mahali agent meanwhile reached 1,542 print-verified native readings and
909 pending after batch 42, with 18 focused tests passing. Its source remains
in preparation; no rows are installed from it yet. Source metadata validates
for 206 source files and 201 citation keys. The combined focused source suite
for the twelve recently installed diversity packages passes, 125 tests in
26.48 seconds. The full CLDF build, full suite, compiled identity/graph/
reference checks, browser database and app QA remain deferred under the user's
explicit no-build instruction. No cluster/remote execution occurred.

### Further source-side verification, 25 September 2026

Grierson's *Linguistic Survey of India* XI supplies two separately scoped
West Indo-Aryan tables. Kanjari's Sitapur and Belgaum columns yielded 163
installed rows from 164 audited cells: three printed blanks, one repeated
answer collapsed, and two independent extra answers. The Sikalgari Belgaum
column yielded 82 installed rows from 82 audited cells. Each package preserves
prompt/page/column provenance and a source-local visual audit; both cite the
same volume under `grierson1922lsi11`. Their dialect registrations use blank
coordinates rather than inferred village points. Append orders are 43 and 45.

Bailey's 1908 Sainji glossary and cardinals on printed pp. 55–56 yielded 55
installed rows from 52 accepted of 66 audited source units. Thirteen uncertain
typographic readings and one incomplete suffix were withheld. A full 66-unit
typography review checked underlining, italic vowels, underdots and breves;
grammatical tags appear only where the printed numeral section licenses them.
Its source package includes the scan hash, per-unit audit and source-specific
profile. Append order is 42.

The six new source packages (Gaddi, Gtaʔ, Koda, Kanjari, Sikalgari, Sainji)
pass 31 focused tests together (2.54 seconds). Source metadata validation
passes for 209 settings files and 203 citation keys. These are source-side
checks only: the user has reserved the full data/database build, compiled
reference/identity/graph validation and browser QA for a later direct request.
No ASJP input or remote execution was used.

### Sansi source-side installation

The ordinary Sasi column in Grierson's 1922 LSI XI, printed pp. 178, 182,
186 and 190, maps to the existing Sansi language. Eighty-two scoped prompt
cells yield 87 source rows because five cells print two independent answers.
The separate criminal-argot column was not imported. One historically named
Sansi dialect row has blank coordinates, and the shared
`grierson1922lsi11` bibliography record now covers Kanjari, Sikalgari and
Sansi without creating three duplicate volume citations. The source-local
audit, 20-cell print sample, importer and profile are installed with append
order 46. The three Grierson packages pass 18 focused tests together;
the agent's expanded focused and dialect suite passes 24. All 87 Sansi rows
parse and convert, with no Sansi sound-profile-policy finding. Existing six
compiled Sansi attestations have no exact form-and-gloss overlap with this
bounded source. Full compiled and browser gates remain user-deferred.

### Suketi source-side installation

The Suketi column in Grierson's 1916 LSI IX(IV) Mandi-group table, printed
pp. 759–761, contributes 55 rows from 53 accepted of 61 audited prompt cells
(1–13, 32–79). Five typographically uncertain cells, two printed blanks and
one complex oblique-sister cell were withheld; two accepted cells printed two
independent answers. The complete cell audit and a seeded 20-cell visual review
are preserved. The original-resolution public-domain PDF is pinned by URL and
SHA256 and remains an ignored local cache, rather than a committed 231 MB
binary. The registered base Suketi is reused because this table gives no finer
field site. `grierson1916suketi` distinguishes the primary printed table from
the already included LSI comparative vocabulary so shared survey material is
not counted as independent field evidence. Append order is 47. Eleven focused
Suketi/dialect checks and all 55 parse/conversions pass, with no scoped
sound-profile-policy finding. Across the eight new sources, 41 focused source
tests pass together (1.63 seconds); source metadata validates 211 settings
files and 204 citation keys. The full data build and downstream gates remain
deferred by the user's direct instruction.

A separate bounded scan of all eight newly installed CSVs found 1,476
15-column rows, with nonempty and globally distinct source entry keys.

### Mahali print-review checkpoint

The CFEL Mahali 2024 dictionary's complete 2,451-record native ledger has
now been visually reviewed against the printed edition: 2,443 readings are
publisher-assisted and eight are direct print transcriptions, with zero native
readings pending. Twenty source-preparation tests pass independently under
root. This is still a preparation checkpoint: the agent is completing the
whole-source output/provenance audit and canonical source registration. No
Mahali CSV has yet been installed or compiled, and the user's no-build rule
still applies.

### Kaikadi source-side installation

The Kaikadi (Sholapur) column of Grierson's public-domain LSI IV (1906),
printed pp. 650 and 654, adds the first Kaikadi source rows in Jambu. The
bounded prompts 32–79 contain 48 cells: 40 installed, two printed blanks and
six held because the 992×1404 DjVu scan does not resolve letters/diacritics
securely. One canonical `Kaikadi` row maps to `kaik1244` in `S. Dravidian I`,
with blank coordinates, and a historical Sholapur dialect is registered
without claiming an exact speaker site. The two target-column crops, OCR
comparisons, per-cell audit and seeded 20-cell 0-error review are in the source
package; the 701-page public-domain scan is a local cache. Bibliography key
`grierson1906lsi4`, append order 48. All 40 rows parse/convert; six focused
source tests pass under root, and the agent's expanded source/dialect suite
passes 30. Source metadata validates 212 settings files / 205 citation keys.
The nine-profile scoped sound-policy check printed no findings for any newly
installed profile; its CLI still returned failure because its summary counts
eight findings in `sil-gutob-gorum` and one in `sil-pahari-pothwari` even
with a filter. These pre-existing profiles were not rewritten during this
source-side check.
Full build and downstream gates remain user-deferred.

### Mahali and Surkhuli source-side installation

CFEL's 2024 Mahali–Bangla–Hindi–English dictionary now contributes 2,451
source rows, one for every printed record. All 2,451 Native readings were
visually checked: 2,443 use publisher-assisted Unicode proposals and eight
use direct print transcription. Twenty-two typed uncertainties remain in the
audit. A fresh seeded 20-record print-to-source-output review found zero
material errors, and all 2,451 rows parse/convert. Mahali uses existing
canonical language metadata, no invented dialect/site coordinate, no inferred
graph edges, bibliography key `pradhan-tripathi2024mahali`, and reserved append
order 44. Twenty-one focused tests pass. The publisher asserts copyright and
offers no open licence; the package records Jambu's extracted-lexical-facts
editorial basis and keeps the PDF, page images, prose definitions and full API
responses outside installed assets. This is not a permission or public-domain
claim, and reuse should be rechecked before any public release.

Bailey's 1920 standalone Surkhuli vocabulary is bounded to printed p. 155,
above–give: 69 inventoried source units, 26 securely read, 43 withheld for
unresolved breve/macron, underlining, dotted-consonant or complex-letter
typography. Three accepted units print two independent answers, giving 29
installed rows. A seeded 20-item visual review covers 10 accepted matches and
10 documented holds; all 29 rows parse and convert. Its source package,
profile and `bailey1920surkhuli` bibliography entry use append order 49.
Five source tests pass under root after bibliography registration.

At this checkpoint the eleven newly installed packages pass 74 focused tests
together (1.86 seconds). The full data build and compiled/browser gates remain
deferred by the user's instruction; no remote execution or database build
occurred.

### Yerukula/Korvi and Bāghī source-side installation

Grierson's 1906 LSI IV Korvi (Belgaum) column is mapped to canonical
Yerukula (`yeru1240`), while the historical `Korvi (Belgaum)` label remains a
registered dialect/alias. The bounded printed prompts 32–79 contain 48 cells:
39 installed rows and nine held because the available image does not resolve
their typography. The volume's single `grierson1906lsi4` bibliography record
now covers both Kaikadi and Korvi without treating overlapping Lexibank
digitization as an independent field survey. Append order 50. All 39 rows
parse/convert; Korvi and Kaikadi source/dialect focused checks pass together.

Bailey's 1920 Bāghī Kōci vocabulary is bounded to printed p. 144 and mapped
to existing `ba` after checking the source's p. 113 locality. Of 59 audited
source units, 18 clear cells yield 18 installed rows; 32 typography-uncertain
cells, six mixed-use cells, two cross-references and one cell with no distinct
Bāghī answer remain audit-only. The adjacent Rampur control is excluded.
Its source-local audit, seeded 20-cell review, importer, profile and scoped
`bailey1920baghi` bibliography record use append order 51. All 18 rows
parse/convert.

The thirteen new source packages pass 85 focused tests together (2.73
seconds), and a bounded scan found 4,053 15-column rows with globally
distinct nonempty source entry keys. Source metadata validates 216 settings
files and 208 citation keys. The global profile-policy command still reports
eight existing `sil-gutob-gorum` rules and one `sil-pahari-pothwari` rule;
the newly added profiles have no scoped findings. The user-deferred full data
build, compiled reference/identity/graph checks, full suite and browser QA
have not been run, so the ingestions are not declared complete under the
checklist.

### Gorum MLA source-side installation

The 2020-10-22 Wayback snapshot of Donegan and Stampe's Gorum Munda Lexical
Archive file is pinned by URL and SHA-256 in the source-local manifest. The
dictionary/glossary and website-snapshot addenda apply. Its explicit licence
permits derivative redistribution only with the same conditions; the source
package's `LICENSE` applies those conditions to its derived outputs. A
secondary PDF mirror was used only to confirm boundaries and layout and is
not installed. All 5,824 numbered archive chunks are inventoried: 559
conservative single Z-marked headwords are installed and 5,265 excluded with
typed reasons (3,523 not single unqualified Z-marked heads, 162 repeated IDs,
1,531 transcription/segmentation review holds, seven gloss holds and 42
grammar-label holds). Ninety-five installed rows carry editorial uncertainty.
The exact archive-to-CSV seeded review found 0/20 material errors. Source
Romanization is retained as spelling without asserting verified IPA, and no
unprinted graph or dialect relationships are inferred. The canonical Gorum
language is reused; append order 52, bibliography key `DSGO`. Four focused
tests, all 559 parse/profile checks and source metadata pass; the new sound
profile has no scoped policy findings. The earlier SIL Gorum source has 206
rows but only three exact source-form intersections; these attestations remain
distinct pending the compiled identity review.

### Barari and Holiya source-side installation

Bailey 1920 printed p. 185 adds the standalone North Jubbal or Barari
vocabulary page (above–give). The 71 inventoried units comprise 28 securely
read cells yielding 32 installed rows, 36 typography holds, six mixed-use
holds and one cross-reference. Exact comparison with the existing Barari
registry and Zoller 2023 shows that the source lect belongs under canonical
`Barari`, an intentional existing split from `jub` North Jubbali. Zoller
quotes Bailey's p. 185 for bad and daughter/girl, while all four corresponding
printed cells are held in this import; there are no exact-form overlaps with
the 32 installed Bailey rows. The audit records this as shared print evidence,
not independent elicitation. The page transcription, 20-cell image review,
audit, importer and literal preservation profile use append order 53 and
`bailey1920northjubbal`. All 32 rows parse/profile; five source tests pass.
The source-level Barari mapping was corrected after the initial package
draft; no database was built from that draft.

K. M. Metry's 2017 SPPEL Holiya numeral table, snapshotted from Chan's
Numeral Systems site, contributes 39 rows under new canonical `Holiya`
(`holi1239`) with broad Madhya Pradesh provenance and no invented site
coordinate or dialect. The survey-wordlist and website-snapshot addenda
apply. Forty displayed cells contain 42 explicitly numbered items; three
visibly malformed transcriptions (23, 60 and 90) are retained as typed
review holds in the complete per-item audit. The source IPA is preserved in
`Phonemic`; its equality to the raw `Form` is deliberate before the explicit
house-conversion profile. No component, arithmetic or cognacy relationships
are inferred from the numeral values. The source package pins its HTML
snapshot and records the archived dataset's CC-BY-4.0 release. Append order
54, key `metry2017holiya`. Five focused source tests, all 39 parse/profile
checks, source metadata and the new profile policy check pass.

At this checkpoint, 16 dated source CSVs contain 4,683 15-column rows with
globally distinct nonempty `Entry_Key` values. Their combined source-specific
suite passes 99 tests. A broader focused run including dialect and sound
profile tests passed 118 tests and exposed one existing global-policy failure:
the earlier `sil-gutob-gorum` profile has eight output mappings that disagree
with the CDIAL house-policy checker. Its own source regression tests explicitly
expect those outputs; it was not changed without a separate linguistic review.
The new 2026-09-25 profiles have no scoped policy findings. Full build,
compiled reference/identity/graph checks, full-suite tests and browser QA
remain user-deferred, so these source-stage installs are not declared
complete under the checklist.

### Inner Siraji and Aranadan source-side installation

Bailey 1908 printed p. 49 left column, mare–body, contains 32 inventoried
Inner Siraji lexical units. Thirteen secure cells yield 15 installed rows;
14 typography-uncertain cells, four suffix-only cells and one questioned
gloss remain in the complete audit. All 32 cells were visually reviewed and
a seeded 20-cell image comparison was retained. The original public-domain
scan is SHA-256 pinned. Canonical `insir` is reused, with no exact overlap
against its ten previously compiled forms; two existing LSI ear readings
depend on Bailey and are not treated as independent elicitation. The source
package has a reproducible importer, literal profile, key
`bailey1908innersiraji`, append order 55, and four passing focused tests.

The 2017 SPPEL Aranadan numeral page, recorded by Rose Mary A., Swaraj
Prabha and Sam Robert, is locally snapshotted with the primary HTML and a
seeded 20-item audit. Forty displayed cells yield 42 explicitly numbered
items; 41 are installed and numeral 60 remains a typed source-typography
hold. The exact Glottolog mapping creates canonical Aranadan (`aran1261`);
no site/dialect coordinate or numeral-component graph links are inferred.
The source IPA is preserved with an explicit profile and its archived
Numeralbank representation is CC-BY-4.0. Append order 56, key
`rosemary2017aranadan`. Five focused source tests, all 41 parse/profile
checks, source metadata and the new profile policy check pass.

At this checkpoint, 18 dated source CSVs contain 4,739 15-column rows with
globally distinct nonempty entry keys and no empty forms, languages or
citations. All cited keys resolve to BibTeX entries. The combined
source-specific suite passes 108 tests. All 18 newly routed profiles have no
scoped findings; the two known global profile-policy findings remain in
earlier profiles. Full build, compiled reference/identity/graph checks,
full-suite tests and browser QA remain user-deferred, so these packages are
still source-stage installations rather than completed ingestions.

### Outer Siraji source-side installation

Bailey 1908 printed p. 42 left column, wheat–little, has 30 separately
inventoried Outer Siraji cells. Nineteen securely transcribed forms are
installed; 11 typography-uncertain cells remain in the complete audit. All
cells were compared to the source image and a seeded 20-cell review is
retained. The historical lect maps to existing canonical `OuterSiraji`,
which had three Zoller attestations and no exact form overlap with these
installed rows. The public-domain scan hash is pinned; the importer, literal
profile, YAML and `bailey1908outersiraji` Bib record use append order 59.
Four focused source tests and metadata validation pass. The compiled build,
graph/identity/reference review and browser gates remain user-deferred.

Mullu Kurumba's 2018 numeral table, supplied by Melwin Jeba and preserved
as a pinned primary HTML snapshot, has 40 displayed cells with two paired
labels, yielding 42 explicit numbered items and 42 installed forms. Its
seeded 20-item raw comparison found zero material errors. The unusual
internal word space in numeral 26 is preserved. Canonical `Mullu Kurumba`
(`mull1244`) is separate from Jambu's Attappady Kurumba; the broad Kerala,
Wayanad and Nilgiri provenance does not justify a precise site point or
dialect row. The source IPA remains in `Phonemic` and has an explicit display
conversion profile; no numeric-component or cognacy edges are inferred.
Its archived Numeralbank representation is CC-BY-4.0. Append order 58,
key `jeba2018mullu`. Five source tests pass, all 42 rows parse/profile,
source metadata validates and the profile-policy audit is clean. With
Outer Siraji, the combined dated packages now number 20 and contain 4,800
source rows; compiled/full-suite/browser gates remain deferred.

### Kotguru source-side installation

Bailey 1908 printed p. 31 left column, face–star, supplies 42 separately
inventoried Kotguru lexical cells. Twenty-two securely read cells yield 23
installed rows because the hill item has two independent answers; 20
typography-uncertain cells remain audit-only. The public-domain scan is hash
pinned, every in-scope cell was compared to its image, and a seeded 20-cell
review is retained. Existing canonical `Kotguru` had two older attestations;
there is no exact form overlap with the new rows. The reproducible importer,
literal sound profile, YAML, audit and `bailey1908kotguru` Bib record use
append order 60. Four focused Kotguru tests, source metadata and all 23
parse/profile checks pass. The compiled build and downstream gates remain
deferred by the user's instruction.

### Kharia MLA source-side installation

Donegan and Stampe's archived Kharia Munda Lexical Archive file adds 829
conservative source rows from 3,620 uniquely numbered records; 2,791
exclusions have typed reasons. Five archive IDs already cited via Rau 2019
(`12601`, `12711`, `32441`, `3721`, `10041`) are explicitly excluded so the
same upstream records are not imported twice. Thirty-five installed rows
carry source uncertainty, and a seeded 20/829 direct archive/output review
found zero material errors. The subset has 29 exact spellings in common with
the existing 521-row Kharia Living source, retained as distinct attestations
pending compiled identity review. The archive's explicit same-conditions
reuse notice is carried through the source package; Romanization is retained
without invented IPA or dialect identity. Existing Bib key `DSKH` was enriched
rather than duplicated; append order 57. Four Kharia source tests, all 829
parse/profile checks, source metadata and the scoped sound-profile audit pass.

At this checkpoint 22 dated source CSVs contain 5,652 rows, each with 15
columns and a distinct nonempty entry key. Their combined source-specific
suite passes 125 tests. All rows have languages, forms
and valid BibTeX citation keys; no replacement characters or non-NFC fields
were found in a bounded scan. The full build, compiled reference/identity/
graph checks, full test suite and browser QA are still deferred by the
user's no-build/no-remote instructions.

### Kotkhai source-side installation

Bailey 1908 printed p. 24 gives five explicit Kotkhai lexical differences
in a short paragraph; all five are inventoried and visually checked. Two
clear readings, `Shāṇā` ‘see/look’ and `dēs` ‘sun’, are installed; field and
cold are typography holds. The rice reading `bīūjṇā` is excluded because
Jambu's sole older Kotkhai/Zoller row cites this same Bailey page and is
therefore the identical source evidence. The public-domain scan hash,
source-local audit and importer, preservation profile, YAML and
`bailey1908kotkhai` Bib record use append order 62. Three focused tests and
metadata validation pass. This is deliberately a complete five-item
source-stage slice, not an asserted complete Kotkhai vocabulary.

### Kumarbhag Paharia and Bilaspuri source-side installation

George Edward's 2019 Kumarbhag Paharia primary numeral page, preserved as
a pinned HTML snapshot and covered by the archived Numeralbank CC-BY-4.0
representation, has 40 displayed cells, 42 numbered prompts and 43
independent readings: numeral 20 prints two complete answers. Parenthetical
transcription equivalents and arithmetic commentary remain audit-only; no
unprinted components or cognacy links were inferred. A seeded 20-prompt
source/output review found zero material errors. Exact Glottolog mapping
creates canonical `Kumarbhag Paharia` (`kuma1274`, N. Dravidian), distinct
from existing Malto; broad Bihar/Jharkhand provenance does not supply a site
coordinate or dialect. Raw IPA remains in `Phonemic` and `Original`, while an
explicit profile handles display conversion and the one printed colon-length
symbol. Append order 61, Bib key `edward2019kumarbhag`. Five focused source
tests, all 43 parse/profile checks, metadata and full-inventory profile
policy pass. Kobayashi and Tirkey's unlicensed 2006 fieldwork table was used
only to check lect mapping, not systematically imported.

Bailey 1920 printed p. 245 left column, about–cold, has 33 Bilaspuri
lexical cells. High-resolution image review left six securely read rows,
25 typography holds and two mixed-use holds; three initially proposed
readings were moved to holds when additional marks became visible. The full
33-cell audit and seeded 20-cell review are retained. Existing canonical
`bil` has 31 Patyal etymological attestations and no exact form overlap
with the six newly installed Bailey readings. The public-domain scan hash,
source-local importer, literal profile, YAML and `bailey1920bilaspuri` Bib
entry use append order 64. Three focused source tests and metadata pass.

At this checkpoint 25 dated source CSVs contain 5,703 rows; their combined
source-specific suite passes 136 tests. The complete CLDF build, compiled
reference/identity/graph checks, full test suite and browser QA remain
deferred by the user's no-build/no-remote instructions.

### Kakkala and Shoracholi source-side installation

Ravi Sankar S. Nair's 2017 Kakkala primary numeral table has 40 displayed
cells and 43 explicitly printed readings, including two answers for ten and
three for one thousand. A seeded 20-prompt comparison found zero material
errors and no reading is held. The page's productive `-ji/-cci` alternation
is retained as source commentary, with no unprinted forms generated. New
canonical `Kakkala` (`kakk1234`, S. Dravidian I) carries the `Kuḷupe:ccu`
autonym as an alias, not an invented dialect; broad Kerala/Malabar location
does not provide a site coordinate. Raw IPA and the explicit display profile
are kept distinct. Append order 65, Bib key `nair2017kakkala`; five focused
tests, all 43 parse/profile checks, metadata and full-inventory profile
policy pass.

Grierson's 1916 LSI IX(IV) printed p. 602 gives a complete 27-item
Śōrāchōli of Rawain unusual-word list. Ten secure cells yield 11 installed
rows because house prints two answers. Fourteen typography holds and one
phrase-bound verb hold remain audit-only; field and garment are excluded as
likely the same print evidence already quoted by Zoller. The third existing
Zoller Shoracholi form is outside this list. All 27 entries were visually
checked, the public-domain scan is pinned, and a seeded 20-cell comparison
is retained. Canonical `Shoracholi` is reused; the historical Rawain label
is location provenance without invented coordinates. Append order 66, key
`grierson1916shoracholi`. Three focused tests and metadata pass.

At this checkpoint 27 dated source CSVs contain 5,757 rows and their
combined source-specific suite passes 144 tests. The complete CLDF build,
compiled reference/identity/graph checks, full test suite and browser QA
remain deferred by the user's no-build/no-remote instructions.

### Ho MLA and Khirwar source-side installation

The 2020-10-22 archived Donegan–Stampe Ho file is explicitly incomplete:
it contains 1,516 numbered Deeney 1978 records covering only A–C and ends
at source ID 01516. Its archive-specific same-conditions reuse notice and
exact snapshot SHA-256 are retained in the source package. A conservative
top-level single-sense screen installs 91 rows and accounts for all 1,425
exclusions. Three records already represented by `DHED` are excluded as
the same source evidence, while the `baba` ‘father’ record is kept distinct
from the existing `baba` ‘paddy’ homonym. Sixteen initially selected `ch`
headwords were deferred because archive aspiration notation is ambiguous
for loans; eight further donor/narrative/inflection/register-rich rows were
held rather than flattening those claims into glosses. Source `w` remains
in Original and converts to house `v`. The direct seeded 20-item source/
output comparison found zero material errors. Existing `DHED` bibliography
was enriched without a duplicate key. Append order 63; four focused tests,
all 91 parse/profile checks, metadata and scoped profile policy pass.

Suraj Mini's 2018 Khirwar numeral table has 40 displayed cells and 42
numbered prompts. Forty readings are installed; items 5 and 50 remain
typed transcription holds because of unexplained internal spacing. The
source explicitly describes these numerals as Indo-Aryan loans, so each
installed row bears the existing `loanword` tag, but no specific donor or
etymon edge is inferred. New canonical `Khirwar` (`khir1237`, S. Dravidian
II) uses broad Garhwa/Latehar provenance without invented coordinates or
dialect. The primary page is snapshotted and its archived representation
is CC-BY-4.0. The seeded 20-item comparison found zero material errors;
source `ɑ` and `ɐ` distinctions are preserved without guessing length.
Append order 67, key `mini2018khirwar`; five focused tests, all 40
parse/profile checks, metadata and full-inventory profile policy pass.

At this checkpoint 29 dated source CSVs contain 5,888 rows. The only
global sound-profile policy findings are the previously identified eight
`sil-gutob-gorum` and one `sil-pahari-pothwari` rules; the new Ho and
Khirwar profiles have none. The complete CLDF build, compiled reference/
identity/graph checks, full test suite and browser QA remain user-deferred.

### Śōdōchī source-side installation

Grierson's 1916 LSI IX(IV) printed p. 663 Śōdōchī column is bounded to
prompts 1–13 and 32–50 (32 audited cells); pronouns 14–31 and the mixed
p. 648 prose glossary are excluded from scope. Seventeen securely read
cells yield 18 installed rows; 14 typography holds and one complex-response
hold remain in the source-local audit. LSI p. 648 explicitly equates its
Śōdōchī specimens with Bailey's Kotgurū, so the column maps to Jambu's
existing canonical `Kotguru`, with a source-qualified Śōdōchī dialect/
alias and blank coordinates, rather than splitting the same historical
lect across Kotguru and broad `sat` Satlaj. The public-domain scan and
seeded 20-cell review are retained. Tooth and ear match Bailey p. 31
forms/glosses; these are retained as distinct printed attestations without
claiming independent field collection. Append order 68, key
`grierson1916sodochi`. Three focused source tests, metadata, all 18
parse/profile checks, and source/dialect validation pass. A broader sound
profile test still fails solely on the known older `sil-gutob-gorum`
policy mismatch. The full CLDF/graph/reference/browser gates remain
user-deferred.

### Paniya source-side installation

Stephen Daniel's 2013 primary Paniya numeral table has 40 explicit cells.
Thirty-four securely printed readings are installed and six values
(8, 18, 23, 24, 28 and 70) remain in the complete audit as typed holds
for semicolon, diacritic or geminate transcription anomalies. The original
source was not silently emended from an archive comparator. A seeded
20-item raw/output comparison found zero material errors. Existing
canonical `Paniya` (`pani1256`) is reused; its translator's address is
provenance, not a dialect or exact elicitation site. This source will add
Paniya's first lexical rows to a future compiled build. Append order 69,
key `daniel2013paniya`. Five focused tests, all 34 parse/profile checks,
metadata and full-inventory profile policy pass. With this package, 31
dated source CSVs contain 5,940 rows; the full build, graph/reference/
identity checks, full suite and browser QA remain deferred by the user's
no-build instruction.

### Simla Sirājī source-side installation

Grierson's 1916 LSI IX(IV) printed p. 631 has a separately headed Simla
Sirājī column. The bounded lexical scope is prompts 32–52, 21 complete
target cells; pronouns 26–31 and the adjacent Śōrāchōlī control column
are excluded. Fifteen accepted cells produce 16 rows because brother has
two independently printed answers; mouth, tooth, hair, head, sister and
woman remain typography holds. All 21 cells and a seeded 20-cell image
comparison are in the source-local audit. Existing canonical
`ShimlaSiraji` is reused without a finer invented dialect/site; its two
older Zoller forms have no prompt overlap with this span. The scan is
pinned and the importer, profile, YAML and `grierson1916simlasiraji` Bib
entry use append order 71. Three focused source tests, all 16 parse/
profile checks and source metadata pass. At this checkpoint, 32 dated
source CSVs hold 5,956 rows; compiled/full-suite/browser gates remain
deferred by the user's no-build instruction.

### CFEL Koda 2022 bounded print pilot

The 2022 first edition of Pradhan and Tripathi's *English–Hindi–Bangla–Koda
Dictionary* (ISBN 978-81-957226-0-0) is sampled at seven exact publisher
PDF pages, whose 53 visible entries are all inventoried. Seventeen visually
checked Koda forms are installed; 36 entries are excluded, including 33
outside the bounded pilot, Cow and Two for print/API IPA uncertainty, and
Die for a publisher API-versus-print edition mismatch. The 20-item visual
sample covers all 17 installed rows and those three explicit deferrals with
zero material print-comparison errors. The publisher API supplies stable
IDs, Unicode and IPA; the source package retains only minimal selected
lexical evidence and full-response hashes. Every selected Unicode spelling
was checked against the printed page; uncertain IPA is not guessed. The
seven-page PDF (371 PDF pages overall) is hash-pinned as a local research
cache and not installed in the repository.

The PDF is © Visva-Bharati with no identified open licence. Under the
project's established extracted-lexical-facts editorial precedent, the
installed CSV contains attributed lexical facts and page/item locators,
not PDF pages, images, prose definitions or full API responses. Its reuse
must be reconsidered before any public release; this local source-stage
installation is not a permission claim. Existing canonical Koda is reused
without a made-up dialect/site. Five of 17 English labels overlap the
compiled Koda gloss set, but none is the same source record. The 2022
print edition has its own `pradhan-tripathi2022koda` bibliography key,
append order 70, separate from the earlier CFEL adornments API source.
Four focused tests, 17/17 parse/profile checks and source metadata pass;
full build/compiled/browser gates remain user-deferred.

### Korra Koraga source-side installation

Sharma's 1990 Korra Koraga primary numeral table has 40 numbered cells:
29 printed readings are installed and 11 source-blank cells (21–29, 200,
2000) are explicitly audited as blanks. A seeded 20-item raw comparison
found zero material errors, and all 29 rows parse under the explicit
sound profile without replacement characters. The source lect maps
exactly to new canonical `Korra Koraga` (`korr1238`), a spoken L1; Jambu's
older generic `Koraga` uses the broader family-level `kora1289` code and
its existing attestations are left untouched pending a separate registry
review. No extra dialect or precise site coordinate is inferred. The
primary HTML is pinned, its archived representation is CC-BY-4.0, and
the source-local audit, importer, YAML and bibliography use append order
72. Five focused source tests, source/dialect checks, metadata and the
full-inventory profile-policy check pass. Full build/compiled/browser
gates remain deferred.

### Rāmpur Kōci source-side installation

Bailey 1920 printed p. 144 has paired Rāmpur and Bāghī answers. The
bounded first-column Rāmpur scope contains 59 inventoried units: 17
securely read rows installed, 34 typography holds, six mixed grammar/
semantic holds, and two cross-reference exclusions. All 59 units have
image-backed audit records and a seeded 20-cell visual review found zero
material errors. Bailey p. 113 and p. 144 support mapping Rāmpur Kōci to
existing canonical `ramp`; the post-colon Bāghī answers are explicit
controls and the paired packages share print provenance rather than
independent field collection. The scan hash, source-local importer, YAML,
profile and `bailey1920rampur` Bib record use append order 73. Three
source-specific tests (15 in the agent's related focused set), all 17
parse/profile checks and source metadata pass. At this checkpoint, 35
dated source CSVs contain 6,019 rows; their combined source-specific suite
passes 176 tests. A bounded schema/Unicode/citation check found no empty
required cells, duplicate entry keys, missing bibliography keys,
replacement characters or non-NFC fields. Full build, graph/reference,
full-suite and browser gates remain user-deferred.

### Roy Birhor two-page source and held Pal Kurumba candidate

Sarat Chandra Roy's public-domain *The Birhors* (1925), Appendix I printed
pp. 567–568 / PDF pp. 655–656, contributes a bounded historical Birhor
slice. All 59 two-column heads were inventoried and visually checked:
46 installed and 13 held for apostrophe, `ch`/`chh`, clustered consonant
or underdot transcription uncertainty. The initial p. 567 keys remain
stable after adding p. 568. Roy/Pinnow matching forms are retained as
separate cited attestations because direct dependence is unproven;
`āji` ‘grandmother’ remains distinct from Pinnow `aji` ‘elder sister’.
Mundari/Hindi comparison abbreviations remain source notes, not inferred
etymology or graph edges. The 1925 title page, both page images and scan
SHA-256 are recorded in one source package and one CSV/YAML, now named
`20260925-roy-birhor-p567-p568`, with append order 74 and bibliography
key `roy1925birhors`. Four source tests, all 46 parse/profile checks,
metadata and scoped sound policy pass. At this checkpoint 36 dated
source CSVs contain 6,065 rows.

The separate 2020 Attapady/Pal Kurumba numeral page was **not installed**:
the available CC-BY archive scrape predates that elicitation (2019), so
it cannot establish reuse terms for the later page. Its 46-cell analysis
(37 readings and nine blanks) is held only under `raw_data/`; the active
CSV/YAML/profile and copyrighted HTML were removed from the installed
source inventory. No Pal Kurumba rows or bibliography entry are active.
The full build, compiled reference/identity/graph checks, full tests and
browser QA remain user-deferred.

### Kisan (Odisha) numeral source-side installation

The dated 2019-02-12 CC-BY-4.0 Numeralbank archive contains the exact
2018 `Kisan_Odisha.htm` primary page and two separately credited tables
(Kujur and Perumalsamy). Their 82 numbered units produce 73 installed
readings: 68 accepted cells plus five independent second answers. Twelve
source cells are blank and two rows repeat printed `4` where `5` might
have been expected; both anomalous rows remain held rather than silently
renumbered. Seven source-typography flags are retained. A seeded 20-cell
raw/output audit found zero material errors. The Dravidian Kisan lect maps
to Jambu's existing canonical `Kurux`, with source-qualified Kisan (Odisha)
dialect `kisa1261`; unrelated Indo-Aryan `KisanIA` remains untouched.
Contributor/table/row-qualified keys preserve both witnesses. Append
order 75, bibliography key `kujur-perumalsamy2018kisan` and explicit profile route. Six source
tests, source/dialect checks, all 73 scoped parse/profile checks, metadata
and full-inventory profile policy pass. At this checkpoint 37 stable
dated source CSVs hold 6,138 rows; compiled/full-suite/browser gates
remain user-deferred.

### Mālvī (Rāngrī) source-side installation

Grierson's 1908 LSI IX(II) printed p. 307 has a distinct Mālvī (Rāngrī)
column for lexical prompts 32–52, alongside a separate Standard Mālvī
difference column treated as control. All 21 target cells and adjacent
control cells were inventoried. Five target readings are securely legible
and installed; 16 remain typography holds because both the Commons DjVu
and original Internet Archive JP2 resolve to the same 992×1404 page
image. The seeded 20-cell visual review found zero material errors in
the accepted/excluded classification. Existing canonical `Malw` is
reused, with source-qualified Rāngrī dialect `lsi1908-malvi-rangri`
and blank coordinates; the registry's Malwai display name versus source
Mālvī wording is recorded rather than silently renamed. Append order
76, key `grierson1908malvirangri`. Three focused source tests,
source/dialect checks, all five parse/profile conversions and metadata
pass. At this checkpoint 38 dated source CSVs hold 6,143 rows and their
combined source-specific suite passes 189 tests. Full
build/compiled/browser gates remain user-deferred.

### Birbhum Koda isolated LSI examples

Grierson/Konow's public-domain LSI IV (1906) printed pp. 109–110 gives
20 directly glossed Kōḍā/Kōḍī examples from Rev. P. O. Bodding's
Birbhum specimen. Five clearly legible simplex lexical forms are
installed; 15 examples remain audit-only because of transcription,
inflection or pronoun analysis. The interlinear narrative on pp. 111–113
and the separately warned-corrupt Bankura specimen on pp. 114–115 are
excluded rather than turned into inferred word pairs. A source-qualified
Birbhum dialect is registered under existing canonical Koda, with no
invented site coordinate. The existing `grierson1906lsi4` bibliography
record was expanded rather than duplicated; append order 77. Three
source tests (21 in the related LSI/Koda/dialect focused set), all five
parse/profile checks, metadata and the scoped sound-profile policy pass.
Full build/compiled/browser gates remain user-deferred.

The earlier SIL Gutob/Gorum sound-profile policy mismatch was resolved
without changing the eight source-faithful output mappings. The profile
preserves source contrasts between ɪ/i, ʊ/u, ə/a, ʌ/a and ɕ/ʃ; eight
exact grapheme/profile exceptions were documented in `profile_policy.py`
and a corpus-contrast regression test added. The full sound-profile test
file now passes; the only whole-inventory policy finding is the older
Pothwari length-mark exception, already accounted for by that test.

### Southeastern Kolami (Naikri) numeral source-side installation

The dated 2019-02-12 CC-BY-4.0 Numeralbank archive preserves Subhangi
Kardile's 2013 Southeastern Kolami table. Its 40 filled target cells yield
43 installed readings, including separate printed answers at 16, 40 and
200. The adjacent 40-cell Northwestern Kolami table is audited as an
excluded comparator. The archived Chinese heading and one sentence
conflict with the English heading, credit and separately headed Northwestern
table; this provenance conflict is recorded, and no borrowing claim or
directional variant is inferred. The source lect reuses existing canonical
`Naikri` (`sout1549`), rather than adding a duplicate; one form string
overlaps 689 previously compiled Naikri rows, with a different source
attestation. An archived HTML hash, table-qualified extraction audit and
20-cell review (zero material errors) are retained. Append order 78,
bibliography key `kardile2013southeasternkolami`; focused tests and all 43
scoped parse/profile checks pass. Full compiled/browser gates are deferred.

### Bailey Kiunthali bounded source-side installation

Bailey's public-domain 1908 Kiunthali glossary, printed p. 18/PDF p. 40
left column, has 40 printed lines from water through high. Thirteen
visually secure readings are installed under existing canonical `kiuth`;
27 lines remain transcription holds for unresolved diacritics or letter
boundaries. The separate right-column continuation and grammatical text
are outside the declared scope. All 40 lines have page/column/line audit
locators; a seeded 20-line visual review found zero material errors among
accepted forms. No matching Bailey p. 18 print-cell form plus gloss was
found in existing Zoller rows. Append order 79, bibliography key
`bailey1908kiunthali`; focused tests and 13 scoped parse/profile checks
pass. Full compiled/browser gates are deferred.

### Tūri LSI site-example source-side installation

The same public-domain LSI IV (1906), printed pp. 128–129/DjVu pp.
147–148, contains 24 explicitly glossed examples or controls associated
with Tūri sites. Three secure simplex responses are installed: Sambalpur
`hor` ‘man’, Jashpur `ñel` ‘see’, and Ranchi `lel` ‘see’. Twenty-one units
are held for numeral diacritics, inflection, unstable pronouns, conjecture
or comparison-control status. Interlinear specimens from p. 130 onward
are excluded; no prose-derived pairs or comparison-based graph links are
asserted. Three site dialects are registered under existing Turi with
blank coordinates. The source package retains page images, raw audit,
scan identity and overlap review; there are zero exact form-plus-gloss
matches against 269 previously compiled Turi rows. The existing
`grierson1906lsi4` bibliography key was extended, append order 80, and
the dedicated profile preserves all three source forms. Focused tests
(six Tūri/Koda checks), metadata and 3/3 scoped parses pass. Full
compiled/browser gates remain deferred.

### Bailey Bhalesi bounded source-side installation

Bailey's public-domain 1908 Bhalesi comparison list, printed pp. 73–74/
PDF pp. 187–188, has 34 directly glossed lines across a page and column
break. Fifteen secure cells yield 16 installed rows because the father
line prints two answers. Sixteen typography holds, one same-print
LSI/Zoller woman exclusion, one horse/mare stem hold and one incomplete
suffix hold account for the remaining lines. Adjacent lects, numbered
sentences and grammar are outside scope. Existing canonical `bhal` is
reused without an invented coordinate. Every line has a page/column/item
audit locator; a seeded 20-line visual review reports zero material
accepted-form errors. Append order 81, bibliography key
`bailey1908bhalesi`; four focused tests and local scoped parsing pass.
Full compiled/browser gates remain deferred.

### Aheri Gondi numeral source-side installation

The dated 2019-02-12 CC-BY-4.0 Numeralbank archive preserves Benny
Kurian's 2018 Aheri Gondi table. Twenty two-column HTML rows contain 42
explicitly numbered prompts/readings, all installed. Two mixed cells are
split only at their printed 100/200 and 400/800 labels. Adjacent inline
spans for 3 and 23 are joined without a fictitious intervening space;
raw markup and joined text remain in each audit record. A seeded 20-prompt
review found zero material errors, and no target prompts are held.
Glottolog's `aher1237` supports a distinct canonical `Aheri Gondi`, with
broad Maharashtra provenance and blank coordinates; generic Gondi is
unchanged. Source IPA remains in Original/Phonemic, while the dedicated
profile maps source vowel/consonant length into house conventions. No
numeral etymons or component edges are inferred. Append order 82,
bibliography key `kurian2018aherigondi`; 23 combined Aheri/Bhalesi/
sound-profile checks, metadata and 42/42 scoped parsing pass. Full
compiled/browser gates remain deferred.

At this checkpoint, 44 dated source CSVs hold 6,265 rows. A scan of all
dated CSVs finds no malformed 15-column rows, missing required fields,
non-NFC forms, duplicate entry keys, missing companion YAML files, or
colliding append orders. The 44 dated-source test modules pass 213 tests.

### Crooke Mirzapur Korwa source-side installation

Crooke's four-page Korwa glossary in the 1892 *Journal of the Asiatic
Society of Bengal* is explicitly public domain on the BHL article record
(whose catalog year is 1893; printed running heads say 1892). All 123
form–gloss lines on printed pp. 125–128 were audited: 93 direct lexical
forms installed under existing Korwa `kw`, and 30 inflected, multiword or
uncertain units held. The separate Driver article begins below Crooke's
13th p. 128 line and is excluded. A source-qualified southern Mirzapur
dialect has blank coordinates because the exact speaker locality is
unspecified. `lutur` ‘ear’ overlaps later independent Bahl/Pinnow
attestations, retained with its own citation; no graph edge is inferred.
The page-stratified 20-line image review found zero material accepted-form
errors. Append order 83, bibliography key `crooke1892korwa`; 24 focused
source/dialect/sound checks, metadata and 93/93 scoped parses pass. Full
compiled/browser gates remain deferred.

### Bailey Pādari bounded source-side installation

Bailey's public-domain 1908 Pādari list, printed p. 82/PDF p. 196,
contains 28 directly glossed right-column lines from pig through river.
Thirteen secure forms are installed under existing canonical `Padri`;
13 typography-uncertain lines are held, and two same-print hair/fox
citations already represented through Zoller/LSI are excluded. The left
column, continuation, paradigms and sentences are outside the declared
scope. All 28 lines have page/column/item audit locators, and a seeded
20-cell image review found zero material accepted-reading errors. No
precise historical collection point is inferred. Append order 84,
bibliography key `bailey1908padari`; four focused tests and local parsing
pass. Full compiled/browser gates remain deferred.

### Adilabad Gondi numeral source-side installation

The 2019-02-12 CC-BY-4.0 archive preserves Mark Penny's 2013/2017
Western Southern/Adilabad Gondi table. Forty numbered cells give 50
installed readings: six cells have two printed alternatives and two have
three. Eight individual answers are annotated `< Indic` and tagged
`loanword`; unmarked alternatives and forms sharing their components are
not tagged by inference, and no donor edge is fabricated. Raw markup,
per-answer qualifiers, 40-cell audit and a seeded 20-cell zero-error
review are retained. This explicitly labelled source maps to new precise
canonical `Adilabad Gondi` (`utno1237`) with blank coordinates. Historical
generic-Gondi `adil` material, including legacy pseudo-language rows,
remains untouched pending a full-build identity audit; no same-print Penny
attestation was found among it. Append order 85, bibliography key
`penny2017adilabadgondi`; five focused tests, metadata and 50/50 scoped
parses pass. The combined sound-profile inventory is to be rerun after
concurrent Rambani metadata staging; full compiled/browser gates remain
deferred.

### Bailey Rambani bounded source-side installation

Bailey's public-domain 1908 Rambani numbered list, printed p. 48/PDF
p. 272 left column, contains items 1–36. Twenty-one secure responses are
installed under existing canonical `ram`; 14 typography-uncertain units
are held and item 6 ‘six’ is excluded as a likely same-print Zoller
quote. Eight accepted numeral-section rows receive only the explicit
`num` tag; grammar and etymology are not inferred. Every item has a
source locator, and a seeded 20-item visual review found zero material
accepted-reading errors. The right column, p. 49 continuation and
neighboring Poguli section are outside scope. Append order 86,
bibliography key `bailey1908rambani`; four focused tests, 21/21 scoped
parses, metadata and the full sound-profile test pass. Full
compiled/browser gates remain deferred.

At the post-Korwa/Rambani checkpoint, 48 dated source CSVs hold 6,442
rows and the 48 dated-source test modules pass 230 tests. This precedes
subsequent Jennu Kurumba and Bhadrawahi source staging.

### Jennu Kurumba numeral source-side installation

The CC-BY-4.0 2019 archive preserves Melwin Jeba's 2018 Jennu Kurumba
table and a separately credited 2015 Kodagunti table. Jeba's 42 explicit
number prompts give 42 installed readings; 40 second-table HTML cells are
audited as excluded comparator units, for 82 audit units total. The first
table's apparent gap at 26 was introduced by naive spacing between
adjacent HTML spans; raw markup supports the continuous form. A seeded
20-prompt audit found zero material errors. The explicit lect maps to new
canonical `Jennu Kurumba` (`jenn1240`) with broad Karnataka provenance
and no invented coordinate. About 300 historical Jennu-labelled Kannada
dialect forms remain untouched pending a full-build identity review; no
exact house-form overlap was found with Jeba's 42 readings. Append order
87, bibliography key `jeba2018jennukurumba`; five focused source tests,
the broader 25-test source/sound/dialect suite, metadata and 42/42
scoped parses pass. Full compiled/browser gates remain deferred.

### Bailey Bhadrawahi bounded source-side installation

Bailey's public-domain 1908 Bhadrawahi glossary, printed p. 65/PDF
p. 179 left column, contains 42 directly glossed lines from plain
through sweet. Twenty-two secure cells yield 23 rows because the swift
line prints two answers; 20 typography-uncertain cells are held. The
right column, preceding list page, grammar and nearby Bhalesi/Pādari
sections are outside scope. Existing canonical `bhad` is reused; no
secure form-plus-gloss duplicate was found among 59 Zoller rows. Every
line has an audit locator, and a seeded 20-line image review found zero
material accepted-reading errors. Append order 88, bibliography key
`bailey1908bhadrawahi`; four focused tests, metadata and 23/23 scoped
parses pass. Full compiled/browser gates remain deferred.

At this checkpoint, 50 dated source CSVs hold 6,507 rows. All 50 have
companion YAML files, unique append orders and unique row keys; the rows
have valid 15-column shape, required fields and NFC forms. The 50
dated-source test modules pass 239 tests.
The complete sound-profile test file separately passes 14 tests at this
checkpoint; the older Pothwari length-mark policy exception remains its
only allowed whole-inventory case.

### Hislop/Temple Kuri–Muasi (Korku) source-side installation

The public-domain 1866 Hislop/Temple comparative vocabulary has a
source-labelled Kuri/Muâsi column on printed pp. 1–4/PDF pp. 53–56.
The editor and later classification evidence identify this historical
lect with Korku, not adjacent Gondi or Keikadi control columns. All 32
English prompt cells were visually audited: 17 filled lexical cells
yield 18 installed forms (two basket alternatives), 14 cells are blank,
and `Be (v.) Danyâ` is held as grammatical. A draft bag reading was
corrected to printed `Tēili` before installation; the complete 32-cell
300-dpi recheck found zero residual material accepted-form errors. Keys
use printed page and prompt ordinal because Temple reordered Hislop's
manuscript. Existing canonical `ko` gains a source-qualified historical
Kuri/Muasi dialect with blank coordinates. Eight later Stahl rows
overlap two Hislop forms; independent citations remain separate. Append
order 89, bibliography key `hislop1866papers`; four source-focused tests,
the 24-test source/dialect/sound set, metadata and 18/18 scoped parses
pass. Full compiled/browser gates remain deferred.

### Bailey Rohru bounded source-side installation

Bailey's public-domain 1920 Rohru glossary, printed p. 127/PDF p. 153
left column, contains 35 lines from able through cock. Twelve secure
cells yield 13 installed rows because anyone/anything are printed as
two answers; 22 typography-uncertain units are held and one
cross-reference-only cell is excluded. Existing canonical `roh` is
reused, with no invented new collection coordinate; no accepted exact
form-plus-gloss overlap was found in its five earlier source-stage rows.
The 35-line audit uses page/column/line keys and a seeded 20-cell visual
review reports zero material accepted errors. Append order 90,
bibliography key `bailey1920rohru`; four source-focused tests, a 24-test
source/sound/registry set, metadata and 13/13 scoped parses pass. The
alternative Bailey Doda Siraji list was screened out because all 36
gloss prompts mirror an existing LSI set and the historical `dod`/`sir`
mapping is conflicted. Full compiled/browser gates remain deferred.

### Koya numeral source-side installation

The CC-BY-4.0 2019 archive preserves the Andronov/BSI 1995 Koya table.
Its 40 numbered cells contain 62 printed answer segments; 58 complete
forms are installed and four abbreviated slash fragments remain held
instead of being silently expanded. Eight individually marked `< Indic`
answers receive `loanword` without invented donor edges. The source's
printed alternative under 100 is retained, but its conflicting `(2 x
20)` arithmetic note does not create a component relation. A seeded
20-cell audit found zero material errors. New precise canonical `Koya`
(`koya1251`) has broad India provenance and blank coordinates. About
2,262 historical Koya-related dialect rows under generic Gondi remain
untouched pending a full-build identity review; four new forms overlap
independently cited older forms, with no same-print 1995 citation found.
Append order 91, bibliography key `andronov1995koya`; five focused tests,
the broader 25-test source/sound/dialect set, metadata and 58/58 scoped
parses pass. Full compiled/browser gates remain deferred.

At this post-Koya checkpoint, 53 dated source CSVs hold 6,596 rows.
All have companion YAML settings and unique append orders; no row has
invalid width, missing required fields, non-NFC form or duplicate
entry key. The 53 dated-source test modules pass 252 tests.

### Cust/Norton Korku bounded source-side installation

The public-domain 1884 JRAS English–Kor vocabulary begins on printed
p. 165/PDF p. 184. All 65 two-column headword cells on that page were
visually audited: 45 selected cells yield 49 installed forms after four
printed alternatives; 20 cells remain held for paradigms, unexplained
`(F.)` markers, segmented or non-equivalent species lists. The reverse
Kor–English index on pp. 173–177 repeats the underlying records and is
excluded; forward pp. 166–172 remain outside this pilot. Cust only says
the vocabulary *appears* to be Norton's, and Ellichpūr is context rather
than a proven elicitation point. The source-qualified historical Korku
dialect therefore has blank coordinates. Enlarged print resolved two
macrons that OCR suggested were diaereses; the complete audit and
page review retain those decisions. Fifteen exact-form overlaps with
later Korku witnesses are kept as separate citations. Append order 92,
bibliography key `cust1884korku`; three focused tests, metadata and
49/49 scoped parses pass. Full compiled/browser gates remain deferred.

### Malasar numeral source-side installation

The CC-BY-4.0 2019 archive preserves Aswini Babu's 2018 Malasar table.
Its 40 HTML cells print 41 prompts/readings because one cell separately
labels 400 and 800; 200 is absent and was not reconstructed. All 41
readings are installed with raw cell audit and stable keys. A seeded
20-prompt review plus boundary checks found zero material errors.
The exact lect reuses existing canonical `Malasar` (`mala1458`), not
separate MalaMalasar (`mala1457`); no site or coordinate is invented.
Earlier 446 compiled Malasar forms are chiefly Varghese 2015 survey
witnesses, with no exact display-form/numeral-gloss overlap here. The
profile preserves unusual source symbols rather than imposing an
unwarranted analysis; no borrowing or cognacy edge is inferred. Append
order 93, bibliography key `babu2018malasar`; five focused tests,
the broader 25-test source/sound/dialect suite, metadata and 41/41
scoped parses pass. Full compiled/browser gates remain deferred.

### Haijong (Mymensingh) LSI source-side installation

Grierson's public-domain LSI V(I) (1903) printed p. 354/scan image 370
has a Haijong (Mymensingh) column aligned to exactly 25 numbered
English prompts on printed p. 352/scan 368. Fourteen securely legible
cells are installed; 11 remain typography holds for unresolved length,
nasal or consonant marks. Adjacent Bengali/Siripuri columns are
excluded controls, and only numeral prompts receive `num`. Existing
canonical `Hajong` (`hajo1238`) gains a source-qualified historical
Mymensingh dialect with blank coordinates. A scan-hashed importer and
25-cell audit preserve both page locators and separate prompt keys;
the seeded 20-cell visual review found zero material accepted errors.
Restricted SIL 2011 Hajong lists were not used. Append order 94,
bibliography key `grierson1903haijong`; four source-focused tests,
the 24-test source/sound/dialect set, metadata and 14/14 scoped parses
pass. Full compiled/browser gates remain deferred.

### Eravallan numeral source-side installation

The CC-BY-4.0 archive preserves a Vijayan 2018 Eravallan table with
42 explicit prompts/readings in 40 HTML cells, and a separately
credited Gnanasundaram 2014 table with 47 prompts. The 42 later readings
are installed; all 47 older readings are audited as excluded comparator
units. Twenty-two of 39 shared prompt readings are identical, so the
tables are not claimed as independent elicitation witnesses and no
cross-table variant edges are inferred. A seeded 20-prompt audit found
zero material errors. Existing canonical `Eravallan` (`erav1242`) is
reused with no invented site; four installed forms overlap independently
cited Varghese survey rows. The profile preserves source symbols whose
phonological interpretation is not secure. Append order 95,
bibliography key `vijayan2018eravallan`; five focused tests, the 25-test
source/sound/dialect set, metadata and 42/42 scoped parses pass. Full
compiled/browser gates remain deferred.

### Betta Kurumba numeral source-side installation

The CC-BY-4.0 archive contains a Selvaraj/BSI 1996 Betta Kurumba
table with 40 numbered cells/readings and a separately credited Coelho
2010/2011 table with 40 cells. All Selvaraj cells are installed; all
Coelho cells are audited as excluded comparators, preserving its
phonemic/phonetic layers where well delimited and flagging malformed
delimiters at 200 and 1000. No independence or revision relation between
the two elicitation sets is assumed. A seeded 20-cell review found zero
material errors. Existing canonical `BettaKurumba` (`bett1235`) is reused
with no invented site; one reading overlaps an independently cited Blair
survey witness. Append order 97, bibliography key
`selvaraj1996bettakurumba`; five focused tests, the 25-test
source/sound/dialect suite, metadata and 40/40 scoped parses pass. Full
compiled/browser gates remain deferred.

At this checkpoint, 58 dated source CSVs hold 6,782 rows. All 58 have
companion YAML settings, unique append orders and unique entry keys;
row widths, required fields and NFC forms check cleanly. The 58
dated-source test modules pass 274 tests.

### Hahn Asur comparison-page source-side installation

Hahn's public-domain *A Primer of the Asur dukmā* is printed 1900
(some catalogs say 1901). Its p. 170/PDF p. 182 comparison table has
31 Asur target cells, 31 Mundari controls and English glosses. Twenty-
nine directly glossed Asur forms are installed; a plural suffix and a
paired deictic cell without individual gloss alignment remain typed
holds. All target cells were visually reviewed at full page resolution
with zero residual material accepted-form errors. The separate
possessive-inflected kin list and later examples are outside this pilot.
Existing Asuri canonical is reused; no source speaker/site coordinate
or graph relation is invented. `bitil` ‘sand’ overlaps an independently
cited CUJ witness and is retained as separate attestation. Append order
98, bibliography key `hahn1900asur`; three source-focused tests, metadata
and 29/29 scoped parses pass. Full compiled/browser gates remain deferred.

### Muduga numeral source-side installation

The CC-BY-4.0 archive preserves a 2018 Muduga table credited to Siby
Kuriakose for voice material and Stephen Daniel for IPA transcription.
Forty HTML cells print 42 explicit prompts/readings because 100/200 and
400/800 share cells; all 42 are installed. Raw adjacent spans at 15 and
25 have no literal gap, so their continuous forms are joined without
invented spaces. A seeded 20-prompt audit including both cases found
zero material errors. Existing canonical `Muduga` (`mudu1239`) is reused;
the transcriber's address and credited audio are provenance, not a
collection site, so no new coordinate or dialect is inferred. Earlier
558 Muduga forms are chiefly Varghese survey witnesses, with no exact
display-form/numeral-gloss overlap here. The profile retains unclear
source symbols without phonological guesses, and no donor or cognacy
edges are inferred. Append order 99, bibliography key
`kuriakose2018muduga`; five focused tests, the 25-test
source/sound/dialect suite, metadata and 42/42 scoped parses pass. Full
compiled/browser gates remain deferred.

At this checkpoint, 60 dated source CSVs hold 6,853 rows. Every CSV has
a companion YAML with a unique append order; all rows have valid shape,
required fields, NFC forms and globally unique entry keys. The 60
dated-source test modules pass 282 tests.

### Pottangi Ollar Gadaba numeral source-side installation

The CC-BY-4.0 archive preserves Varghese John's 2013 Pottangi Ollar
Gadaba table and a separately credited 1990 Konekor Gadabar comparator.
Forty selected numbered cells yield 46 installed answers because numerals
1–3 each print three class-marked forms. Only explicitly marked masculine,
feminine and singular distinctions become tags; the first answer's
non-human reading is retained in the audit, not imposed as neuter.
Numeral 4 alone is marked `< Oriya` and tagged `loanword`, with no
invented donor-form edge. All 40 comparator cells are audited as
excluded, including 11 printed blanks. A seeded 20-cell review found
zero material errors. The exact lect reuses canonical `OllariGadaba`
(`pott1240`) with no invented site; three form/gloss overlaps with
independently cited Bhattacharya rows are retained as new source
attestations. Append order 101, bibliography key
`john2013pottangiollargadaba`; five focused tests, the 25-test
source/sound/dialect set, metadata and 46/46 scoped parses pass. Full
compiled/browser gates remain deferred.

### Samuells Juang source-side installation

Samuells's public-domain 1856 Juang glossary, printed pp. 302–303,
contains 31 prompt cells (24 plus seven). Twenty directly glossed
simplex cells yield 21 installed forms because water has two printed
alternatives; 11 phrase, inflection or uncertain-compound cells remain
held. Every cell was compared with the original pages; `Runkoo` ‘rice’
corrects misleading OCR `Kunkoo`, with zero residual material accepted
errors. Samuells says he collected the list during his 1854–56 visits,
but no form has a secure per-entry speaker/site, so the source-qualified
Juang lect has blank coordinates. The 1856 printed versus 1857 catalog
date is documented. One normalized `Minna` overlap with Pinnow is
retained as a separate citation; Dalton's later Juang column remains
analysis-only because its original collector is unclear. Append order
102, bibliography key `samuells1856juang`; three focused tests, metadata
and 21/21 scoped parses pass. Full compiled/browser gates remain deferred.

### Eastern Muria numeral source-side installation

The CC-BY-4.0 archive preserves Irene van Riezen's 2013 Eastern Muria
table. Of 40 numbered cells, 39 secure readings are installed; numeral
29 has a printed unmatched closing bracket and is held verbatim rather
than repaired. Only 7 and 21 are locally annotated “as in Hindi” and
tagged `loanword`; broad prose about Hindi numerals is not applied to
unmarked cells. The seeded 20-cell audit found zero material errors.
The explicit source lect maps to new precise canonical `Eastern Muria`
(`east2340`) with blank coordinates. Legacy generic-Gondi `muria` uses
family-level `muri1262` and its 809 compiled forms remain untouched
pending a full-build identity audit; none is an exact form/gloss overlap
with these 39, though nine other Gondi-site rows overlap. Append order
103, bibliography key `riezen2013easternmuria`; five focused tests,
the 25-test source/sound/dialect suite, metadata and 39/39 scoped parses
pass. Full compiled/browser gates remain deferred.

At this checkpoint, 63 dated source CSVs hold 6,959 rows. Companion YAML
coverage, append-order uniqueness, row widths, required fields, NFC forms
and entry-key uniqueness check cleanly. The 63 dated-source test modules
pass 295 tests.

### Haldar/Dalton Juanga compilation source-side installation

Dalton's public-domain 1872 table credits Babu Rakhal Das Haldar as
compiler, not necessarily collector. The bounded printed p. 236 Juanga
column has 42 English prompt rows: 29 directly glossed lexical rows
yield 31 installed forms after two printed alternatives each for brother
and dog; 10 uncertain or morphologically ambiguous rows are held and
three Juanga cells are blank. Eight neighboring language columns contain
302 printed forms and 34 blanks; each row records their presence/blank
state, with a 21-cell control sample, but none is installed. Dalton's
1866 list is a probable same-record predecessor and not counted as a
second elicitation. Samuells 1856 differs, with zero exact selected-form
overlap. Seven Haldar forms match later Juang source strings without
proving independent elicitation. The source spelling is preserved; no
speaker/site coordinate or graph relation is invented. Append order
104, bibliography key `dalton1872haldarjuang`; three source-focused tests,
metadata and 31/31 scoped parses pass. Full compiled/browser gates
remain deferred.

### Kapp Alu Kurumba numeral source-side installation

The CC-BY archived page credits Dieter B. Kapp's 1995 Alu Kurumba
table. All 40 numbered cells are complete and yield 40 source rows;
there are no blank or control cells. The HTML cell markup and a 40-cell
audit preserve the original forms, and a seeded 20-cell manual review
found zero material errors. The source maps to existing canonical
`AluKurumba` (`aluk1238`) without inventing a site or dialect. Six
display-form/gloss pairs overlap seven older independently cited rows;
34 pairs are newly represented. Decomposed `ë` is NFC-normalized to
`ë` in the installed CSV while the audit keeps the original sequence;
its quality and printed superscript nasal `ⁿ` remain uninterpreted.
The source's broad Kannada comparison is not a per-form loan claim.
Append order 105, bibliography key `kapp1995alukurumba`; five focused
tests, the 25-test source/sound/dialect set, metadata and 40/40 scoped
parses pass. Full compiled/browser gates remain deferred.

### Source candidates held before installation

Grierson's 1919 LSI Kachchhi comparative vocabulary (VIII(I), pp.
214–231) has a plausible existing `kcch`/`kach1277` mapping, but the
primary scan could not be recovered legibly enough to align the target
column with English prompts and controls. No cell count or source row is
claimed; the access and transcription hold is recorded in
`data/other/forms/raw_data/kachchhi_lsi1919_screening.md`.

The archived Mallikarjun/Bhat Yerava tables appear to be one underlying
1993 list: 36 of 40 printed cells match and four differ only in spacing.
The original monograph says it is based on Paniya Yerava speech while the
archive heading says Ravula without specifying the list's subcommunity.
Neither `Paniya` nor `Ravula` is assigned pending the original numeral
section; no CSV, YAML, bibliography entry or new lect is installed. See
`data/other/forms/raw_data/mallikarjun_yerava_1993_hold/README.md`.

### Bailey Kāgānī bounded source-side installation

Bailey's public-domain 1920 vocabulary, printed p. 106 left column,
has 32 alphabetical headword entries in the bounded `able`–`cloak`
span. Ten secure literal readings yield ten rows; 21 entries are held
for unresolved typography and the multiword connective at line 14 is
held as nonlexical. The PDF text layer was only a sequencing cross-check
against enlarged page images. A seeded 20-unit image review found zero
material accepted errors. Bailey's p. 87 and Glottolog support new
canonical `Northern Hindko` (`nort2662`) with a source-qualified Kāgānī
dialect; the existing `awan`/Avankari record is Southern Hindko and is
not reused. The Kāgān Valley is broad provenance, so coordinates remain
blank. No phonemic values, etymologies or exact same-print overlap with
other surveys are inferred. Append order 100, bibliography key
`bailey1920kagani`; source/sound/dialect tests, metadata and 10/10
scoped parses pass. Full compiled/browser gates remain deferred.

The older registry already carries `nort2662` on `HKAT-hno` under
`awan` and three LSI dialect rows under `L`. These pre-existing
parent/code combinations need separate source-specific review in a
future identity audit; the Kāgānī package does not silently remap them.

At this checkpoint, 66 dated source CSVs hold 7,040 rows. Their 66
dated-source test modules pass 307 tests; `source_meta.py` validates 269
settings files and 253 citation keys. Companion YAML coverage,
append-order uniqueness, 15-column row widths, required fields, NFC
forms and source-local entry-key uniqueness all check cleanly. The next
source registrations will require a fresh aggregate check.

### Sounderaraj Dandami Maria numeral source-side installation

The 2019 CC-BY archive preserves Joseph and Omania Sounderaraj's
1995 Bison-Horn Madiya table. Forty complete numbered prompts produce
42 rows because 20 and 100 each have two independently printed answers.
All cells and answer splits have stable keys and raw-markup audit records;
a seeded 20-cell comparison found zero material errors. The frozen page,
not a later live page with changed heading and a new loan annotation,
controls this package. Broad Indo-Aryan comparison prose supplies no
form-specific borrowing claim; no graph edge or loan tag is inferred.
The precise spoken L1 maps to new canonical `Dandami Maria`
(`dand1238`) with blank coordinates. Nine form/gloss pairs overlap
independently cited 1994 Beine Bison Horn sites under generic `Gondi`;
33 pairs are display-new, and the old sites are untouched pending a
future compiled identity audit. Append order 107, bibliography key
`sounderaraj1995dandamimaria`; five focused tests, the 25-test
source/sound/dialect set, metadata and 42/42 scoped parses pass. Full
compiled/browser gates remain deferred.

### Fawcett Saora kinship table held before installation

HathiTrust's public-domain original volume verifies Fawcett's 1888
article and its kinship table on printed pp. 226–227. The table heads
every relation with possessive “my” and extends to recursive multiword
relations; stripping a free lemma would add unsupported morphological
analysis. Fawcett says an unnamed Oriya speaker who knew Saora first
wrote the tables, then he reviewed pronunciation with an unnamed Saora
speaker using a Missionary alphabet. No form from this table is
installed. The exact scan links and attribution are preserved in
`data/other/forms/raw_data/fawcett_saora_1888/README.md`; search for
direct lexical material in the article continues.

### Grierson Haṇḍūrī numeral source-side installation

The public-domain LSI IX(IV) printed p. 628 has 13 Haṇḍūrī numeral
cells in the bounded numbered section. All 13 were visually reviewed
against enlarged scans: ten secure readings yield ten rows, while five,
nine and fifty remain typography holds. Adjacent Kiūthalī controls and
pronoun/possessive items 14–25 are excluded. Source `Original` retains
secure spelling and capitals; the literal sound profile lowercases
capitals only for parsing, without inferring phonemic values. The source
maps to new canonical `Hinduri` (`hind1267`, `hii`) with a source-qualified
Haṇḍūrī dialect and no invented coordinates. A later classification as
Kiūthalī is documented as a caveat, not a reason to silently remap
these LSI cells. Append order 106, bibliography key
`grierson1916handuri`; 13/13 source cells audited, focused tests and
source metadata pass. Full compiled/browser gates remain deferred.

### Vaz Hill Maria numeral source-side installation

The 2019 CC-BY archive preserves Christopher Vaz's 2011 Hill Madia
(Bhamragad) numeral table. Its 40 target cells yield 40 rows; a second
40-cell table attributed to Natarajan 1985 is audited and excluded as a
separate witness. A seeded 20-cell review found zero material errors.
Inline HTML spans at five numbers are joined as printed single forms;
the unexplained asterisk on seven and arithmetic parentheticals are
retained in the audit, not lexical forms. No per-form loan, component or
etymology is inferred from broad comparison prose. The target maps to
new canonical `Maria (India)` (`mari1414`) and a source-qualified Hill
Madia (Bhamragad area) dialect, with blank point coordinates.
Older `maria` dialect and Beine-site rows remain untouched; zero of the
40 exact form/gloss pairs overlaps their source rows, but the parallel
identity needs a later compiled audit. Append order 109, bibliography
key `vaz2011hillmaria`; the source's 80 cells are accounted for. The
Hill/dialect focused tests (11/11) and source metadata pass. Full compiled/browser gates
remain deferred.

### Prendergast Savara bounded vocabulary source-side installation

The public-domain 1881 JRAS printing carries M. H. Prendergast's 1880
Savara vocabulary, forwarded through Cain and introduced by Cust. The
complete left English–Savara column of printed p. 426 has 59 prompt
cells. Visual and per-record audits select 50 directly glossed forms and
hold nine: four compound English prompts, one possibly derived response,
two unresolved glyph readings, one two-alternative reading, and one
unresolved final macron. The right column and pp. 427–428 remain outside
this bounded pilot, not silent blanks. All 50 rows map to existing `so`
with a source-qualified Savara dialect and blank coordinates; Prendergast's
Vizagapatam post does not establish the elicitation point. The literal
profile preserves macrons and underdots but assigns no phonetic value to
`ch`, `ng`, or `sh`. Three exact normalized form matches and one further
diacritic-insensitive match occur among 2,171 older Sora rows; no
Prendergast citation reuse was found. Append order 108, bibliography key
`prendergast1881savara`; focused tests, source metadata and 50/50 scoped
parses pass. Full compiled/browser gates remain deferred.

At the final source-stage checkpoint, 70 dated CSV packages hold 7,182
rows. Every package has a YAML companion; all 70 append orders and all
7,182 entry keys are unique. Row width, required fields and NFC forms
have zero structural errors. The combined dated-source suite passes 323
tests across 70 modules; generic sound-profile and dialect tests pass
21/21. `source_meta.py` validates 273 settings files and 257 citation
keys. The Pahari-Pothwari long aspirated affricate `čʰˑ` now has an
explicit sequence-aware mapping (`ccʰ`), its two source forms pass
focused conversion, and `profile_policy.py check` reports zero
violations after the old suppression was removed. All required complete
CLDF build, compiled reference/identity/graph checks, full test suite
and browser-database QA remain deferred under the user's explicit
no-build and no-remote instructions.

### Completeness policy for continuing ingestion

The user has now directed that a source be ingested completely whenever
it is ingested. Earlier packages explicitly labelled as bounded pilots
or first-page/first-column extracts remain **unfinished source coverage**,
even when their bounded audits and focused tests pass. The preceding
70-package count is a count of installed source-stage packages, not a
claim that all underlying vocabularies or books have been exhausted.
Further work must inventory the complete relevant lexical section for
each selected source lect, account for every cell and exclusion, and
reconcile any earlier pilot rows before calling that source ingested.
Grammar, paradigms, narratives and other lects may be excluded only with
an explicit section-level rationale; arbitrary page or column boundaries
are not a completion boundary. Existing source README files identify
remaining pages/columns for several unfinished pilots, including
Prendergast Savara, Grierson Kaikadi, Bailey's first-page vocabularies,
and Norton Korku. No full database build has been requested.

For the unfinished Cust/Norton 1884 Korku pilot, the pinned public-domain
scan has now been verified and an eight-page OCR sequencing extract for
the complete forward English–Kor vocabulary (printed pp. 165–172) is
checked in beside its importer. OCR joins columns and cannot authorize
forms; the full page-image inventory and reverse-index reconciliation
remain open. No extra Norton rows were installed from this extract.

### Erza Pattapu complete numeral-table source-stage installation

The CC-BY 2019 archive preserves Erza's 2015 Pattapu numeral page via
George Edward and the Bible Society of India. Its complete 20-row,
two-column table contains 42 separately numbered prompts and 43 forms
because item 200 prints two complete alternatives. Every source cell,
HTML answer, caption and attribution is accounted for; there are no
blank or held items. A seeded 20-item audit found zero material errors.
The 2015 witness is independent of the 2013 Rebbavarapu list represented
in older compiled Pattapu rows. The source has no exact elicitation point,
so no new dialect or coordinates are assigned. Its unusual superscript
`ⁱ` and source `ə` are preserved in `Original`/`Phonemic`; display
conversion remains a documented linguistic review point. No per-form
loan or variant graph edge is inferred. Append order 111, bibliography
key `erza2015pattapu`; 26 focused/source/sound/dialect tests and metadata
(275 settings files, 259 citation keys) pass. Full compiled/browser gates
remain deferred.

### Bailey Poguli complete numbered-vocabulary source stage

Bailey's public-domain 1908 Poguli section has one complete 100-item
numbered vocabulary on printed pp. 58–59. Every item is inventoried:
89 accepted cells yield 94 rows because five print two answers; four
typography cases, six stem-dependent grammatical fragments and one
printed blank are audit-only. The seeded 20-item page-image comparison
found no accepted-form material error. Existing Jambu `pog` represents
Poguli under Kashmiri at the base-language level; Glottolog's
`pogu1238` identifies the Poguli dialect, so this source uses the
existing base plus source-qualified dialect and does not create another
language. Append order 110, bibliography key `bailey1908poguli`;
four focused tests, 94/94 scoped parses, profile policy and metadata
pass. The complete numbered section is source-stage installed, while
the full compiled/browser gates remain deferred.

### Prendergast Savara complete vocabulary source stage

The complete Savara vocabulary in the 1881 *Journal of the Royal Asiatic
Society* has 266 printed prompt/response cells on pp. 426–428, including
both columns of each page. Four printed alternatives yield 270 source
rows; two uncertain historical glyphs remain explicitly tagged. Printed
p. 425 is Cust's attribution, and the preceding Koi article is a separate
source. The pinned scan's physical PDF locators are pp. 472–474, correcting
the old pilot locator. A seeded 24-cell visual review found no material
errors, 24 focused/profile/dialect checks and source metadata passed, and
the new rows were screened against 2,171 older Sora forms. The full
database build, compiled graph/reference checks, and browser QA remain
deferred under the user's no-build/no-remote instructions.

### Rambani complete numbered table and Haijong boundary correction

Bailey's 1908 Rambani numbered lexical table spans all 100 items on
printed pp. 48–49. The prior 1–36 pilot now has the remaining cells
reconciled: 61 accepted cells yield 62 rows, with 27 typography holds,
11 stem-fragment holds, and one same-print exclusion. A seeded 20-cell
two-page visual audit had zero material errors; four focused tests,
62/62 scoped parsing, source metadata, and the sound profile passed.
Bailey's surrounding grammar and sentences are separate sections.

The Grierson Haijong package was mislabeled partial in the batch ledger.
Its 25 cells on printed p. 354 are the complete Haijong column. Printed
pp. 355–365 continue the numbered prompts for other Bengali lects only;
there are no further Haijong cells. The ledger now records the complete
source-lect column rather than demanding unattested continuation rows.
Neither package has undergone the deferred full compiled/browser gates.

### Bailey Kotguru complete lexical and numeral section

The existing printed-p. 31 left-column Kotguru pilot represented only one
part of Bailey's source-lect material. The complete lexical, cardinal,
ordinal/fraction, and numeral-note section spans printed pp. 30–33 and has
201 cells. Every cell is now audited: 107 accepted cells generate 112 rows;
77 typography cases, 16 stem/fragment cases, and one complex alternative
remain held with typed reasons. The p. 33 sentences are a distinct excluded
section. Four focused tests, 112/112 scoped parsing and source metadata pass.
The full compiled/browser gates remain deferred.

### Korvi full-column source-stage completion

The Korvi (Belgaum) LSI IV install now covers all 241 numbered printed
cells on the alternating pp. 646–678. The source-local audit records 153
accepted cells producing 156 rows, 66 transcription holds, and 22 excluded
full-sentence cells. Item 29 contributes only its clear second answer;
its first alternative remains held. Items 86, 87, and 93 have two accepted
answers each. All nine original-scan source-column crops and OCR sequencing
aids are preserved. The full importer retains all 39 old pilot keys and
corrects five readings (40, 54, 62, 69, 71); 69 is tagged uncertain for
its soft final lateral diacritic. An independent stratified 24-cell scan
review across eight pages found zero material errors. The 156/156 scoped
parse, 28 focused/profile/dialect tests, and source metadata pass. Full
compiled graph/reference and browser checks remain deferred by the user's
no-build/no-remote instruction.

### Grierson Kaikadi full-column source-stage completion

The companion Kaikadi column of LSI IV now covers all 241 numbered prompts
on printed pp. 646–678. The complete audit identifies 162 reviewed answer
cells yielding 164 rows, 40 transcription holds, 17 printed blanks, and
22 excluded full-sentence prompts. Existing 32–79 pilot keys are preserved.
The 1906 scan controls the readings; the 1928 digitization is comparison
evidence only. A seeded 23-cell audit across the extended pages found zero
material errors, 28 focused/sound/dialect tests passed, and source metadata
validated. Full compiled graph/reference and browser checks remain deferred.

### Bailey Outer Siraji complete lexical and numeral section

The complete Outer Siraji source-lect section on printed pp. 41–43 has
178 cells: 123 lexical lines, 46 cardinals, and nine ordinals. The audit
records 99 accepted cells producing 100 rows, 65 typography holds, and
14 stem-dependent fragment holds. The old p. 42 left-column keys remain
stable. Direct page-image review corrected the accepted p. 41 water reading
to printed `pāṇī`. Four focused tests, 100/100 scoped parses and profile
checks, and source metadata passed. Full compiled/browser gates remain
deferred.

### Dalton–Haldar Juanga complete target-source stage

The 1872 Haldar-compiled comparison table was broader than the p. 236
pilot: its Juanga target column continues through printed pp. 235–241,
and a separate Juang vocabulary fills the lower p. 241 and p. 242. The
complete audit now covers 361 target cells: 189 accepted cells yield 196
forms, 19 readings are held with typed reasons, and 153 target cells are
printed blank. Eight adjacent comparison-language columns remain controls.
The former p. 236 pilot's 31 entry keys are preserved. An independent
original-scan review of all 30 accepted cells in the comparative extension
corrected p. 235 row 15 to `ainyá` and finished with zero material errors;
a separate-list sample checked 20 cells with zero errors. The 196/196
scoped parse, 26 focused/profile/dialect tests, and source metadata pass.
Full compiled graph/reference and browser gates are deferred under the
user's no-build/no-remote instruction.


### 2026-09-26 local continuation after disk cleanup

- Local-only, Jambu-directory-only changes; no ASJP, remote execution, database/full pipeline build, commit or publication. Disk remains approximately10GiB free after3.05GiB regenerable-image cleanup.
- Rohru full255-cell glossary now yields94 forms;172 typography holds and3cross-reference-only exclusions. Independent seeded20 across all4pages:0material errors. Source-specific/profile/dialect checks passed; ledger source stage closed.
- Norton Korku full article lexical scope now956 units (477forward cells,427reverse heads,12numeral cells,40introduction/notes records) yields1,032forms. Nine holds and15sentence/control exclusions remain explicitly audited. Reverse OCR omitted4heads. Three failed independent samples exposed repeated macron errors; all427reverse heads were rechecked at450dpi, followed by a fresh20-entry sample across all6pages with0material errors. Earlier failed evidence retained. Source-stage ledger closed after focused checks; compiled gates deferred.
- Baghi full245-cell inventory yields96forms from88accepted cells. Independent first20 exposed a breve/macron error in a pilot reading; full pilot and accepted-length-mark review corrected4old readings under stable keys. Fresh20 across4pages:0material errors and installed keyed forms agree. Source owner finalizing source-stage documentation.
- Hahn Asur expanded to full-article inventory; first independent20 found2lost-macron errors and1unresolved reading. Corrections and broader transcription review precede a fresh independent audit. No completion claim yet.
- Hockings Badaga additional image audit explicitly waived by user; ledger and source note record waiver without treating it as passed or clearing provisional OCR tags.

All counts describe source inputs, not a newly rebuilt database. Full compiled graph, identity, reference-output and browser verification remain deferred.

## 2026-09-26 — Cust/Norton Korku full source stage

Expanded the 49-form first-page pilot to 1,032 forms from 956 audited units:
477 forward cells, 427 reverse heads, 12 numeral cells, and 40 introductory or
grammatical-note records. Nine unresolved records remain held; 15 sentence/control
examples are explicitly excluded. All original pilot keys survive. Exact reverse
form/gloss repetitions add citations; distinct spellings and senses retain keys.

Three failed independent reverse samples exposed lost or imported macrons. Every
one of the 427 reverse heads was reread at 450 dpi in column crops; a fresh
independent sample of 20 across all six reverse pages then passed with zero errors
(seed 2026092607). Failed audits and the comprehensive repair report are retained.
Focused importer/profile/dialect checks passed 24 tests; the final scoped importer
checks passed 3 tests and all 1,032 forms convert without errors. Source settings
validation passed. The full pipeline, compiled graph/identity/reference checks,
full suite, and browser refresh remain deferred under the user's no-build and
no-remote instructions. Regenerable review images were removed after audit.


### 2026-09-26 correction: inventory coverage is not recovered ingestion

The prior checkpoint overstated source-stage completion for Bailey Rohru,
Baghi and North Jubbal: their full cell inventories still excluded readable
words behind generic typography placeholders. Those packages are reopened as
confirmed_partial for exhaustive literal transcription recovery. Bilaspuri
and Kāgānī share the same generic-hold pattern (96 and156 placeholders) and
are also reopened pending evidence sufficient to validate their omissions.
This does not invalidate the specifically sampled accepted readings.

North Jubbal's independent20accepted-cell review across pp185–188 passed,
and its98installed rows include corrections to six old pilot readings; this
is accepted-output evidence only. The IA agent is recovering Rampur using
literal source symbols and will revisit the held Bailey material.

Gorum now has a full5824-chunk structural census and independent81-case
duplicate-ID reconciliation.24cases are exact raw repeats; other cases include
commentary-only differences, embedded unnumbered parent heads, actual ID
collisions and lexical extensions. These records replace the previous blanket
exclusion plan; full extraction remains unfinished and no new Gorum rows have
been installed. Asur remains under independent fresh-sample review following
further diacritic corrections; staged output is not declared complete.


### Rohru full literal recovery (2026-09-26)

Replaced broad typography exclusions with full450dpi transcription: 255cells → 250accepted / 309forms, two individually described vowel-stack holds and three English-only crossreferences. All94previouskeys retained. Independent audit1 found one length-mark error; freshaudit2 passed0/20 after correction, with nasal stacks/underlining/grammar edges reviewed.26focused profile/dialect/importer tests and1audit-hash test passed; scopedparser309/309, no errors;source_meta valid. Full compiled pipeline/fullsuite/database/browser remain deferred. This supersedes the earlier94-row partial extraction; it does not claim compiled validation.

### 2026-09-26 Turi full chapter source-stage completion

Replaced three-row pilot with291 forms from352 prose/specimen units across printed128–134. Four named historical sites registered,57same-site exact reuses retain every locator,4contextual controls excluded,1typed ambiguous sons reading retained. Original3keys survive. Independent20-entry audit0errors plus finaleditorial addendum;6focusedtests and291-row scopedparse pass. FullCLDF/database/compiledgraph/reference/fullsuite/browser gates deferred by explicit user instruction. No remotes used.

### Roy Birhor whole-appendix source-stage recovery (2026-09-26)

Replaced46-rowpilot with1019forms acrossAppendixI559–591:875vocabularyheads+114introattestations+2nestedsubentries,9exactreuses,38variants,1physicallydamagedrainhold. All46legacykeys retained. Fullindependent rereading reconciled18vocabulary and5introfindings after repeated sample failures;fresh20sample passed0errors.15focusedtests,1019sourceparse,source-localvariantgraph andmetadata pass. Comparisons moved toNotes; no inferredcognate/borrowededges. FullDB/CLDF/compiledgraph/reference/fullsuite/browsergates deferred under explicituser instruction. No remote work or publication.
