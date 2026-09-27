# CUJ Asur dictionary: installed source inputs; integration pending

Status on 2026-09-21: **source inputs installed; ingestion incomplete**.
The complete compiled build, full suite, formatted-reference/durable-ID/graph
checks and required browser QA remain open. This is not a completion claim.

- 2,005 physical source entries → 2,106 installed sense rows.
- One undecoded native head (CID127) excluded and fully retained in the audit.
- 185 variant links; one explicit compound with two ordered component links.
- 16 unresolved relationships: 15 target/sense ambiguities plus one missing
  printed target (`सुकुल`). Affected rows carry typed audit reasons and uncertain.
- 290 native-only rows; no fabricated romanization. 352 blank glosses reflect
  source placeholders, reference-only records, or unexpanded grammatical codes.
- 47 uncertain rows include native-source anomalies, two superscript-stop
  conventions, unresolved grammar labels/codes and unresolved variant links.
- 14 source tests and six dialect tests pass; source settings validate; the
  profile passes scoped policy and complete NFC/NFD coverage checks.
- Actual parse_file retains all 2,106 rows with no conversion errors and keeps
  Original/Phonemic and source-local graph keys separate.
- Fresh rich-output seed 20260926: **0/20 material errors**, plus rare-symbol,
  embedded-subentry and reference-scope checks. See visual-review.json.
- Compiled-survival test explicitly FAILS: 0/2,106 source keys exist in current
  CLDF. The full build has not run; no app entries can yet be demonstrated.

Canonical inputs are `../../20260921-cuj-asur.csv` and the adjacent YAML;
`20260921-cuj-asur-audit.json` reconciles every physical record to installed rows.
`manifest.json` pins the source; `proposal-review.json` records current checks
(the filename is retained from the proposal stage). Rebuild source inputs with:

```sh
.venv/bin/python data/other/forms/raw_data/cuj_asur_2020/import_source.py --install
```

The dictionary addendum applies. OCR and external CLDF/API addenda are
inapplicable; the source text layer was decoded directly. No control language,
printed etymon ID, new base language, or uniform named dialect requires mapping.
No coordinates were inferred. No donor edges were guessed from language labels.
No release, commit, push or browser database rebuild has been performed.

## Why this source

Asuri has only 29 existing input rows: 19 from Pinnow and ten from Zoller.
The Centre for Endangered Languages, Central University of Jharkhand, links an
Asur–Hindi–English dictionary containing over 2,000 entries. It offers much
greater coverage of an underrepresented Munda language than adding more
Ho/Mundari/Santali dialect lists.

The 110-page PDF has dictionary text on PDF pp.6–110 / printed pp.1–105.
The university publication page dates the release 2020-08-16; the PDF metadata
says 2020-10-22, its copyright notice says 2019, and the PDF explicitly calls
itself a draft. Preserve these distinctions rather than calling it a final 2020
edition. `manifest.json` pins the exact URL, SHA-256, editors and provenance.
No open redistribution license was found; the PDF itself is not checked in.
The eventual installed output concerns lexical facts; extended illustrative
prose is not proposed for republication as dictionary entries.

The official downloads page was inspected and is empty. The official mobile
applications page links `org.celcuj.asurdictionary.asur` on Google Play, but no
accessible source-data export was found. A third-party app listing adds no
authoritative lexical evidence. The phonology paper linked by the same centre
was downloaded and pinned for transcription review: Zoya Khalid, *A Phonological
Sketch of Asur*, Language in India 20(8), 2020, pp.252–268.

## Current artifacts and measured counts

- `extract.py`: page/font/column-aware source extraction; no OCR or installation.
- `entries.jsonl`: 2,005 physical entry blocks with immutable page/column/top
  keys, exact source tokens, fonts, bounding boxes and native glyph evidence.
- `parse.py`: a **candidate**, non-installing semantic parser.
- `candidates.jsonl`: one candidate audit record for every physical block.
- 1,805 candidate articles, 200 explicit cross-reference entries, 2,107 candidate
  sense slots, and 1,714 entries with a printed IPA headword.
- 291 entries have no printed IPA head; 345 have at least one missing English
  gloss in the candidate parse. These flags overlap and are **not exclusion
  decisions**. Some are intentional source placeholders or cross-references;
  each needs classification before final installation.
- Nine native-head review flags: eight printed dotted-circle sequences and one
  unresolved CID127 on p.102, column 1, top 324.8. All remain visible.
- `audit.py` / `audit.json` accounts for all 2,005 records: 1,661 with English
  definitions, 200 explicit cross-references, 84 bare heads, 50 transcribed
  undefined heads, nine annotation-only records and one botanical identification
  without English prose. These are mechanical classifications, not exclusions.
- 541 parsed reference targets: 540 exact unique matches and one unmatched
  (`सुकुल`). The rich importer accepts only reviewed relation types and unambiguous
  target senses; the final compiled graph remains unverified. Whole/part/general/specific labels
  must not be converted into etymological ancestry. Nested references retain
  their parenthesis depth and are explicitly marked as describing a referenced
  word, not necessarily the entry. Six `unspec. comp. form of` targets are
  recorded as complex-form labels, not asserted morphological components.

## Extraction errors identified and fixed

The PDF uses a true text layer but has defective Devanagari ToUnicode mappings.
Native bold pre-base-i glyphs have width 291/1000 em and map to a whole syllable
such as रि, followed by the actual consonant glyph. The extractor decodes the
mark and restores logical order. This corrects e.g. अखरिर → अखरि mechanically.
Glyph drawing order, not horizontal coordinate order, also matters: x-sort
produces अगंुर instead of अंगुर. Source drawing order is now retained.

The Type1 font's `/Encoding` names CID52 as थ but its ToUnicode maps it to र्थ;
the extractor restores the atomic value. CID53 is the separately drawn repha,
visually checked in all eight affected headwords. CID127 remains unresolved.
CID117 is /uni25CC (dotted circle), despite ToUnicode mapping it to ो.
Eight source heads contain this printed mark, which is retained and flagged.
A fresh sample also exposed double reordering of already-correct ि; repair now
uses a sentinel only for the defective pre-base glyph, including word breaks.
Do not interpret these font repairs as editorial or phonological emendations.

Alphabet headings span the entire page. Reading every page's whole left column
before its whole right column wrongly attaches the previous alphabet's last
senses to the next alphabet's entries. `ordered_words` divides the page into
bands at every 14-point centered heading and reads each band left–right.
It also retains the last body lines through y=565, excluding larger footer
page numbers. The initial 1,982-block count was incomplete; the corrected count
is 2,005. All correction classes have regression tests.

Examples of checked structural repairs:

- p.39 `jom`: senses 1, 2 and the blank sense 3 stay together across columns.
- p.39 `jʰaiɽ`: “continuous rain for more than a week” continues within the new
  alphabet band; it no longer receives `jom`'s third sense.
- p.60 `pai naːg`: the bottom-line gloss survives and does not absorb `ne` entries
  from the preceding alphabet's right column.
- Headword homonym numbers remain separate from IPA.
- Usage sentences and `spec:` references are not appended to English glosses.

## Reproduction and checks

From the data repository, using the PDF at the URL in the manifest:

```sh
.venv/bin/python data/other/forms/raw_data/cuj_asur_2020/extract.py PATH_TO_PDF --output /tmp/asur-entries.jsonl
.venv/bin/python data/other/forms/raw_data/cuj_asur_2020/parse.py --output /tmp/asur-candidates.jsonl
.venv/bin/python -m pytest -q tests/test_cuj_asur_extraction.py
```

Fourteen source tests pass, including source-reference, botanical-identification
and complete audit-accounting regressions. A fresh seed-20260924 sample found
0/20 material errors in native head, IPA, English gloss and sense boundaries;
`visual-review.json` records the exact scope and keys. This is not yet the
complete rich-import audit: relations, tags, annotations and normalization
remain under review. Earlier seeds exposed i-reordering and dotted-circle
problems, both corrected before this fresh sample.

## Import decisions and verification

`import_source.py` supports proposal output (`--output-dir`) and canonical
installation (`--install`). It preserves homographs/senses using immutable
physical source keys and sense child keys. Parenthesized references following
a target belong to that target, even at parenthesis depth one; they must not
be attached to the current headword. Explicit forward variant direction takes
precedence over reverse inference from a variant list. Cycles are rejected.
Full-sized Annapurna Bold sense references remain distinct from small native
homonym numbers. Missing and ambiguous relations stay unlinked and audited.

Four embedded subentries (pp.44, 65, 74) have separate alphabetic records. Three
Hindi-only definitions are explicitly transcribed/translated in the importer;
both locators survive. The fourth has no definition. `comp.` no longer leaks
into “soil”. Only explicit `comp. of` creates ordered components; `unspec. comp.
form of` is preserved as source commentary. Scientific identifications retain
printed spellings and attach to their source sense. Unknown grammatical codes
remain in Notes/audit with uncertainty rather than masquerading as lexical gloss.

The [transcription policy](transcription-policy.md) documents c/j affricates,
y glide, retroflex conversion, length/nasalization preservation, superscript-stop
uncertainty and native-script preservation. The profile is routed in YAML.
Bibliography is registered under `cuj2020asur`; formatted references await build.

Run source checks from the data repository:

```sh
.venv/bin/python -m pytest -q tests/test_cuj_asur_extraction.py -k 'not compiled_source_survival'
.venv/bin/python data/other/forms/raw_data/cuj_asur_2020/audit_sample.py --pdf PATH_TO_PDF --output OUTPUT --seed 20260926
```

The full survival test is intentionally separate while CLDF is stale. Do not
call ingestion complete until the consolidated full pipeline, full tests,
reference formatting, durable-ID/graph/concept/deduplication checks and required
browser QA pass. Required heavy gates remain deferred under the laptop resource
policy; no suitable authorized runner for these uncommitted inputs is available.
