# Bailey North Jubbal / Barari: approved whole-source files

Installed on 2026-09-27: **837 rows and 688 audit units**, preserving all **330 previous glossary keys** and literal forms. This completes the reviewed source-file stage; full ingestion is not claimed while the database/build/browser gates remain deferred.

The source scope covers the complete chapter pp. 172–188, the explicitly North/South Jubbal suffix on p. 143, applicable shared notation on p. 114, and the tentative currency discussion on p. 167. Independent review covers414 grammar units,22 sentences yielding24 whole expressions,8 notes, and the shared suffix. Forty-three reviewed glossary metadata corrections and two sentence vowel-length corrections are included. Three rows retain typed source-analysis uncertainty, including Bailey's suggested note2 reading and the two currency-related records. Other-lect comparisons and contextual controls remain in the audit; no inferred stems, donor links or word-level translations are added.

Fresh independent whole-output review passed20/20 units (21 emitted rows). Isolated regeneration reproduces the approved CSV, audit and profile byte-for-byte. Nineteen focused tests pass; the actual canonical parser preserves all837 ordered keys, Originals, Notes, citations and tags without conversion errors. The preservation profile supports NFC/NFD, both raised-breath conventions, and source marks; only established display conversions apply. Canonical Barari identity, the existing quality-C regional coordinate caveat, legacy-ID policy and append order remain unchanged.

Use `install_whole_source.py --install` for guarded regeneration/installation. The original `import_source.py` is hash-pinned historical evidence: **never run it against the live installation**. The verifier invokes it only inside an isolated sandbox. Prior canonical CSV, audit, YAML and profile are preserved under `historical-glossary-installation/`; earlier previews, first readings and independent reviews remain immutable.

Final evidence: `root-whole-output-audit-20260927-pass1.json`, `whole-reviewed-regeneration-validation.json`, `whole-installation-result-20260927.json`, and `whole-installed-validation-20260927.json`. The expanded bibliography entry and YAML route identify the approved importer and scope.

Deferred: full CLDF/database build, compiled durable identity/graph/reference checks, full suite and browser/app QA. No DB build, remote execution, commit or release was performed. See `source_checklists/20260927-bailey-north-jubbal-whole.md` for the gate handoff.

## Historical glossary-only stage (superseded scope)


T. Grahame Bailey, *Linguistic Studies from the Himalayas* (London: Royal Asiatic Society, 1920), “The Dialects of Jubbal State,” labels this North Jubbal or Barari. It maps to existing canonical `Barari`, distinct from the book's Koci chapters. The LSI IX(IV) pp. 599–601 Barari account supplies a prose specimen and geographic context, not this glossary. Existing Zoller 2023 Barari entries cite Bailey pp. 185–186; these are derivative attestations of the same printed evidence, not independent elicitation. No narrower consultant/site or new historical coordinate is asserted.

The public-domain [Internet Archive edition](https://archive.org/details/linguisticstudie00bailrich) has 310 PDF pages, SHA256 `7a1577a2d2a24eca8b95e9b1ed700ef01f716d1069e4468f36a673611d093b39`. The ignored local cache is `tmp/pdfs/bailey1920/bailey1920.pdf`; `--check-pdf` verifies it. Source-side regeneration uses checked-in inventories and does not need the PDF.

## Full scope and unresolved readings

Every headword cell on **pp. 185–188 (PDF 211–214)** is accounted, from *above* to the final *you/your* cell. There are **243 cells**: 71 on p. 185, 68 on p. 186 (34/34 columns), 71 on p. 187 (36/35), and 33 on p. 188 (19/14). Wrapped definitions and separately glossed answers remain one source accounting cell. Grammar, paradigms, prose specimens, and South Jubbal are explicit exclusions; no glossary page or column is omitted.

The authoritative `full-transcription.tsv` contains every literal reading. Older `transcription.tsv`, `completion.tsv`, raw text-layer extracts and pilot audits remain historical evidence. **242 accepted cells produce 330 forms**, with one English cross-reference (*above*) and no unresolved readings. All 98 previously installed keys survive and are recorded in `legacy-entry-keys.json`; corrections preserve those keys.

The 15-column installed CSV retains printed-page/item keys and `:answer2` etc. for additional printed answers, with aligned lexical glosses and grammar. Source noun/verb/adverb, gender, transitivity, causative, relative/interrogative and instrumental labels are structured. Correlative prose stays in the audit rather than a lexical gloss. No phonemic interpretation, etymology, borrowing or directional variant relationship is inferred; all 330 are unlinked attestations.

The profile preserves source breves, macrons, nasal stacks, underdots and underlining literally. Underlined sh uses U+0332 beneath each letter; the printed raised breath mark is represented by U+2018 ‘ without phonological interpretation. House conversions `w → v` and `ṅ → ŋ` affect display only; Original retains source spelling. Source word boundaries remain intact.

Full recovery's initial independent 20-cell review found five affected cells: omitted nasal tilde, an absent underdot, an added underdot, a mistaken vowel/mark, and a grammatical label in a lexical gloss. The ensuing source-wide second image reading corrected 24 form cells, recorded in `literal-corrections-20260926.json`. It distinguished daughter/girl ṅ from ṇ, plain l from ḷ in say/speak and maize, and real vowel lengths and nasal stacks. Two disputed readings were independently adjudicated: call `budno` (the source explicitly says not -no) and thou `tū` (u under a single macron; the reviewer's earlier ī interpretation was retracted). Fresh stratified seed 2026092652 excluded the first 20 and passed **0/20**, with hashes matching the final transcription and installed CSV. Historical failed and pilot reports remain available.

The earlier seven focused glossary tests passed, including exact accounting/regeneration, all 98 prior keys, literal error-class regressions, grammatical alignment, source-wide profile coverage, fresh audit hashes, and **330/330 scoped parse** without conversion errors while preserving Original. This evidence applies only to the glossary installation; whole-source recovery remains open as described above. Full CLDF generation, compiled graph/durable-ID checks, regenerated references and full suite remain deferred by the user's instruction. Browser database and representative app QA require an explicit refresh request. No remote work, commits or publication were performed.

```sh
.venv/bin/python data/other/forms/raw_data/bailey_north_jubbal_1920/install_whole_source.py --install
.venv/bin/python -m pytest -q tests/test_bailey_north_jubbal_1920.py
.venv/bin/python source_meta.py
```
