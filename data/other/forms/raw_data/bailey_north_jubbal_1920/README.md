> **Whole-source scope reopened (2026-09-26):** Existing installation covers only the glossary. Preglossary grammar and complete translated responses remain to be recovered; see `whole-source-reopening-20260926.json`.

# Bailey 1920 North Jubbal (Barari) full glossary

T. Grahame Bailey, *Linguistic Studies from the Himalayas* (London: Royal Asiatic Society, 1920), “The Dialects of Jubbal State,” labels this North Jubbal or Barari. It maps to existing canonical `Barari`, distinct from the book's Koci chapters. The LSI IX(IV) pp. 599–601 Barari account supplies a prose specimen and geographic context, not this glossary. Existing Zoller 2023 Barari entries cite Bailey pp. 185–186; these are derivative attestations of the same printed evidence, not independent elicitation. No narrower consultant/site or new historical coordinate is asserted.

The public-domain [Internet Archive edition](https://archive.org/details/linguisticstudie00bailrich) has 310 PDF pages, SHA256 `7a1577a2d2a24eca8b95e9b1ed700ef01f716d1069e4468f36a673611d093b39`. The ignored local cache is `tmp/pdfs/bailey1920/bailey1920.pdf`; `--check-pdf` verifies it. Source-side regeneration uses checked-in inventories and does not need the PDF.

## Full scope and unresolved readings

Every headword cell on **pp. 185–188 (PDF 211–214)** is accounted, from *above* to the final *you/your* cell. There are **243 cells**: 71 on p. 185, 68 on p. 186 (34/34 columns), 71 on p. 187 (36/35), and 33 on p. 188 (19/14). Wrapped definitions and separately glossed answers remain one source accounting cell. Grammar, paradigms, prose specimens, and South Jubbal are explicit exclusions; no glossary page or column is omitted.

The authoritative `full-transcription.tsv` contains every literal reading. Older `transcription.tsv`, `completion.tsv`, raw text-layer extracts and pilot audits remain historical evidence. **242 accepted cells produce 330 forms**, with one English cross-reference (*above*) and no unresolved readings. All 98 previously installed keys survive and are recorded in `legacy-entry-keys.json`; corrections preserve those keys.

The 15-column installed CSV retains printed-page/item keys and `:answer2` etc. for additional printed answers, with aligned lexical glosses and grammar. Source noun/verb/adverb, gender, transitivity, causative, relative/interrogative and instrumental labels are structured. Correlative prose stays in the audit rather than a lexical gloss. No phonemic interpretation, etymology, borrowing or directional variant relationship is inferred; all 330 are unlinked attestations.

The profile preserves source breves, macrons, nasal stacks, underdots and underlining literally. Underlined sh uses U+0332 beneath each letter; the printed raised breath mark is represented by U+2018 ‘ without phonological interpretation. House conversions `w → v` and `ṅ → ŋ` affect display only; Original retains source spelling. Source word boundaries remain intact.

Full recovery's initial independent 20-cell review found five affected cells: omitted nasal tilde, an absent underdot, an added underdot, a mistaken vowel/mark, and a grammatical label in a lexical gloss. The ensuing source-wide second image reading corrected 24 form cells, recorded in `literal-corrections-20260926.json`. It distinguished daughter/girl ṅ from ṇ, plain l from ḷ in say/speak and maize, and real vowel lengths and nasal stacks. Two disputed readings were independently adjudicated: call `budno` (the source explicitly says not -no) and thou `tū` (u under a single macron; the reviewer's earlier ī interpretation was retracted). Fresh stratified seed 2026092652 excluded the first 20 and passed **0/20**, with hashes matching the final transcription and installed CSV. Historical failed and pilot reports remain available.

Seven focused tests pass, including exact accounting/regeneration, all 98 prior keys, literal error-class regressions, grammatical alignment, source-wide profile coverage, fresh audit hashes, and **330/330 scoped parse** without conversion errors while preserving Original. Metadata validation passes. This closes the source stage; full CLDF generation, compiled graph/durable-ID checks, regenerated references and full suite remain deferred by the user's instruction. Browser database and representative app QA require an explicit refresh request. No remote work, commits or publication were performed.

```sh
.venv/bin/python data/other/forms/raw_data/bailey_north_jubbal_1920/import_source.py --install --check-pdf
.venv/bin/python -m pytest -q tests/test_bailey_north_jubbal_1920.py
.venv/bin/python source_meta.py
```
