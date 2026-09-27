> **Whole-source scope reopened (2026-09-26):** Existing installation covers only the glossary. Preglossary grammar and complete translated responses remain to be recovered; see `whole-source-reopening-20260926.json`.

# Bailey 1920 Surkhuli, full vocabulary

T. Grahame Bailey, *Linguistic Studies from the Himalayas* (London: Royal Asiatic Society, 1920), “Koci: Surkhuli dialect.” The public-domain [Internet Archive edition](https://archive.org/details/linguisticstudie00bailrich) is the primary source; LSI IX(IV)'s addenda cite this later study, not conversely. The 310-page scan is cached at workspace `tmp/pdfs/bailey1920/bailey1920.pdf`, SHA256 `7a1577a2d2a24eca8b95e9b1ed700ef01f716d1069e4468f36a673611d093b39`.

The complete alphabetical vocabulary occupies printed **pp.155–158/PDF181–184**, from *above* through *your*. Every cell is accounted: page totals 69/70/72/13, with left/right columns 34/35, 36/34, 36/36 and 8/5. **223 accepted cells produce 267 forms; one English cross-reference (*on*, see *upon*) has no separate lexical answer.** No lexical cell is withheld. Grammatical paradigms/prose and neighboring Koci lects are outside this glossary scope.

`full-transcription.tsv` is authoritative. Historical `transcription.tsv` and pilot audit remain for comparison; all 29 legacy keys survive. Four legacy readings were corrected without changing keys: bad nĭkāmmau, bed mănzā, bird tsīṛū and book kătāb, recorded in `legacy-reading-corrections-20260926.json`. The complete pages were visually read; retained OCR text-layer extracts provide an inventory aid only, and metadata credits that contribution. Exact page/item provenance and each decision remain in `audit.jsonl`.

Canonical language `surkh` is retained; no finer locality, speaker or new dialect is asserted. Printed answers get stable child keys, with aligned lexical glosses and grammatical tags. Gender, noun/adverb/verb, relative/interrogative and instrumental labels are structured; correlative labels stay in the audit. No etymological, borrowing or directional variant relationship is inferred: all267 forms remain unlinked attestations.

The preservation profile retains readable breves, macrons, underdots, nasal stacks and underlining literally, except house w→v and ṅ→ŋ. Underlining is retained beneath both letters of sh or kh. Source notes about very long vowels in eleven/twelve/thirteen, and first-syllable accent in nineteen, remain lexical Notes rather than invented phonemic symbols. The single internal stop in return `ōru. ăs̲h̲ṇo` is preserved in Original/audit but removed from normalized Form, yielding `ōru ăs̲h̲ṇo`. The independently printed come verb supports a phrase reading; typed transcription uncertainty, a discovery tag and Notes disclose the punctuation interpretation. No other cell contains a period affected by this rule.

Independent seed2026092671 reviewed five accepted cells per page and passed **0/20 material errors**. Additional enlarged edges confirmed fourteen, daughter, remain, maize and underlined kh. The return punctuation edge was resolved after that sample; `return-normalization-addendum-20260926.json` pins final source/CSV/profile/audit hashes and verifies all20 sampled forms, glosses and tags stayed unchanged.

Seven focused checks pass, including full accounting/regeneration, preserved keys, literal/grammar regressions, profile coverage, audit/addendum hashes and **267/267 scoped parse** preserving Original without conversion errors. Metadata validates. This is source-stage closure; full CLDF generation, graph/durable-ID verification, regenerated references and full suite remain deferred under the user's instruction. Browser database construction and representative app QA require an explicit refresh. No remotes, commits or publication were used.

```sh
.venv/bin/python data/other/forms/raw_data/bailey_surkhuli_1920/import_source.py --install --check-pdf
.venv/bin/python -m pytest -q tests/test_bailey_surkhuli_1920.py
.venv/bin/python source_meta.py
```
