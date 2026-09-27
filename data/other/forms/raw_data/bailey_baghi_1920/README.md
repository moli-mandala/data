> **Whole-source scope reopened (2026-09-26):** Existing installation covers only the glossary. Preglossary grammar and complete translated responses remain to be recovered; see `whole-source-reopening-20260926.json`.

# Bailey 1920 Bāghī complete glossary

The source is T. Grahame Bailey, *Linguistic Studies from the Himalayas* (London: Royal Asiatic Society, 1920), “The Koci Dialects of Rampur State.” Printed p. 113 places Bāghī in the small area around Baghi. This maps to existing canonical `ba` Baghi; the author gives no consultant or finer elicitation locality. No new dialect or historical coordinate is invented.

The original edition is public domain. The [Internet Archive scan](https://archive.org/details/linguisticstudie00bailrich) has 310 PDF pages, SHA256 `7a1577a2d2a24eca8b95e9b1ed700ef01f716d1069e4468f36a673611d093b39`. The local ignored cache is `tmp/pdfs/bailey1920/bailey1920.pdf`. The importer can verify it with `--check-pdf`; source-side regeneration uses checked-in inventories.

## Exhaustive glossary accounting

The scope is every Baghi answer in the complete **pp. 144–147 glossary**, *above* through *your*. The first answer before each colon is Rampur; the Baghi answer is after the colon, as explained on p. 144. All **245 source headword units** are accounted: 59 on p. 144, 66 on p. 145 (33 per column), 63 on p. 146 (32 left/31 right), 57 on p. 147 (30 left/27 right). The *much* block starts at p. 145's end and continues atop p. 146; it is one inventory unit including that continuation. No arbitrary page slice remains. Grammar, paradigms, other Koci lects, and Rampur controls are explicitly excluded.

`full-transcription.tsv` now supplies the complete visually recovered source readings. Earlier `transcription.tsv`, `completion.tsv`, raw OCR and pilot reports remain historical evidence. **240 accepted cells yield 296 installed forms.** There are no unresolved Baghi readings. The other five units are two English cross-references (above, conquer) and three unpaired first-column answers (eighty, forty, sixty), excluded under the author's first-word Rampur convention. The historical decision label `hold_no_baghi` means an excluded control, not a pending reading. The unpaired *whatever* continuation is likewise excluded within the otherwise accepted *what* cell. Root independently checked the p. 144 introduction and exclusions against the scan.

All 96 previously installed keys survive, recorded in `legacy-entry-keys.json`. Corrections retain their keys. Distinct printed answers expand to stable `:answer2` etc. keys with aligned glosses and grammar: causatives, relative/interrogative forms, gender, transitivity and instrumental labels. Each row has an exact printed-page/item locator. No etymological, borrowing, phonemic or directional variant relationship is inferred; all 296 are unlinked attestations. Canonical language `ba` remains unchanged, with no new site or dialect coordinates.

The profile preserves readable breves, macrons, underdots, tilde stacks and underlining literally; `s̲h̲` uses U+0332 beneath both letters. The house conversion `ṅ → ŋ` changes display only; Original preserves the source. The profile is complete over all installed inputs. Ambiguous phonological interpretation is not grounds to suppress a legible printed form. Source identity settings protect separate concepts and homographs.

Earlier pilot audits remain historical. The full literal recovery's independent pass 1 found four affected cells in 20, including drink causative `pĩnēṇo`, river `dăryaio`, and missing causative/relative grammar. These readings were corrected and grammatical labels reviewed source-wide, including the paired Rampur source. Learn is `s̲h̲īkṇo`, without an added h after k. Fresh independent pass 2 (seed 2026092642, five cells per page excluding pass 1) found **0/20 material errors**; its hashes match the authoritative transcription and installed CSV. The audit report and earlier failures are retained.

Eight focused tests pass: exact reconciliation, full regeneration, all 96 prior keys, corrected literal classes, grammar alignment, exclusions, fresh independent hashes, source metadata/profile routing and 296/296 scoped parse with Original preservation and no conversion errors. Full CLDF generation, graph/durable-ID checks, formatted references and full-suite tests remain deferred under the user's no-build restriction. Browser database and representative app QA require an explicit refresh request; no newly built app entries exist for this stage. No new renders, remote actions, commits or publication were performed.

From the data repository:

```sh
.venv/bin/python data/other/forms/raw_data/bailey_baghi_1920/import_source.py --install --check-pdf
.venv/bin/python -m pytest -q tests/test_bailey_baghi_1920.py
.venv/bin/python source_meta.py
```
