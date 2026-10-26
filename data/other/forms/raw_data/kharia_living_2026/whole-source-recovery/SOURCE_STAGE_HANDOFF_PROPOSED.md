# Kharia Living Dictionary: whole-snapshot recovery

Status: owner-reviewed, frozen proposal awaiting independent output audit and approval. Canonical data has not changed.

The checklist's dictionary/glossary and website/API addenda apply. Living Dictionaries reuse is authorized by the user's standing confirmation, “living dictionary is allowed per sarva.” No media or personal speaker metadata is redistributed by this recovery.

## Source version and completeness

The original 2026-08-20 SQLite snapshot is no longer cached locally. Its historical manifest and complete 452-entry raw audit remain immutable. The small public snapshot downloaded on 2026-09-27 has upstream modification date 2026-09-18 and SQLite SHA-256 `bc4ab203e8f6edc902cda41171c9feed602fe854ea34aee3869b610199167f93`. It is recorded as a distinct revision, not silently substituted for the historical hash.

All 452 entry IDs, source lexemes, phonetic strings, sense IDs, English/Hindi glosses, semantic domains and dialect assignments match the retained historical evidence exactly. The full table census records 452 entries, 452 senses, three example sentences, three sentence/sense associations and no additional populated grammatical, etymological, morphological or text collections. `lexical-snapshot-20260927.json` preserves every relevant lexical field; `snapshot-lexical-comparison-20260927.json` records the exact comparison. Administrative tables, 438 audio records, 69 photographs and two speakers are accounted but not imported as lexical forms.

The proposal contains 524 rows: all 521 existing keyed forms plus three complete native-script sentences. The audit has 455 source units. All existing Forms, Native strings, Phonemic values and keys are unchanged. There are 152 reviewed metadata/Notes changes, including explicit numeral/ordinal, personal/count, clause and POS analyses. English plural nouns do not supply source plural morphology; bare “you” has no inferred number, and “her” has no invented case or possessive classification. Qualified source spellings and homographic analyses remain separate.

## Sentences and transcription

The three sentences have source native text and free translations but no phonetic transcription or interlinear analysis. The full native strings, including apostrophes and danda punctuation, are preserved as Form/Native. Phonemic remains blank. Source automatic tokenization splits meaningful apostrophes as ignored punctuation and offers ambiguous dictionary candidates; these links stay raw audit evidence and do not become aligned-word rows or graph edges. No sentence dialect is inferred from its linked headword's dialect.

The existing profile gains only 12 literal Devanagari character mappings. Existing phonetic conversion is unchanged. There are no new etymological or donor links; the 68 existing source variant links remain closed. The culture head's unexplained leading hyphen is retained with typed uncertainty rather than asserted to be a suffix.

## Review and installation plan

The owner read all 452 entries and every changed metadata decision. The parent independently checked all three sentences directly from the verified SQLite and reviewed all 45 clause outputs plus seven explicit verbal definitions; this targeted review is not whole-output approval. Fresh seed 20260927116 selects 20 units / 25 output rows across sentences, clauses, numerals, pronouns, alternatives and ordinary lexical entries, excluding the historical 20 sample IDs.

Four focused tests passed in 0.69 seconds using the proposed YAML/profile and actual parser. They check stable keys/fields, exact native expressions, grammar scope, every profile symbol in NFC/NFD, complete emitted-key survival, citations and closed variants. The installer dry-run validates the frozen manifest and all baseline fields. Installation requires a parent approval file pinning that exact manifest.

After approval: install the exact CSV/YAML/profile, apply only the scoped `living-kharia2026` bibliography block, and delegate the historical `kharia_living.py --install` entry point to the guarded coordinator so it cannot restore the headword-only snapshot. Keep historical audit/manifest unchanged. Verify repeated regeneration and actual canonical parser/reference/dialect routes, then update current README/checklist/ledger and save installed hashes separately from immutable review history.

Full CLDF, complete test suite, compiled graph/reference refresh, database generation and browser QA remain explicitly deferred. No publication or release is authorized by this ingestion task. Representative source keys will be supplied after installation; no new browser entries have been verified.
