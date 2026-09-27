# Bailey Padari: full-source recovery

Status: full source staged, awaiting independent audit and installation. The installed 13-row pilot is unchanged.

The dictionary/OCR and linguistic-source ingestion checklist applies. The 358-page original book was searched and the complete Padari accounts and every explicitly Padari-attributed supplementary comparison were inventoried. All 753 physical source units have two visual readings against the original scan. The complete account is Part III pp.76–84; the earlier account is Part IV pp.33–35. Additional printed witnesses occur in Part III pp.ii,28,53–55,89–90 and Part IV pp.70–71. General discussion and examples assigned to other languages are not silently relabelled Padari.

The staging contains 768 rows from 748 nonblank units, with all 20 separately printed alternatives retained. Five numbered cells in the earlier account are blank: ass, camel and the three deer prompts. They remain individually accounted for rather than filled from another witness. All 13 installed pilot keys are preserved. The full unit inventory, printed stem abbreviations, expanded forms, grammar labels, page locators and decisions are in `full-transcription.jsonl` and `proposal-audit.jsonl`.

## Transcription

Form retains Bailey's literal Roman notation, not inferred IPA. The staged profile preserves every source character. Superscript half-vowels, short-vowel breves, nasalization, underlined digraphs and the shared macron over eu remain distinct. The latter represents the source legend's single long French-eu vowel and is not flattened to independently long ēū. Contrastive italic vowel values are identified in the affected Notes; typographic italics used throughout prose examples are not automatically treated as an additional contrast. The author's explanation that certain ĕ letters should have appeared above the line but could not be typeset that way is retained as a source claim.

Source variants and apparent inconsistencies are preserved locally, including the earlier mare/mares r versus ṛ distinction. No word is supplied from analogy. Two ink-fused i marks on Part III p.77 have unresolved length: first dative alternative meuĩ and singular Agent maĩ. They carry specific glyph-uncertainty notes and the uncertain tag. The separately printed second dative alternative maī̃ is clearly long and is not tagged uncertain.

Case, gender and grammatical labels follow explicit source paradigms; glossary POS is contextual and identified as such in Notes. Historical Agent is represented with erg. The registry's pret, participle, prep and impv tags represent the source past, participle, preposition and imperative labels. Asterisks in the earlier account mean resemblance to Pangwali words, not reconstruction, borrowing or inferred cognacy. Source comparison claims do not generate unsupported etymological edges.

## Existing evidence

`same-source-reuse-alignment-20260926.json` records exact same-print overlap for hair rŏṭṭh, quoted by Zoller from Bailey p.82. Both evidence paths remain and are not counted as independent field evidence. The old fox exclusion is rejected: Zoller's cited path is LSI and its transcription differs; identity of the printed source was not established. No Zoller rows or their etymological evidence are removed. Downstream identity reconciliation that requires compiled graph checks remains deferred.

## Validation and deferred gates

`prepare_full.py` generates staging only. Seven focused staging tests pass (eleven including the unchanged pilot): complete scope/key accounting and alternate-target closure, difficult typography regressions, complete literal profile round trips with registered tags, scoped parsing of all 768 rows, explicit source-header metadata, and frontend/backend correlative-tag parity. Independent pass1 seed20260926116 found no literal errors in20 units but two omitted header categories. All75 affected units across the explicit header classes were repaired; the original four frozen files are preserved in `independent-pass1-frozen-inputs.zip`. Fresh disjoint20 selection seed20260926118 and current frozen hashes are in `independent-full-sample-20260926-pass2.json`.

Canonical installation, independent audit result, final metadata/profile integration and installed parity checks remain pending. Database generation, full build, compiled graph checks and browser QA are explicitly deferred under the user's no-build/no-remote instruction. No ingestion-complete or app-verification claim is made.

Six-position verb tables have no printed person-number column labels. The source supplies partial clues elsewhere but also calls bhōnal “I shall be” despite its fifth/sixth position. Positional evidence and the discrepancy are retained without inventing a uniform person-number interpretation.

Independent pass2 found one hau/han reading error and omitted adjectival-degree metadata. A239-unit plain n/u review identified two literal corrections in total (earlier auxiliary1 and main sentence14); all three explicit comparison constructions now carry adj+degree without inferred inflection. Prior inputs are retained in `independent-pass2-frozen-inputs.zip`. Fresh pass3 seed20260926120 excludes all40 prior sampled units.
