# KeraɁ Mundari complete pinned website proposal

The ingestion checklist and dictionary/website/API addenda are active. This is a **review proposal**, not a canonical installation or completed end-to-end ingestion.

## Scope and completeness

The public SQLite snapshot has 446 entries and 446 senses, exactly matching the acquired website's advertised count. All tables are counted in `snapshot-completeness.json`. The complete published snapshot replaces pagination: there is no unvisited page cursor. Its latest record timestamp is 2026-07-23; retrieval was 2026-09-27. Later live changes are outside this immutable snapshot.

Every entry, sense, 443 provided POS assignments, 440 notes, semantic-domain code and all 28 borrowing annotations are accounted for. The notes contain elicitation prompt identifiers, sometimes a 2011-09-28 date, not omitted lexical prose. All 445 audio IDs, storage paths, source/timing fields and dates survive in the audit. Audio bytes and personal account fields are not copied into lexical output. The source has no sentence, text, grammar-section, example, relation, photo or video records. Empty structures remain explicitly inventoried.

The separately authored about page contributes 16 target-family lexical/morphological units, producing 17 readings. These include KeraɁ past -kera, other unspecified Mundari -keda and two explicitly HasadaɁ comparisons with separate dialect routing. The phoneme inventory and abstract syllable templates remain contextual evidence. Singular unmarked morphology and the two English phrases “three sons” and “two dogs” are three further audit-only units without invented visible target forms. Background publication Kobayashi and Murmu (2008) has an auxiliary bibliography record and is not claimed as a directly inspected lexical source.

Total proposal: **479 rows**, comprising 462 dictionary readings and 17 about-page readings; **466 audit units**. All 446 upstream IDs and sense IDs remain recoverable. Homonyms are not collapsed. Thirteen explicit tilde pairs and three optional-segment spellings produce 16 additional dictionary readings. Their links only connect readings of the same explicit source form. The about-page optional final glottal stop produces one further variant. No donor-language forms, etymological links or unattested bases are invented.

## Editorial decisions

The source labels `adp` as generic adposition. All 30 retain this category without guessing preposition or postposition. Backend and frontend register `adposition`. Other POS categories map only exact source labels; pronoun features use the reviewed exact pronoun gloss vocabulary, and ten numeral prompt entries retain both nominal POS and numeral classification. Source labels and unaltered glosses remain in the audit. Lexical parentheticals are preserved; explicit grammatical qualifiers are structured. A full utterance is sentential, not automatically verbal. Two adverbial-phrase labels do not imply an invented verb or orthographic word boundary.

The dictionary's so-called dialect assignments actually contain 26 Hindi Loan, one English Loan and one Possible Hindi Loan labels. These become explicitly attributed borrowing statements; the possible assignment remains uncertain. No assignment defines a separate lect.

All upstream phonetic fields are empty. The proposal therefore preserves NFC Roman lexemes literally, with blank Native and Phonemic. The identity profile documents every NFD symbol and performs no phonological conversion: literal `g` remains `g`, and diacritics, glottal-stop shapes and unusual source spellings are not harmonized. The generic parser uses `convert: false`; the profile's historical `IPA` column is only a literal identity mechanism and makes no IPA claim. About-page general phonological statements do not justify inferred pronunciation for individual entries.

Proposed lect is Mundari (`mu`) qualified as KeraɁ Mundari, Glottocode `kera1257`. Eastern Ranchi is a source community description, not a pinpoint collection site. Coordinates remain blank. The source's atlas point is not adopted as elicitation coordinates.

Compiler Gregory D. S. Anderson and publicly credited speakers/technical contributors are recorded in the proposed bibliography. Alex Mussomeli authored the background description. Reuse is authorized by the user's explicit confirmation that Living Dictionaries is allowed per Sarva; no CC or public-domain licence is inferred from absent copyright metadata. Acquired terms and page bodies remain with the original snapshot. Public administrative account details are excluded from the lexical export.

## Validation and remaining gates

`prepare_source.py` deterministically generates the 15-column proposal, full audit, literal profile and counts. Eight focused tests cover actual generic YAML parsing of all 479 rows, source identity conservation, meaningful alternate/homonym checks, borrowing uncertainty, exact grammar registration, NFC/NFD coverage and the actual source metadata/reference formatter. The final run result is retained in `focused-tests.log`. No full database build was run.

Independent source-to-output 20-entry review and targeted controls are pending. Canonical installation, registered bibliography/dialect metadata and durable identity checks will follow a passing audit within the user's standing ingestion authorization. Compiled graph checks, full database build and browser QA remain deferred under the current no-build instruction. No completeness claim extends to these deferred gates. The installed source count remains zero for this new package.
