# Grierson 1916 Handuri: full source stage

The active ingestion checklist covers survey/comparative tables, grammatical paradigms and historical page-image transcription. This package accounts for the complete Handuri chapter (printed pp. 586–592) and all 241 Handuri prompts in the standard list (even printed pp. 628–644). The pinned original is `tmp/pdfs/lsi-v9-4/LSI-V9-4.pdf`, SHA256 `ef007663270b3a0ef5ba26804404e4db1281fb5eb239614cadf69b20b3d5395f`; printed page +16 gives the PDF page.

There are **670 source units and 587 installed forms**:

- 241 table cells: 239 nonblank cells produce 241 forms, including two explicit alternatives. Items 174 and 201 are printed blanks.
- 86 grammatical units produce 103 forms. Four morphological endings remain inventory-only; they are not invented standalone words.
- 343 interlinear specimen attestations produce 243 distinct exact form-and-gloss entries. The 100 repeated attestations retain every locator on the first matching entry. No cross-section, case-folded or reconstructed-lemma merging occurs.

All ten earlier numeral entry keys remain. The old dialect token in Derivation_Parent_Keys was corrected to Tags. No source-supported etymological or derivational edge is asserted by this source; adjacent future expressions in item 173 remain one literal sequence because the print supplies no separating mark. Item 231 preserves the discrepancy between the English prompt and the source’s “than him” annotation.

The chapter labels the lect Handuri and its specimen Nalagarh State; the introduction also discusses east Nalagarh and Mailog. It maps to existing Hinduri (`hind1267`, `hii`) and the existing Handuri source dialect. No precise coordinates are inferred. Neighboring Kiuthali, Siraji, Shoracholi, Patiala and Kochi material is excluded as separately identified language evidence. Native-script pp. 588–589 and aligned Roman pp. 590–592 are two representations of the same specimen. Sequence and passage boundaries were reconciled; no diplomatic Devanagari transcription is claimed.

The three `*-transcription-staged.tsv` files are the authoritative reviewed transcription inputs despite their historical filenames. Every table cell and grammatical unit was visually reread at 350–600 dpi; all 42 interlinear lines were inventoried and reread with bounded crops. Seven table readings (18, 19, 24, 25, 92, 112, 127) retain specific scan-mark uncertainty and are included with `uncertain`, rather than discarded. Audit JSON retains complete review/process evidence. Final CSV Notes contain only material interpretation and uncertainty.

The registered `grierson-handuri-1916` profile preserves the printed historical distinctions, lowercases capitals, and applies house `w → v` and `ṅ → ŋ`. Spaces and hyphens remain boundaries. Parentheses and question marks remain in Original but are removed from normalized forms. No phonemic reconstruction is claimed. Explicit source paradigm person, number, gender, case, tense and voice are structured in Tags. Comparative/superlative meanings remain explicit in the gloss because the shared registry has no dedicated degree tags.

Independent review: `independent-audit-20260926-pass1.json` sampled 20 fresh units across all sections with zero material errors. A regression compares those actual final CSV form/gloss/citation/key values with the reviewed preview after Notes cleanup and grammar additions. `fullscope-validation-20260926.json` records hashes and final checks. All 25 focused source/profile/dialect tests pass; all 587 rows pass the scoped parser; source metadata validates. Full database, full suite, compiled graph/reference integration and browser QA remain deferred by the user’s explicit no-build instruction. This is complete at the source stage, not an application-release claim.

Regenerate from `data/`:

```sh
.venv/bin/python data/other/forms/raw_data/grierson_handuri_1916/import_source.py --install --check-pdf
.venv/bin/python -m pytest tests/test_grierson_handuri_1916.py tests/test_sound_profiles.py tests/test_dialects.py -q
```

Representative stable entries: `grierson1916handuri:p628:item:8` (Aṭh, eight), `grierson1916handuri:p636:item:133` ((Tĕs-tē) kharā, better), `grierson1916handuri:p587:grammar:servants-example` (hāṛīyā̃-khē, to the servants), and `grierson1916handuri:specimen:p590-l10-u02` (tĕs-khē, him-to, with repeated citations). Application display verification is deferred with the database build.
