# Paired source-stage handoff

The complete proposals are frozen in `whole-paired-output-freeze-20260927.json` for independent actual-output review. They are not canonical installations. No database, full CLDF build, full suite, browser or publication gate is claimed.

| Package | Proposed rows | Existing keys preserved | New rows | Audit units |
| --- | ---: | ---: | ---: | ---: |
| Rampuri carrier | 647 | 288 | 359 | 640 |
| Baghi | 698 | 296 | 402 | 640 |

Pass1 failed on one inherited Baghi quantity error. Its complete frozen artifacts remain in `whole-audit-pass1-artifacts`. Pass2 is separately versioned in `whole-paired-output-freeze-20260927-pass2.json`. The exact keyed delta preserves 573 old rows unchanged and corrects eleven: one literal vowel mark and ten parsed grammatical gloss labels (including two newly explicit correlative tags). Every old key survives; no other field changes are authorized by that delta.

Rampuri includes four shared Koci examples under existing `kc`, once only, with blank glosses and precise attribution uncertainty. Its additional body row `bailey1920rampur:p144:item:19` preserves literal `jĕā́`, independently checked by root; the historical hold remains in its snapshot and final audit. Baghi has no source-stage reading holds. Typed uncertainties are retained on 16 Rampuri-carrier and 7 Baghi rows, including attribution and alternative-segmentation questions, not all illegible readings.

The paired supplementary inventory has 395 source groups: 305 grammar/heading groups, 44 whole translated responses, 19 commentary groups and 27 introduction groups. Twenty-three groups preserve contextual or other-lect information rather than emitting target-language forms. Same-as paradigms have candidate-key reuse accounting. The 44 physical responses yield 45 complete expressions because Baghi sentence18 explicitly permits two water terms. Each package also retains the 245-cell glossary audit; unpaired columns and cross-reference-only entries remain explicitly accounted.

Source scope is shared introduction113–115, heading-scoped Rampuri116/Baghi116–117, paired grammar131–140, sentences141–142, commentary142–143 and glossary144–147. Other lects on115–118 and boundary pages130/148 are controls. Original printed page n is one-based PDF n+26. The original PDF hash and exact review locators are in the freeze and sidecars.

Before installation:

1. Complete root's independent actual-output audit and reconcile any errors with preserved failed candidates.
2. Retain the existing glossary generator as `generate_glossary`; add a guarded whole-source `generate` that checks approved hashes and reproducibility. `install_paired_whole_proposal.py` proposes a bounded source-only sandbox with immutable legacy stubs, so the draft builder's canonical-input guard need not be weakened. Require root's zero-error audit to pin the current freeze before verification or installation; verify regenerated CSV/audit/profile bytes before any canonical write.
3. Preserve source filenames, citation IDs, append orders (Rampuri73; Baghi51), all584 existing keys and key-based deduplication. Apply only the eleven documented legacy corrections after root verifies them. Existing split-alternates:false remains because alternatives are explicit rows with closed source-local links.
4. Set both YAML transcription defaults to `input: form` and `preserve_hyphens: true`; retain their own profiles. Keep editor/OCR flags and etymology_provenance:none. Update importer notes to whole-source scope. Existing canonical `ramp`, `ba`, `kc` require no new language, dialect or coordinate.
5. Update both existing bibliography entries to whole-source pages and paired-source provenance. Do not create a new independent field event for this paired table. The generic Koci examples are only in the Rampuri carrier, with their exact shared-source label.
6. Install only the exact approved CSV/audit/profile outputs, update package READMEs and whole-source ledgers/checklists, then verify the real canonical parser and metadata routes without monkeypatches.
7. Update old glossary-only tests to distinguish immutable historical fixtures from current whole-source outputs. Retain the seven new source regressions and add canonical regeneration/identity checks.

Current source-only validation: all1,345 actual parser rows pass in both NFC and NFD with exact ordered keys, Original, Notes, Source, Tags and expected display; both proposed profiles have zero shared-house-policy violations; seven focused regressions pass. The parser uses unchanged `make_cldf.py` and `source_meta.py` in a small workspace sandbox with the intended YAML settings. Canonical CSVs, profiles, YAML and bibliography remain unchanged at this handoff.
