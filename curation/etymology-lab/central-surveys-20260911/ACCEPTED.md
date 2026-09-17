# Approved Malvi, Nimadi and Bagheli analyses

**All 757 numbered proposals are saved: 3,777 links on 3,731 records.** The user's approval includes the qualified proposals. It does not supply analyses for the 164 held records or 2,680 unexamined records; those remain unlinked by this task.

| Survey | Proposals | Records | Saved links |
|---|---:|---:|---:|
| Malvi | 291 | 1,043 | 1,054 |
| Nimadi | 230 | 1,908 | 1,932 |
| Bagheli | 236 | 780 | 791 |

**Perso-Arabic nesting:** 223 loan links now point to Persian or Arabic etymological entries rather than Hindi-Urdu proxies. This is an editorial grouping of the loan families, with immediate transmission unspecified. Notes retain morphological and semantic qualifications: regional feminine murgī under murɣ; the regional woman/wife sense of Persian-labelled ʻaurat; regional adjectival sabūt under Arabic s̤ubūt; and the complete waznī and wazn-dār adjectives. All inherited, derived and ordered-component proposals otherwise retain their approved analyses.

Nine missing dictionary heads were added: Arabic badan, s̤ubūt, makān and waznī; Persian ʻaurat, zabān, kam and wazn-dār; Hindi pʰūlgobī. The Hindi head resolves the six cauliflower links without treating an unlinked survey token as an established donor. No existing Hindi head was relabelled. Every new head has a source audit and stable ID. See [donor corrections](donor-nesting-corrections.json) and [source checklist](../../../source_checklists/20260911-central-surveys-donors.md).

**Validation:** the fresh temporary graph accepted all 3,777 links, including dependencies, and changed nothing on a second application. All 22,644 previous overlay rows and every unrelated graph edge were preserved. The registry gained exactly nine rows, preserving its previous bytes. The accepted overlay is `data/etymology-assignments.csv`; backups and exact hashes are recorded in [acceptance validation](acceptance-validation.json).

The complete compilation stages, including references, concepts and alignment, ran in `/tmp/jambu-central-approved-build`. All 3,777 edges, all nine new heads and all selected source IDs, meanings, citations and dialect tags survived. No previously compiled ID was lost; exactly nine IDs were added. See [compiled checks](compiled-acceptance-validation.json). The final `make all` manual-survey gate failed two tests concerning the existing Rajasthani record count and source-owned overlay rows; both failures reproduce against the shared checkout. Thus `make all` did **not** pass as a whole.

The dedicated supplement tests passed (3/3). The combined donor, sound-profile, dialect, identity and edge checks returned 36 passed, 9 skipped, 1 failure: an existing dictionary self-reference count of 2,603 versus expected 2,604, independently reproduced against the shared checkout. The full suite returned **1,733 passed, 21 skipped, 52 failed** in 437.70 seconds. Failures include missing external source-PDF/sibling-frontend fixtures and unrelated corpus/reference assertions. None is in the new donor test module; this is not a clean global suite. See [test results](test-results.json) and [full log](acceptance-logs/full-suite.log).

The subsequent user-requested shared CLDF and browser rebuild is staged locally as cache version 32. All approved links and donor heads passed verification and representative browser QA. See [rebuild results](DB-REBUILD.md). No commit, push or deployment. The overnight automation remains paused. [Saved review index](TRIAGE.md) and [complete review with held cases](REVIEW.md) now reflect the accepted state; the original research manifests are archived in `before-approval-manifests.json`.

Representative new entry IDs (available in the rebuilt local database):

- Ar **badan**: `f_e3ub3b5pwfylq`.
- Pers **ʻaurat**: `f_qc74nzbufr4lq`.
- Pers **zabān**: `f_ouyhnvl4bjh5i`.
- Pers **kam**: `f_oz4z6vzzul6kw`.
- Ar **s̤ubūt**: `f_vb2pgt55djlt2`.
- Ar **makān**: `f_djo4xiwrnmgdw`.
- Ar **waznī**: `f_ymrdn4fxzm2dk`.
- Pers **wazn-dār**: `f_uar6au5j3rvqo`.
- H **pʰūlgobī**: `f_a3zgar2k27pzm`.

All nine new heads were audited; there are zero donor-specific conversion errors and zero unrelated compiled-form changes. The complete data-ingestion checklist cannot be marked globally green because the broader tests above fail. The approved overlay save and batch-specific graph/source checks are complete.
