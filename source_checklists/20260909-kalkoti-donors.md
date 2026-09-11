# Kalkoti batch 17: approved donor selection and assignments

Scope: user-approved proposals 335–402, saved 2026-09-09. This is a selected editorial donor supplement, not a complete import of any dictionary. All 68 analyses were reviewed before saving.

- 68 proposals / 84 existing Kalkoti records / 108 assignment rows: 48 component edges for 24 constructions, one derived edge, 59 borrowing edges.
- 43 new curated donor heads: 31 Pashto, two Gawri (registered Bshk), two Hindi-Urdu (H), eight English. No new languages, dialects or coordinates.
- Sources: 21 selected OPED entries, 12 Hultman donor comparisons, two Platts entries, eight editorial English source-family identifications attributed to arora and the Kalkoti attestation source.
- All previous assignment rows and identity rows were preserved. Only 43 new durable donor identity rows were appended.
- Compiled remainder in the isolated validation build: 202 Kalkoti records, 177 distinct exact forms; 56 multiword records.

## Checklist audit

1. Source/scope: exact selected head inventory and citation locators are in the per-record audit. OPED web edition states 9 August 2026 update; consulted 9 September 2026. DOI 10.5281/zenodo.17487678. Platts 1884 is public domain. OPED redistribution licence was not retrieved; only selected lexical facts and short evidence excerpts are included. English identifications are approved editorial analysis. Full dictionary coverage, other senses and unapproved analyses are excluded.
2. Extraction: no OCR or bulk parser. Existing source extraction and the inspected dictionary entries underpin the approved heads. Stable dictionary IDs and dated source evidence are pinned in the audit. Native/source text remains in the evidence excerpt when supplied; these are curated etymon heads rather than a new attestation inventory.
3. Files/identifiers: deterministic parameter emitter in data/other/params/raw_data/kalkoti_donors.py; complete audit JSON and approved-analysis JSON beside it. Stable source-local donor IDs map to build-issued opaque IDs recorded as Persistent_ID.
4. Languages/dialects: four existing canonical language IDs; Gawri uses Bshk. No new lect/site claims, metadata or coordinates; those gates are inapplicable.
5–7. Schema and transcription: intentionally use the established five-column curated-parameter format. Preserve reviewed donor transcription with NFC only. Do not infer phonemic conversion for these foreign donor heads. Original reviewed spelling, variants, evidence and source remain in the audit. No replacement characters or empty donor heads. Grammatical analysis of Kalkoti inflections stays in the accepted notes; the teacher donor head is ustād with explicitly attested ustāz recorded as its source variant. Rich attestation morphology/IPA emission, OCR and new conversion routing are inapplicable.
8. References: added oped2026 and platts1884 bibliography records; existing Hultman, Liljegren and editorial references retained. make_refs.py completed in the isolated build. All donor and affected Kalkoti citation keys resolve; no snapshot-date placeholder citation survives. Snapshot locators use commas, preserving semicolons as citation separators.
9. Graph: approved relations stored in data/etymology-assignments.csv. 43 linkable entry nodes, no invented reconstructions. Donor-route qualifications retained. Compounds have contiguous component positions, and the relative noun has a derived link. Gold, thousand and poison remain distinct. The room donor uses OPED 31562, not either homonym. The older xośħāl node mislabeled Arabic was not reused or silently modified.
10. Audit: all 43 selected donor records accounted for; 0 selected heads excluded. Seed 1709 sample of 20 checked against research evidence, 0 material discrepancies. Homonyms, teacher variants, retained plural morphology and citation serialization inspected separately.
11. Focused checks: four source-specific tests passed on the compiled isolated build, including regeneration, homonyms/languages, all 84 Kalkoti records and all 108 graph relations, and citation integrity. Input-only checks also apply to the source checkout before its next build.
12. Pipeline: make_cldf, link_refs, unify_cldf, assign_form_ids, concepts, align and make_refs all executed in /tmp/kalkoti-b17-build/data. No new source/profile errors. The final make all survey checks failed two tests, reproduced unchanged in the original checkout (Rajasthani accepted-link count 11256 vs expected 11245; existing source-owned overlay rows). The full pytest collection initially hit duplicate test module names; rerun with importlib mode. Full-suite final result is recorded below. Therefore repository-wide validation is not claimed clean.
13. Browser refresh: not requested; browser database and application were not rebuilt. Compiled files in the shared checkout were not replaced by the isolated build. Representative future entry IDs are below.
14. Handoff: no commit, push, release or deployment. Canonical sources, donor identity additions and approved overlay are saved. Complete repository-wide ingestion validation remains qualified by the reported test failures.

## Representative compiled entries
- #335: gān drā: f_qtwcqkcg3mo6s
- #369: kitāb, kitā̌b, kitā̌buni: f_ltcpk65cfivq2, f_3ezmuwwrb5rtg, f_rnriwlubwl7hw
- #376: zar / zar: f_jjx73wkn6duwa
- #377: zir, zir: f_5wtiazgyhnie6, f_25cdy64hp2i64
- #378: zār: f_llifejo6hzk6c
- #379: ustad, ústāz: f_oxu223l2j2vhc, f_bhec7qwr55psu
- #387: kæmrá: f_cn34ikml56lia
- #400: drādi: f_l7zs6kavvipse
- #401: yete bāb̥: f_vr22xlnsbs52a
- #402: tani: f_s2fos4w4sbbki

## Full-suite result

Full isolated suite (`pytest -q --import-mode=importlib`): 1650 passed, 52 failed, 17 skipped, 417.15 seconds. Failures include unavailable PDF/frontend fixtures in the isolated copy, stale source-checklist snapshots, existing source/count expectations, and the two survey failures reproduced in the unchanged workspace. No batch-specific test failed in this final run. Four focused checks passed again after the suite. Complete global validation remains non-clean; no blanket claim that all 52 failures were independently proven pre-existing.

Additional verification: all 43 donor IDs remained stable under input reversal; replaying the 108 saved assignments against a temporary copy of the compiled graph made zero changes. Existing overlay and identity rows matched their pre-save records exactly.
