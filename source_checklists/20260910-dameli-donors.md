# Approved Dameli donor supplement — 10 September 2026

Status: donor records and approved etymology assignments installed; **full ingestion validation incomplete**. Dictionary/glossary and website/external-data addenda apply; survey addendum applies to seven promoted existing attestations. The user authorized new missing donor entries and approved all 99 overnight proposals. Pashto where best matched, followed by Urdu for broadly Perso-Arabic terms, is an editorial donor preference; it does not establish otherwise uncertain transmission.

## Scope, evidence and reproducibility

- 33 selected heads: 18 Pashto (`Psht`), 15 Hindi-Urdu (`H`, Urdu donor usage). Six OPED heads, twelve Platts heads, eight donors explicitly printed by Perder, seven promoted existing LSI/Hindu Kush/Hallberg attestations. This is a selected donor supplement, not complete coverage of any dictionary.
- Parameter CSV: `data/other/params/20260910-dameli-donors.csv`. Matching 15-column self-attestations in `data/other/forms/20260910-dameli-donors.csv` preserve native script, grammar, source keys and source etymology on the heads.
- Reproduce with `python3 data/other/params/raw_data/dameli_donors.py --install`; without `--install`, checks byte-identical installed files. The adjacent audit JSON is the editorial input. Every selected head has an immutable key, exact citation locator, evidence, and approval status.
- OPED: preliminary XML archive, 30 October 2025, DOI 10.5281/zenodo.17487678, CC-BY-4.0. The selected XML and archive checksum are frozen in `20260910-dameli-oped-snapshot.json`. Homonym IDs distinguish bən co-wife and gulābí pink (32639, not barber 32638).
- Platts: 1884 dictionary, public-domain lexical source, selected facts checked in Rekhta's reproduction. Audit evidence explicitly combines checked lexical excerpts with editorial paraphrase; it is not a diplomatic transcription of full entries. No entire modern website content is reproduced.
- Perder: 2013 Stockholm thesis, printed pp. 41, 42 and 57, Tables 10, 13 and 16. Selected explicit donor citations; no fresh OCR. Original source licence not established here; only selected lexical facts and analysis are added.
- Seven existing source records retain their persistent IDs and full compiled provenance in the audit. Original survey-page visual verification remains outstanding; this supplement does not claim to have repeated it. Existing upstream licences and extraction manifests govern those installations; licence re-audit remains outstanding.
- Excluded: other dictionary entries, OPED alternate and inflected forms not needed as donor heads, unrelated homonyms, and speculative donor reconstructions. OPED sātəl's final plural noun cross-reference is excluded from the verbal selection, retained in XML. The remaining 13 approved proposals (18 Dameli records) retain explicit donor/base blockers in the review. Five win/want records remain held.

## Transcription, metadata and graph

NFC preservation is deliberate. Selected donor display spellings are already audited; native spelling is a separate field and donor spelling must not be fed through the cited survey's IPA decoder again. `conversion/dameli-donors.txt`, `utils.py` and a narrow filename route in `make_cldf.py` preserve all input symbols, meaningful spaces and stress. The regression checks all 33 routed forms exactly. Raw survey transcription and phonemic evidence remain in the audit; no reconstructed pronunciation is invented for the new heads.

Source grammar is represented in canonical tags, including sense-level adjective/noun labels where included; the eight Perder donor POS labels are editorially inferred from table usage and glosses, explicitly identified as such in the audit; XML retains scope details. Dialect tags are reused from existing registered source records. No new language, dialect, coordinate, family or Glottocode is introduced. LSI/Hallberg prompt IDs and locations remain in citations and evidence; survey controls outside the selected donor records are excluded. No map point was inferred from a speaker name.

The real isolated build issued 33 new persistent donor identities. Only those 33 rows were appended to the latest shared registry, preserving all 849,866 previous rows. Another 22 previously installed donor heads were reused. Initial isolated output contains all 55 donor heads with matching language, form and `entry` status. Four heads in that pre-fix output lacked noun tags because of the double-conversion error; corrected compiled metadata assertions remain pending a successful rebuild. Later sense-tag corrections likewise await compilation. No shared generated CLDF or browser DB was replaced.

Saved 114 accepted overlay rows for 61 proposals / 106 Dameli records: 55 borrowing proposals and six co-wife compounds. Validation against actual build-issued nodes passed; a temporary graph changed 220 times (edges plus statuses), every requested edge was checked, and a second application changed zero. Preserved all 16,218 pre-existing overlay rows. See `curation/etymology-lab/Dm/donor-links-save-20260910.json`. Current cumulative overnight totals: 86/99 approved proposals, 166/184 records saved; 13 proposals/18 records blocked. Earlier numeral batch 43 is separate.

## Checklist gate disposition

1–5, source acquisition/layout/record parsing/language mapping: selected records audited and deterministically emitted; no new OCR or automatic headword matching. Full source coverage intentionally inapplicable. Survey original-page and licence re-audit are outstanding.

6–9, grammar/profile/reference/graph: source-backed grammar and native text emitted, explicit profile route tested, existing bibliography keys extended with supplement provenance, typed borrowing/components saved with uncertainty notes. The initial build generated references; later provenance edits require the eventual rebuild. Donor ancestry beyond the attested head is not invented.

10, audit: seeded 20-record frozen-evidence/output review (seed 20260911) found the sense-level grammar omissions, now corrected. Separate routing error has an exact all-row regression. Audit sample JSON records comparison scope and residual source-page/compiled checks; no claim of a fresh 0/20 original-PDF audit.

11–12, validation: source and profile focused checks pass. Two global dialect metadata tests fail (quality and missing coordinates). Initial isolated `make all` ran through references, then failed two final survey tests: Rajasthani expected 11,245 linked rows but got 11,256, plus existing source-owned overlay rows. Corrected rebuild could not write `cldf/forms.csv` because the disk was full. Full-suite results are recorded in the adjacent validation JSON. **Do not call this ingest fully validated.**

13, browser refresh: deferred explicitly by user. No browser database build, app inspection, or updated app-entry claim. Future representative IDs: bən `f_vxfcck45zjqh4`, ter `f_a7ttp4d7gm5vw`, zyā́t `f_ld6rhr4ok7efq`, sātəl `f_z3cplgqnyj7t6`.

14, handoff: README and triaged review updated. No commit, push, deployment or release. Next technical gate: enough free disk for a fresh isolated full build, then `DAMELI_COMPILED_CHECK=1` focused checks, reference checks and global failure review.
