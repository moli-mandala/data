# Kharia Living Dictionary — ingestion review

The complete SOURCE_INGESTION_CHECKLIST.md is active. Addenda: dictionary/glossary
and website/API. Source reuse permission was confirmed by the user on 2026-09-11.
This is the Living Tongues/Kiro dictionary, not Peterson (2009); no Peterson
content or claims are silently attributed to it.

## Coverage and modelling

Public SQLite snapshot uploaded 2026-08-20, acquired 2026-09-11: all 452 entries
and 452 upstream senses become 521 rich rows. The increase is 68 printed
phonetic alternatives (including optional segments) and one additional numbered
meaning. Every upstream entry and sense ID is preserved. None of the lexical
entries is excluded. Six newer entries have no phonetic field: retain their
native-script spelling in Form and Original, blank Phonemic, and an uncertain
tag with the missing-transcription reason in the audit. Do not invent pronunciation.

Use canonical language kh. Dhelki and Dudh are registered dialects; the 441
Dhelki/Dudh attestations carry both tags. Two entries are Dhelki only, three Dudh
only, six unassigned. No dialect coordinates or dialect Glottocodes were supplied;
leave those blank instead of copying the dictionary's base-language map point.

The source makes no etymological claims. All 68 graph links are source-local
variants with existing targets; all other new nodes remain unlinked. Preserve
same-source homographs (e.g. ɖena 'wing' versus 'come') by upstream key. Sources,
linguistic-history, POS, morphology, and source relationships are empty upstream;
inferred POS, donors, genealogy, and morphological analyses are inapplicable.

## Transcription and extraction audit

No OCR: direct SQLite fields with integrity and immutable SHA-256 checks.
Native script is separate from source phonetic transcription. The explicit
kharia-living preservation profile retains IPA-like transcription, optional
variant readings, short-vowel breves, nasalization, aspiration, and distinct
ʔ/ˀ. Uninterpreted brevity/glottalization notation is marked uncertain. Every
NFC and NFD input symbol is covered; the six native-only entries use a tested
literal preservation fallback. Full unexpanded phonetic fields remain in audit.

The first 20-record audit exposed tilde/parenthesis alternatives, English
parenthetical comments inside phonetic fields, and numbered meanings. Those are
now separated and regression-tested. A fresh sample with seed 20260913 was
checked raw-to-output: 0/20 material extraction errors. First/last records,
all native-only records, all optional/tilde forms, and the ɖena homographs were
also inspected. Multilingual glosses and upstream semantic-domain IDs remain
in each audit record; English is the displayed definition.

438 audio records, 69 photos, speaker metadata, and three example sentences are
outside this lexical-headword import. No media files are fetched or redistributed.
Two source phonetic comments '(same as child)'/'(same as baby)' are usage Notes,
not phonemes or automatically accepted graph edges. No auxiliary work is cited.

## Validation status

Final focused source/profile/dialect/checklist/Sheth/Lāḷas suite: **52 passed**.
All 521 source keys resolve to distinct compiled nodes; all 68 expected variant
edges and no other new ancestry edges survive. 453 nodes remain unlinked. The
saved corpus has 408 concept links for this source. No source conversion errors.

All seven build stages completed; `make all` then failed two existing survey
assertions: 15,887 Rajasthani nodes versus expected 15,876, and source-owned survey
forms present in the etymology overlay. Full pytest first hit duplicate module
names; importlib-mode full suite: **1,737 passed, 18 skipped, 31 failed**. The
failures concern other source/graph/metadata expectations; they were not all
baseline-tested, so they are not collectively claimed to be pre-existing.
No Kharia-specific test failed. The mandatory full-validation gate remains open.

Generated counts: forms +521, edges +68, source keys +521, permanent identity
registry +521, references +1; concepts and alignments unchanged in count.
Aliases +6,535 reflect the shared identity reconciliation; no source keys were
reminted. A same-process comparison with/without Kharia yields 408 new concept
links and zero additions/removals among prior forms; aggregate saved-link changes
include the existing mapper’s cross-process variation. Full hashes/deltas and logs are in the adjacent audits directory.
The corporate-author short citation bug discovered during browser QA was fixed
in make_refs.py and covered by a regression test; L2026a now renders correctly.

An isolated browser database was built and inspected without replacing staged
release files. Integrity is ok; 99,975,168 bytes expanded (above the 97 MB warning),
44.12 MB compressed (passes the strict 50 MB limit). Source table: 521 forms.
Kharia page: 1,468 forms / 9 sources. Dhelki and Dudh each filter 512 forms and
say not located. Source search elephant returns only केलुंग; its entry has no
invented pronunciation. The ELEPHANT atlas has 210 forms and includes that entry
among 113 unetymologised attestations. Variant entry maʔgʰrel renders both dialect
tags and the correct makʰrel parent. Visual layout inspected.

Representative local app entries: f_dbhiyre32y4do (maʔgʰrel, January variant),
f_afl6zgm74xh2s (makʰrel, parent), f_d5dzsxi43euuw (केलुंग, native-only elephant),
f_vexng72lob2ja (kosor, dry). Not deployed; no commit/push/release performed.
This source is installed and locally inspected, but not fully validated.
