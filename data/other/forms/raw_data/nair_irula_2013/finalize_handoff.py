"""Write unpinned installed handoff and append only this source's ledger row."""
import csv
import hashlib
import io
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
DATA=next(p for p in HERE.parents if (p/'source_meta.py').exists())
SHA=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=json.loads((HERE/'canonical-source-stage-validation.json').read_text())
assert report['status'].startswith('passed') and report['repeat_install']=='exact no-op'
receipt=HERE/'installation/installation-receipt.json'
review=HERE/'root-original-review-final-r2-20260927425.json'
assert json.loads(review.read_text())['material_errors']==0
manifest=json.loads((HERE/'review-manifest-r2.json').read_text())
assert all(SHA(HERE/n)==v for n,v in manifest['sha256'].items()),'frozen lexical input drift'
handoff='''# Irula source-stage handoff

The complete Nair 2013 Irula chapter, printed/PDF pages 22–27, is installed as `20260927-nair-irula.csv`. All 261 physical units are accounted: 245 target units yield 256 direct readings; 16 units remain audit-only. Controls include explicit Old Tamil comparisons, zero exponents, documentary contexts, reported bibliography and an expressly rejected noun compound. Every physical unit, repeated occurrence and raw scholarly claim is preserved in the accepted audit.

Literal Roman notation, en dashes, internal spacing and three separately drawn lowlines are preserved through `convert: false`. Native and Phonemic remain blank. One visible i-diaeresis/text-layer Cyrillic-character discrepancy and one unexplained trailing backslash retain typed source-notation uncertainty. Source contradictions remain literal. The unheaded personal-ending columns have no inferred singular/plural labels. Existing Irula/irul1243 and its registry metadata remain unchanged; no lect or coordinates are invented. The pinned shared PDF has no explicit reuse licence and remains local.

Twelve local alternate pointers cover one explicitly freely varying conditional pair and eleven same-slot source-listed realizations. Their source conditioning is retained; no free interchangeability or cross-group relation is asserted. Four source-supported participial-noun derivations point to printed relative-participle exemplars. Feminine singular `iva` remains distinct from the neuter plural alternate. Reported stem-tense conditioning is in Notes rather than intrinsic tense tags on temporal suffixes. Permissive, injunctive, concessive and polite categories have synchronized backend/frontend registration.

The final independent fresh sample passed 20/20 after all six original pages and source scope were reread. Exact r1-to-r2 continuity records only 26 Notes, eleven alternate pointers and eight Tags changes; all other CSV fields and keys remained exact. Eight source/parser checks and eight isolated journal/recovery installer checks passed. The guarded four-target installation and exact repeat no-op passed. The actual canonical parser, without monkeypatching, passes all 256 rows in canonical, NFC and NFD inputs: every fifteen-column field and display Form is preserved, zero conversion, 256 distinct initial source entry keys, twelve alternate and four derived endpoints closed. All three references are declared once and format correctly; 627 unrelated bibliography records remain semantically unchanged.

The installation-only metadata addendum adds the guarded importer, canonical provenance and primary `etymology_provenance: source` for four source-explicit internal derivations. Auxiliary works are source-reported, not independently collated, and retain provenance `none`. The original r2 YAML/Bib and README/SCOPE remain frozen historical preparation records; this handoff and installed metadata describe current state.

Three verifier-only API/path assumptions were corrected during installed checks: normalized transcription rules use the public `transcription` API, parser copies use the required manual `other/forms` path, and the `.asset` backup specifies BibTeX format. Failed attempt evidence is retained. No canonical source bytes changed in these corrections.

## Exact evidence

'''
for label,path in [('Canonical CSV',DATA/'data/other/forms/20260927-nair-irula.csv'),('Canonical YAML',DATA/'data/other/forms/20260927-nair-irula.yaml'),('Accepted audit',HERE/'installed-audit.jsonl'),('Installation receipt',receipt),('Canonical validation',HERE/'canonical-source-stage-validation.json'),('Final independent review',review)]:
    handoff+=f'- {label}: `{path.relative_to(DATA)}` — SHA-256 `{SHA(path)}`.\n'
handoff+=f'- Durable identity SHA-256 remained `{report["durable_identity_sha256"]}`.\n\nThe full CLDF/database build, compiled identity/graph/reference survival, full suite and browser QA/app-entry verification are explicitly deferred by the authorized source-stage scope. No full production completion, release or publication is claimed. The shared bibliography window was released after installed validation.\n'
(HERE/'SOURCE_STAGE_HANDOFF.md').write_text(handoff)
checklist='''# Nair 2013 Irula source-stage acceptance

Complete pp. 22–27: 261 physical units, 256 Irula readings, 16 audit-only units, twelve local alternate pointers and four source-explicit derived relations. Original controls, repeated occurrences, raw claims, literal typography and two typed source-notation uncertainties are preserved. Existing Irula/irul1243 metadata is unchanged. Three references are registered with editor/OCR/provenance flags; auxiliary publications are attributed background, not independently collated.

Final independent original-page/source-scope review and fresh 20/20 sample passed. Eight focused source/parser checks and eight isolated journal/recovery transaction checks passed. Actual registered canonical SourceMeta and parser checks pass all fifteen fields in canonical/NFC/NFD input, zero conversion and 256 unique initial entry dedupe keys. All graph endpoints resolve; exact repeat install is a no-op. Durable identities and 627 unrelated bibliography records are preserved.

Evidence: `data/other/forms/raw_data/nair_irula_2013/SOURCE_STAGE_HANDOFF.md`, `canonical-source-stage-validation.json`, `installation/installation-receipt.json`, and `root-original-review-final-r2-20260927425.json`. Frozen preparation records remain immutable; the installed handoff describes current state.

Deferred: coordinated full CLDF/database build, compiled identity/graph/reference checks, full suite, browser QA and representative app entries. Source stage is complete; full production ingestion acceptance is not claimed.
'''
(DATA/'source_checklists/20260927-nair-irula-source-stage.md').write_text(checklist)
ledger=DATA/'source_checklists/20260925-source-completeness-ledger.tsv'
before=ledger.read_bytes();records=list(csv.DictReader(io.StringIO(before.decode()),delimiter='\t'))
assert all(r['package']!='20260927-nair-irula' for r in records),'ledger row already exists; refuse duplicate'
row=['20260927-nair-irula','data/other/forms/raw_data/nair_irula_2013','complete_chapter_in_shared_book','complete_source_stage','asur_finish','Complete pp. 22–27: 261 physical units, 256 Irula readings, 16 audit-only units, 12 local variants and 4 source-explicit derived links. Final independent originals/fresh 20/20 passed; 8 source checks and 8 transaction fixtures passed; real canonical/NFC/NFD all15fields, 256 entry dedupe keys, references/graph closure and guarded no-op pass. Existing Irula identity and durable IDs unchanged. Evidence SOURCE_STAGE_HANDOFF.md and canonical-source-stage-validation.json. Full CLDF/DB, compiled identity/graph/reference, full suite/browser gates deferred; no whole-book completion claim.']
buf=io.StringIO(newline='');csv.writer(buf,delimiter='\t',lineterminator='\n').writerow(row)
assert ledger.read_bytes()==before,'concurrent ledger drift'
ledger.write_bytes(before+(b'' if before.endswith(b'\n') else b'\n')+buf.getvalue().encode())
(HERE/'ledger-handoff-evidence.json').write_text(json.dumps({'status':'only source row appended; prior bytes preserved','ledger':str(ledger.relative_to(DATA)),'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':SHA(ledger),'rows_before':len(records),'rows_after':len(records)+1,'source':'20260927-nair-irula'},indent=2)+'\n')
print('Wrote source-stage handoff, checklist and one ledger row; frozen inputs unchanged.')
