"""Record final source-stage evidence, never claim deferred compiled gates."""
import hashlib
import json
from pathlib import Path
P=Path(__file__).resolve().parent;D=P.parents[4];W=D.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
validation=json.loads((P/'canonical-validation.json').read_text())
installation=json.loads((P/'source-stage-installation-result.json').read_text())
assert installation['installed_sha256']==validation['canonical_sha256']
for name,digest in validation['canonical_sha256'].items():assert sha(D/name)==digest,name
logs={name:W/'tmp/kholosi-recovery'/name for name in ('installed-tests.log','registry-validation.log','registered-repeat-install.log','canonical-validation.log')}
assert '21 passed' in logs['installed-tests.log'].read_text()
assert 'source files with settings' in logs['registry-validation.log'].read_text()
assert 'source-stage files installed' in logs['registered-repeat-install.log'].read_text()
report=dict(status='whole-source stage installed and focused gates passed; full ingestion gates deferred',
 rows=1135,audit_units=1110,legacy_keys=198,new_keys=937,historical_assignment_corrections=5,
 source_scope=dict(pdf_heads=448,pdf_subentries=70,lexicon_records=401,sentence_records=116,
   quoted_embedded_occurrences=29,morphology_occurrences=15,morphology_comparison_controls=6,
   whole_untranslated_text=1,acoustic_controls=3,separate_historical_witnesses=21),
 exclusions=['Three entirely blank sentence CSV rows','Nonlexical acoustic tracks and explicitly identified comparison/phonological controls'],
 unresolved=dict(uncertainty_tagged_rows=107,blank_gloss_rows=25,
   description='Source uncertainties, alternative ancestry, unresolved donor targets, unglossed responses/text and morphology without lexical definitions remain explicit; ke has no unsupported ancestry link.'),
 transcription='Printed literal orthography and supplied IPA remain separate; source-specific display profile, NFC/NFD parity, no phonetic inference or OCR.',
 graph=dict(ancestry_linked_rows=380,borrowing_tagged_rows=317,variant_links=35,derivation_parent_links=90,
   claim_statements=830,all_source_keys_closed=True,serializer_addendum='Six multi-parent delimiters repaired to compiler pipe syntax; historical r3 evidence retained',reference_provenance='mixed'),
 independent_audit=dict(report='independent-r3-original-review.json',fresh_rows=20,material_errors=0,
   targeted_groups=12,targeted_rows=13,complete_donor_claim_census=830,
   historical_report='independent-historical-link-review-r3.json'),
 validation=dict(focused_tests=21,actual_canonical_parser_rows=1135,NFC_NFD_rows_each=1135,
   registered_importer_repeat_byte_parity=True,registry_valid=True,all_198_existing_identity_records_retained=True),
 canonical_sha256=validation['canonical_sha256'],
 evidence_sha256={name:sha(P/name) for name in ('whole-proposal-r3/review-manifest.json','independent-r3-original-review.json',
   'independent-full-donor-census-review-r3.json','independent-r3-metadata-addendum-review.json',
   'source-stage-installation-approval.json','source-stage-importer-routing-approval.json','canonical-validation.json','source-stage-parent-delimiter-addendum/root-approval.json','source-stage-parent-delimiter-addendum/review.json')},
 log_sha256={str(path.relative_to(W)):sha(path) for path in logs.values()},
 representative_source_entries=[dict(key='kholosi:pdf:p1:c1:u1:sense:1',form='aber',gloss='cloud'),
   dict(key='kholosi:pdf:p1:c1:u20:sense:1',form='avval',gloss='first'),
   dict(key='kholosi:pdf:p1:c2:u11:sense:1',form='barg',gloss='leaf')],
 app_entries='No new app/browser claim; representative source keys above await the authorized future build.',
 deferred=['Full database/CLDF build','Compiled durable identity and graph reconciliation','Full suite','Browser QA'],
 no_remote_or_publication=True)
(P/'source-stage-final-handoff.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k in ('status','rows','audit_units','legacy_keys','new_keys','canonical_sha256','deferred')}))
