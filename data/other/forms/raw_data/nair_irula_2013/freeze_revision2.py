"""Verify exact r1-to-r2 continuity and freeze reviewed source-local assets."""
import collections
import csv
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
SHA=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old={r[10]:r for r in csv.reader((HERE/'revision1-before-metadata-refinement/proposed.csv').open())}
new={r[10]:r for r in csv.reader((HERE/'proposed.csv').open())}
assert old.keys()==new.keys() and len(new)==256
expected={k:list(r) for k,r in old.items()}
columns={'Tags':14,'Notes':6,'Variant_Of_Key':11}
for name in ['metadata-revisions-r2.json','scope-revisions-r2.json']:
    for c in json.loads((HERE/name).read_text())['changes']:
        r=expected[c['entry_key']];i=columns[c['field']]
        assert r[i]==c['old'],(name,c['entry_key'],'baseline mismatch')
        r[i]=c['new']
assert expected==new,'unapproved output difference'
delta=[{'entry_key':k,'field':next(n for n,i in columns.items() if i==j),'old':old[k][j],'new':new[k][j]} for k in old for j in range(15) if old[k][j]!=new[k][j]]
assert all(old[k][j]==new[k][j] for k in old for j in range(15) if j not in columns.values())
first=HERE/'physical-units-first-reading.jsonl'
assert SHA(first)==json.loads((HERE/'revision1-before-metadata-refinement/review-manifest-r1.json').read_text())['sha256']['physical-units-first-reading.jsonl']
report={'status':'passed exact authorized revision continuity','r1_csv_sha256':SHA(HERE/'revision1-before-metadata-refinement/proposed.csv'),'r2_csv_sha256':SHA(HERE/'proposed.csv'),'r2_audit_sha256':SHA(HERE/'proposal-audit.jsonl'),'rows':256,'keys_unchanged':True,'literal_gloss_source_identity_and_derived_fields_unchanged':True,'physical_source_evidence_unchanged':True,'net_changed_field_counts':dict(collections.Counter(c['field'] for c in delta)),'changes':delta,'temporal_review':'Six intermediate suffix tense Tags reverted after independent review; Notes preserve stem conditioning and reported Zvelebil attribution.','graph_policy':'Eleven new same-slot pointers, with distinct conditioning groups and no free-interchangeability claim; existing free conditional pair retained.'}
(HERE/'revision2-continuity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
tests=(HERE/'focused-tests-r2.log').read_text()
assert '8 passed' in tests and 'failed' not in tests
names=list(json.loads((HERE/'review-manifest-r1.json').read_text())['sha256'])+['metadata-revisions-r2.json','scope-revisions-r2.json','same-cell-slash-graph-census.json','revision2-continuity.json','install_source.py','prepare_scope_revision.py','freeze_revision2.py','README.md','SCOPE.md']
manifest={'status':'frozen source-local revision2; fresh independent audit pending','source_pdf_sha256':'4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af','sha256':{n:SHA(HERE/n) for n in names},'counts':json.loads((HERE/'proposal-status.json').read_text()),'focused_tests':tests.strip(),'remaining_gates':['fresh independent original/output review','guarded source-stage installation','full CLDF/DB/compiled/full-suite/browser validation deferred']}
dest=HERE/'review-manifest-r2.json';assert not dest.exists(),'freeze is immutable'
dest.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'manifest_sha256':SHA(dest),'net_changed_field_counts':report['net_changed_field_counts'],'counts':manifest['counts']},indent=2))
