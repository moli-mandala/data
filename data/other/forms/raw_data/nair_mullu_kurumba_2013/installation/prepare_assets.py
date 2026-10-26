"""Stage own assets; shared baseline capture requires a reserved root window."""
import argparse
import hashlib
import json
from pathlib import Path
import yaml
from pybtex.database import parse_string

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
DATA=next(p for p in HERE.parents if (p/'source_meta.py').exists())
ROOT=DATA.parent
STEM='20260927-nair-mullu-kurumba'


def sha(path):
    if not path.exists(): return None
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): result.update(chunk)
    return result.hexdigest()


def main(accepted=False,capture_shared=False):
    assert not (HERE/'root-acceptance.json').exists(), 'Activated inputs are immutable'
    freeze=json.loads((PACKAGE/'review-freeze.json').read_text())
    assert sha(PACKAGE/'review-freeze.json')=='815bd4883bd4ad3b0dfcf368f33e451dcd8eab76d8e2243912d410e309acca57'
    for name,value in freeze['files'].items(): assert sha(ROOT/name)==value,name
    review_path=PACKAGE/'independent-review/final-original-output-review-r2-20260927571.json'
    assert sha(review_path)=='9f9ce0925f30ca1859db677a959d3fc5baf8c7c1dde7ebc08d95e66cdc5bcc38'
    review=json.loads(review_path.read_text())
    assert review['material_errors']==0 and review['sample_pass']==20
    for name,value in review['pins'].items(): assert sha(PACKAGE/name)==value
    acceptance_path=PACKAGE/'root-source-stage-acceptance-r2.json'
    if accepted:
        assert sha(acceptance_path)=='3ea8a8f189538ac4b1c35701170eb1dc0e0a1220b5e9fe1f06f5816a6ac06040'
        acceptance=json.loads(acceptance_path.read_text())
        assert acceptance['material_errors']==0 and acceptance['rows']==135
        assert acceptance['csv_sha256']==sha(PACKAGE/'proposed.csv')
        assert acceptance['audit_sha256']==sha(PACKAGE/'proposal-audit.jsonl')
    settings=yaml.safe_load((PACKAGE/'proposed-settings.yaml').read_text())
    settings['defaults']['importer']={'commands':[['data/other/forms/raw_data/nair_mullu_kurumba_2013/installation/install_source.py','--install']],
        'note':'Guarded complete reviewed chapter only. Frozen first readings and revision1 are historical; source-stage compiled/browser gates remain separate.'}
    settings_path=HERE/(STEM+'.yaml')
    settings_path.write_text(yaml.safe_dump(settings,allow_unicode=True,sort_keys=False))
    audit=[]
    for raw in map(json.loads,(PACKAGE/'proposal-audit.jsonl').read_text().splitlines()):
        audit.append(dict(audit_key=raw['source_unit_key'],source_unit_key=raw['source_unit_key'],
            printed_pages=raw['printed_pages'],entry_keys=raw['entry_keys'],
            disposition='emitted_or_reused' if raw['entry_keys'] else 'explicit_documentary_or_context',
            accepted_for_installation=accepted,
            acceptance_scope='Root source-stage decision acknowledged; canonical activation remains separately guarded' if accepted else 'Independent review passed; root source-stage decision pending',
            independent_acceptance_evidence=str(review_path.relative_to(PACKAGE)),
            independent_acceptance_sha256=sha(review_path),source_record=raw))
        if accepted:
            audit[-1]['root_acceptance_evidence']=acceptance_path.name
            audit[-1]['root_acceptance_sha256']=sha(acceptance_path)
    assert len(audit)==65
    audit_path=HERE/'accepted-audit-proposal.jsonl'
    audit_path.write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in audit))
    raw_bib=(PACKAGE/'proposed-references.bib').read_text()
    previous='data/other/forms/raw_data/nair_mullu_kurumba_2013/physical-units-first-reading.jsonl; data/other/forms/raw_data/nair_mullu_kurumba_2013/proposal-audit.jsonl'
    assert raw_bib.count(previous)==1
    staged=raw_bib.replace(previous,'data/other/forms/'+STEM+'.csv; data/other/forms/raw_data/nair_mullu_kurumba_2013/installed-audit.jsonl; '+previous)
    old_refs=parse_string(raw_bib,'bibtex').entries
    staged_refs=parse_string(staged,'bibtex').entries
    assert set(old_refs)==set(staged_refs)
    for key in old_refs:
        assert old_refs[key].persons==staged_refs[key].persons
        assert {k:v for k,v in old_refs[key].fields.items() if k!='provenance'}=={k:v for k,v in staged_refs[key].fields.items() if k!='provenance'}
    (HERE/'staged-references.bib').write_text(staged)
    targets={'csv':'data/other/forms/'+STEM+'.csv','yaml':'data/other/forms/'+STEM+'.yaml',
        'audit':'data/other/forms/raw_data/nair_mullu_kurumba_2013/installed-audit.jsonl','bib':'cldf/sources.bib'}
    planned={'csv':(PACKAGE/'proposed.csv').read_bytes(),'yaml':settings_path.read_bytes(),'audit':audit_path.read_bytes()}
    baselines={name:None for name in targets.values()}
    if capture_shared:
        # Caller must have an explicitly reserved shared window. This option is
        # never used during parallel installer preparation.
        assert sha(DATA/targets['bib'])=='13d183d0479219f8f9a4d94c2d006e8f9fd52779da62ac3a4e434e4423547367', 'Root-reserved shared baseline changed'
        current=(DATA/targets['bib']).read_bytes()
        existing=parse_string(current.decode(),'bibtex').entries
        assert not set(staged_refs)&set(existing)
        planned['bib']=current+(b'\n' if current.endswith(b'\n') else b'\n\n')+staged.encode()
        merged=parse_string(planned['bib'].decode(),'bibtex').entries
        assert all(merged[k]==v for k,v in existing.items())
        baselines={name:sha(DATA/name) for name in targets.values()}
    assets=HERE/'assets';assets.mkdir(exist_ok=True)
    descriptions={}
    for role,name in targets.items():
        local='installation/assets/'+role+'.asset'
        if role in planned: (PACKAGE/local).write_bytes(planned[role])
        descriptions[role]={'target':name,'input':local,'output_sha256':sha(PACKAGE/local) if role in planned else None}
    files={str((ROOT/name).relative_to(PACKAGE)):value for name,value in freeze['files'].items()}
    files.update({name:sha(PACKAGE/name) for name in ['review-freeze.json',str(review_path.relative_to(PACKAGE)),
        'installation/prepare_installer.py','installation/install_source.py','installation/prepare_assets.py',
        'installation/'+STEM+'.yaml','installation/staged-references.bib','installation/accepted-audit-proposal.jsonl']})
    if accepted: files[acceptance_path.name]=sha(acceptance_path)
    for asset in descriptions.values():
        if asset['output_sha256']: files[asset['input']]=asset['output_sha256']
    for name in ('installation/test_installer.py','installation/check_fixture.py','installation/actual-parser-check.json'):
        if (PACKAGE/name).exists(): files[name]=sha(PACKAGE/name)
    draft=dict(installation_approved=False,result='disabled; root activation required',rows=135,audit_units=65,
        witness_records=164,target_rows=110,comparison_rows=25,commentary_reuses=6,audit_only_witness_records=23,variants=11,
        shared_baseline_captured=capture_shared,audit_acceptance_ready=accepted,
        files=files,assets=descriptions,canonical_baselines=baselines,
        source_pdf_sha256=review['pins'].get('source_pdf_sha256',freeze['source_pdf_sha256']),
        durable_identity_sha256=sha(DATA/'data/form-identities.csv') if capture_shared else None,
        shared_baseline_status='captured in root-reserved window' if capture_shared else 'NOT captured: reserved window required',
        deferred=['full compiled build','full suite','compiled identity/graph/reference checks','browser QA'])
    (HERE/'root-acceptance-draft.json').write_text(json.dumps(draft,indent=2)+'\n')
    print(json.dumps({'disabled':True,'rows':135,'audit_units':65,'shared_baseline_captured':capture_shared,'audit_acceptance_ready':accepted}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accepted',action='store_true',help='Only after explicit root source-stage acceptance')
    parser.add_argument('--capture-shared',action='store_true',help='Only during explicit root-reserved shared write window')
    args=parser.parse_args();main(args.accepted,args.capture_shared)
