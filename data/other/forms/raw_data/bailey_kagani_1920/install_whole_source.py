"""Guarded r2 source-stage installer. No CLDF/database build or release action."""
import argparse,csv,hashlib,json,shutil,subprocess,sys,tempfile,re
from pathlib import Path
P=Path(__file__).resolve().parent;W=P/'whole-source-recovery';D=P.parents[4]
F=W/'whole-proposal-r2';R1=W/'whole-proposal-r1';FINAL=W/'uncertainty-addendum'
FREEZE_SHA='cbc675533d0baee66a1b450a83ca0807deb7a7b477c64dc599eaddd540b8d9a2'
CSV=D/'data/other/forms/20260925-bailey-kagani.csv';YAML=CSV.with_suffix('.yaml')
PROFILE=D/'conversion/bailey-kagani-1920.txt';ARCHIVE=P/'historical-glossary-installation'
BIB=D/'cldf/sources.bib';APPROVAL=F/'root-output-review-20260927.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())

def verify_frozen():
    assert sha(F/'review-manifest.json')==FREEZE_SHA,'r2 manifest changed'
    manifest=read(F/'review-manifest.json')
    for name,digest in manifest['output_sha256'].items():assert sha(F/name)==digest,name
    for name,digest in manifest['input_sha256'].items():assert sha(W/name)==digest,name
    assert manifest['rows']==1306 and manifest['audit_units']==1217
    for mode in ['NFC','NFD']:
        report=read(F/'parser-validation.json');assert report['checks'][mode]['rows']==1306 and report['checks'][mode]['errors']==0
    assert report['proposal_sha256']==sha(F/'proposal.csv') and report['profile_sha256']==sha(F/'profile-proposal.txt')
    assert read(F/'profile-policy-validation.json')['status']=='pass'
    return manifest

def verify_approval():
    manifest=verify_frozen();assert APPROVAL.exists(),'Accepted independent r2 report not present; installation blocked'
    approval=read(APPROVAL)
    assert approval['status']=='passed' and approval['material_errors']==0
    assert approval['sample_size']>=20 and approval['frozen_manifest_sha256']==FREEZE_SHA
    for name,digest in manifest['output_sha256'].items():assert approval['frozen_output_sha256'][name]==digest,name
    addendum=read(FINAL/'review-proposal.json');accepted=read(FINAL/'root-approval.json')
    assert accepted['status']=='approved' and accepted['base_freeze_sha256']==FREEZE_SHA
    assert accepted['review_proposal_sha256']==sha(FINAL/'review-proposal.json')
    assert accepted['approved_csv_sha256']==sha(FINAL/'proposal.csv')
    assert addendum['producer_sha256']==sha(W/'prepare_uncertainty_addendum.py')
    for name,digest in addendum['output_sha256'].items():assert sha(FINAL/name)==digest,name
    old=list(csv.reader((F/'proposal.csv').open()));new=list(csv.reader((FINAL/'proposal.csv').open()))
    changed=[(a,b) for a,b in zip(old,new) if a!=b]
    assert len(old)==len(new)==1306 and len(changed)==1
    a,b=changed[0];assert a[10]==b[10]==accepted['entry_key']
    assert {i for i,(x,y) in enumerate(zip(a,b)) if x!=y}=={6,14}
    return approval

def origin(name,digest):
    path=(D/name) if name.startswith(('data/','conversion/')) else W/name
    if path.exists() and sha(path)==digest:return path
    archived=ARCHIVE/path.name
    assert archived.exists() and sha(archived)==digest,f'Missing historical input: {name}'
    return archived

def regenerate():
    verify_frozen();one=read(R1/'review-manifest.json')
    scratch=D.parent/'tmp/kagani-whole';scratch.mkdir(exist_ok=True,parents=True)
    with tempfile.TemporaryDirectory(prefix='guarded-regeneration-',dir=scratch) as name:
        root=Path(name);sd=root/'data';sp=sd/'data/other/forms/raw_data/bailey_kagani_1920';sw=sp/'whole-source-recovery';sw.mkdir(parents=True)
        for item,digest in one['input_sha256'].items():
            target=(sd/item) if item.startswith(('data/','conversion/')) else sw/item
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(origin(item,digest),target)
        subprocess.run([sys.executable,str(sw/'prepare_integration_scaffold.py')],cwd=sd,check=True,capture_output=True,text=True)
        compared={}
        for item in ['proposal.csv','proposal-audit.jsonl','profile-proposal.txt']:
            actual=sw/'integration-working'/item;assert sha(actual)==sha(R1/item),f'r1 regeneration mismatch: {item}';compared['r1/'+item]=sha(actual)
        # Preserve the failed historical freeze as exact input to the guarded revision.
        sr1=sw/'whole-proposal-r1';sr1.mkdir()
        for item in list(one['output_sha256'])+['review-manifest.json','root-output-review-20260927.json']:
            shutil.copyfile(R1/item,sr1/item)
        two=read(F/'review-manifest.json')
        for item,digest in two['input_sha256'].items():
            source=W/item;assert sha(source)==digest
            target=sw/item;target.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(source,target)
        subprocess.run([sys.executable,str(sw/'prepare_revision2.py')],cwd=sd,check=True,capture_output=True,text=True)
        for item in ['proposal.csv','proposal-audit.jsonl','profile-proposal.txt']:
            actual=sw/'revision2-working'/item;assert sha(actual)==sha(F/item),f'r2 regeneration mismatch: {item}';compared['r2/'+item]=sha(actual)
        sf=sw/'whole-proposal-r2';sf.mkdir()
        for item in list(two['output_sha256'])+['review-manifest.json']:shutil.copyfile(F/item,sf/item)
        proof=read(FINAL/'review-proposal.json');assert proof['producer_sha256']==sha(W/'prepare_uncertainty_addendum.py')
        shutil.copyfile(W/'prepare_uncertainty_addendum.py',sw/'prepare_uncertainty_addendum.py')
        subprocess.run([sys.executable,str(sw/'prepare_uncertainty_addendum.py')],cwd=sd,check=True,capture_output=True,text=True)
        for item in ['proposal.csv','proposal-audit.jsonl','profile-proposal.txt']:
            actual=sw/'uncertainty-addendum'/item;assert sha(actual)==sha(FINAL/item),f'addendum regeneration mismatch: {item}';compared['approved-addendum/'+item]=sha(actual)
    report={'status':'passed byte-identical r1 plus r2 sandbox regeneration','freeze_sha256':FREEZE_SHA,'producer_sha256':sha(Path(__file__)),'compared':compared,'canonical_written':False}
    (W/'whole-regeneration-validation.json').write_text(json.dumps(report,indent=2)+'\n');return report

def bib_entry(text):
    start=text.index('@book{bailey1920kagani,');depth=0
    for i in range(text.index('{',start),len(text)):
        if text[i]=='{':depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0:return start,i+1,text[start:i+1]
    raise AssertionError('Unterminated targeted bibliography entry')

def metadata():
    import yaml
    plan=read(F/'whole-metadata-plan-20260927.json')
    ybase=ARCHIVE/YAML.name if (ARCHIVE/YAML.name).exists() else YAML
    assert sha(ybase)==plan['canonical_before_sha256']['data/data/other/forms/20260925-bailey-kagani.yaml']
    y=yaml.safe_load(ybase.read_text());y['defaults']['transcription'].update(plan['transcription'])
    y['defaults']['importer']={'commands':[['data/other/forms/raw_data/bailey_kagani_1920/install_whole_source.py','--install']],'note':'Independently approved1306-row whole Kāgānī source;329 historical keys retained with reviewed delta. Guarded r2 regeneration only. Historical import_source.py must not run against live whole-source files. Full DB/build/browser gates remain separately deferred.'}
    raw=BIB.read_text();start,end,current=bib_entry(raw)
    old=(W/'installation-bibliography-baseline.bib').read_text().rstrip('\n');new=old
    for field,value in plan['bibliography_change'].items():
        pattern=rf'(  {re.escape(field)}\s*=\s*\{{).*?(\}},?\n)'
        new,count=re.subn(pattern,lambda m:m[1]+value+m[2],new,flags=re.S);assert count==1,field
    assert current in {old,new},'Concurrent targeted bibliography edit; reconcile explicitly'
    return yaml.safe_dump(y,sort_keys=False,allow_unicode=True),raw[:start]+new+raw[end:]

def install():
    verify_approval();regenerate()
    rows=list(csv.reader((FINAL/'proposal.csv').open()));legacy=read(W/'legacy-baseline-20260927.json')
    assert len(rows)==1306 and len({r[10] for r in rows})==1306
    assert [r[10] for r in rows[:329]]==legacy['entry_keys']
    plan=read(F/'whole-metadata-plan-20260927.json')
    # Preflight every mutable target before writing any canonical file.
    for target,source in [(CSV,FINAL/'proposal.csv'),(PROFILE,FINAL/'profile-proposal.txt')]:
        key=str(target.relative_to(D.parent));assert sha(target) in {plan['canonical_before_sha256'][key],sha(source),sha(F/source.name)},str(target)
    historic_audit=read(R1/'review-manifest.json')['input_sha256']['../audit.jsonl']
    assert sha(P/'audit.jsonl') in {historic_audit,sha(F/'proposal-audit.jsonl'),sha(FINAL/'proposal-audit.jsonl')}
    yaml_text,bib_text=metadata()
    ykey=str(YAML.relative_to(D.parent))
    assert sha(YAML)==plan['canonical_before_sha256'][ykey] or YAML.read_text()==yaml_text
    ARCHIVE.mkdir(exist_ok=True)
    if not (ARCHIVE/'manifest.json').exists():
        assert sha(CSV)==legacy['csv_sha256'],'Missing first-install historical archive'
        files={}
        for source in [CSV,YAML,PROFILE,P/'audit.jsonl',P/'import_source.py']:
            target=ARCHIVE/source.name;shutil.copyfile(source,target);files[str(source.relative_to(D))]={'archived':target.name,'sha256':sha(target)}
        (ARCHIVE/'manifest.json').write_text(json.dumps(files,indent=2)+'\n')
    else:
        for record in read(ARCHIVE/'manifest.json').values():assert sha(ARCHIVE/record['archived'])==record['sha256']
    for source,target in [(FINAL/'proposal.csv',CSV),(FINAL/'proposal-audit.jsonl',P/'audit.jsonl'),(FINAL/'profile-proposal.txt',PROFILE)]:shutil.copyfile(source,target)
    YAML.write_text(yaml_text);BIB.write_text(bib_text)
    report={'status':'approved whole-source files installed; scoped canonical validation still required','rows':1306,'audit_units':1217,'legacy_keys':329,'freeze_sha256':FREEZE_SHA,'accepted_audit_sha256':sha(APPROVAL),'approved_uncertainty_addendum_sha256':sha(FINAL/'root-approval.json'),'installed_sha256':{str(q.relative_to(D)):sha(q) for q in [CSV,YAML,PROFILE,P/'audit.jsonl']},'deferred':['Full DB/CLDF build','Full suite','Compiled ID/graph validation','Browser QA']}
    (P/'whole-installation-result-20260927.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

def main():
    a=argparse.ArgumentParser();a.add_argument('--install',action='store_true');a.add_argument('--check-regeneration',action='store_true');args=a.parse_args()
    if args.install:install()
    elif args.check_regeneration:print(json.dumps(regenerate()))
    else:verify_frozen();print('Frozen r2 verified; no canonical writes and no approval inferred.')
if __name__=='__main__':main()
