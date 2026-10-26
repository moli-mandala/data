"""Guarded source-stage Koya installer; no CLDF/database generation."""
import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import re
import yaml
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
W = PACKAGE / 'whole-source-recovery'
FROZEN = W / 'literal-integration-metadata-r2'
DATA = PACKAGE.parents[4]
SCRATCH = DATA.parent / 'tmp/koya-install-validation'
# The full value is deliberately literal, rather than accepting a changed manifest.
MANIFEST_SHA = '1d220e49a1c33e752944825ef8767f29b746bd39937055e37817e7f710d46a6c'
STEMS = {'wordlist':'20260828-sil-koya','narrative':'20260927-sil-koya-narrative','comparative':'20260927-sil-koya-comparative'}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
def load(name):
    spec=importlib.util.spec_from_file_location('koya_install_'+name,W/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def verify_frozen():
    assert sha(FROZEN/'review-manifest.json') == MANIFEST_SHA
    assert sha(W/'durable-alias-review-20260927/proposed-source-key-alias.csv')=='420c7b1df1470489c212419efd3158966a74631a04df457a7c5dee05428d2b57'
    manifest=read(FROZEN/'review-manifest.json')
    for field,base in [('outputs_sha256',FROZEN),('scripts_sha256',W)]:
        for name,digest in manifest[field].items(): assert sha(base/name)==digest,(field,name)
    acceptance=read(FROZEN/'root-independent-output-audit-20260927.json')
    assert acceptance['manifest_sha256']==MANIFEST_SHA
    assert acceptance['source_units']>=20 and acceptance['material_errors']==0
    assert acceptance['status'].startswith('accepted independent')
    return manifest


def regenerate():
    """Run immutable generators in separate output dirs, compare all installable bytes."""
    verify_frozen();SCRATCH.mkdir(parents=True,exist_ok=True)
    base=load('prepare_literal_integration');base.OUT=SCRATCH/'literal';base.main()
    historical=read(W/'literal-integration/review-manifest.json')
    for name,digest in historical['outputs_sha256'].items():
        if name.startswith('proposed-') or name=='bibliography-update-proposal.json':
            assert sha(base.OUT/name)==digest,('historical regeneration',name)
    revision=load('prepare_metadata_revision');revision.OUT=SCRATCH/'r2';revision.main()
    manifest=verify_frozen()
    for name,digest in manifest['outputs_sha256'].items():
        if name.startswith('proposed-') or name in ['bibliography-update-proposal.json','metadata-repair-review.json']:
            assert sha(revision.OUT/name)==digest,('r2 regeneration',name)
    for name in ['validate_literal_integration','validate_literal_metadata']:
        validator=load(name);validator.OUT=revision.OUT
        if hasattr(validator,'SANDBOX'):validator.SANDBOX=SCRATCH/'parser'
        if hasattr(validator,'TMP'):validator.TMP=SCRATCH/'metadata'
        validator.main()
    verify_frozen()
    report={'status':'passed guarded regeneration and actual parser/metadata validation','manifest_sha256':MANIFEST_SHA,
            'rows':1665,'historical_freezes_unchanged':True,'parser':read(revision.OUT/'parser-validation.json'),
            'metadata':read(revision.OUT/'metadata-validation.json'),'canonical_modified':False}
    (W/'whole-regeneration-validation-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def mappings():
    pairs=[]
    for kind,stem in STEMS.items():
        pairs += [(FROZEN/f'proposed-{kind}.csv',DATA/f'data/other/forms/{stem}.csv'),
                  (FROZEN/f'proposed-{kind}.yaml',DATA/f'data/other/forms/{stem}.yaml')]
    for profile in ['literal','narrative','comparative']:
        pairs.append((FROZEN/f'proposed-sil-koya-{profile}.txt',DATA/f'conversion/sil-koya-{profile}.txt'))
    pairs += [(FROZEN/'proposed-whole-audit.jsonl',PACKAGE/'whole-audit.jsonl'),
              (W/'durable-alias-review-20260927/proposed-source-key-alias.csv',DATA/'data/other/form_aliases/20260927-koya.csv')]
    return pairs


def reconcile_metadata():
    # Routing-only addition to the accepted YAML; linguistic fields remain identical.
    for kind,stem in STEMS.items():
        target=DATA/f'data/other/forms/{stem}.yaml'
        proposed=yaml.safe_load((FROZEN/f'proposed-{kind}.yaml').read_text())
        proposed.setdefault('defaults',{})['importer']={
            'commands':[['data/other/forms/raw_data/sil_koya_2021/install_whole_source.py','--install']],
            'note':'Accepted whole-report recovery, 1665 rows across three files. Historical import_koya.py is a reconstruction fixture only. Guarded regeneration required; full build and browser gates deferred.'}
        target.write_text(yaml.safe_dump(proposed,allow_unicode=True,sort_keys=False))
    bib=DATA/'cldf/sources.bib';text=bib.read_text()
    key='devagnanavaram-et-al2021koya'
    start=text.index('@techreport{'+key+',');end=text.index('\n}',start)+2
    before=text[start:end];after=before
    fields=read(FROZEN/'bibliography-update-proposal.json')['fields']
    for field,value in fields.items():
        pattern=r'(?m)^(  '+re.escape(field)+r'\s*= \{).*?(\},?)$'
        after,n=re.subn(pattern,lambda m:m[1]+value+m[2],after)
        assert n==1,field
    if after!=before:
        historical=PACKAGE/'historical-wordlist-installation/source-reference.bib'
        if not historical.exists():historical.write_text(before+'\n')
        bib.write_text(text[:start]+after+text[end:])


def install():
    regenerate()
    archive=PACKAGE/'historical-wordlist-installation';archive.mkdir(exist_ok=True)
    manifest_path=archive/'manifest.json'
    if not manifest_path.exists():
        old=[]
        for _,target in mappings():
            if target.exists():
                dest=archive/target.name;shutil.copyfile(target,dest)
                old.append({'path':str(target.relative_to(DATA)),'archive':dest.name,'sha256':sha(dest)})
        manifest_path.write_text(json.dumps({'status':'historical pre-whole-source installation','files':old},indent=2)+'\n')
    for entry in read(manifest_path)['files']: assert sha(archive/entry['archive'])==entry['sha256']
    for source,target in mappings():
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        assert sha(source)==sha(target)
    reconcile_metadata()
    # Only the reviewed source-key relation is installed. Never edit compiled IDs/aliases.
    result={'status':'source files installed; canonical parser and full-build gates separately recorded',
            'rows':1665,'audit_units':2119,'preserved_active_legacy_keys':1437,'retired_source_key_aliases':1,
            'manifest_sha256':MANIFEST_SHA,'installed_sha256':{str(t.relative_to(DATA)):sha(t) for _,t in mappings()},
            'identity_policy':'UTN198 v2 source-key alias only; JAG198 active ID and unrelated positional alias untouched.',
            'deferred':['full CLDF build','compiled durable identity check','full tests','database build','browser QA']}
    (PACKAGE/'whole-installation-result-20260927.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--install',action='store_true');args=parser.parse_args()
    install() if args.install else regenerate()
