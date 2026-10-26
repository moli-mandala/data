"""Guarded Cholanaickan whole-chapter source-stage installation.

Default invocation is read-only. --install writes only four canonical files plus
source-local accepted audit/receipt. It never runs CLDF or browser database builds.
"""
import argparse,csv,hashlib,io,json,os,sys
from pathlib import Path
from pybtex.database import parse_string
HERE=Path(__file__).resolve().parent
DATA=next(p for p in HERE.parents if (p/'tags.py').exists())
sys.path.insert(0,str(DATA))
from source_meta import SourceMeta
WORKSPACE=DATA.parent
EXPECTED={'proposed.csv':'771abcdfe9f6e0c25198397c8afd37e4d592ab982b34e3d8be8781eaa582c308',
          'proposal-audit.jsonl':'a0f18aca7076d0141df0410262888e319ff308b17ee2522ffedcd7e6c97a8373',
          'physical-units-first-reading.jsonl':'7dce05411899a3e99dab489cc3df5f4b59f77e2848e7c4af4fde4fe623effab5',
          'graph-decisions-proposal.json':'6a2c53efb5afa8327da58ad529c06a7bfaff37d8d91d25dbe4059e29206e9733'}
AUDIT_REVIEW=HERE/'sampling-setup/final-freeze-original-audit-20260927338.json'
PDF=WORKSPACE/'tmp/pdfs/tribal-languages-kerala-nair2013.pdf'
PDF_SHA='4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af'
TARGETS={
 'csv':DATA/'data/other/forms/20260927-nair-cholanaickan.csv',
 'yaml':DATA/'data/other/forms/20260927-nair-cholanaickan.yaml',
 'languages':DATA/'cldf/languages.csv',
 'bib':DATA/'cldf/sources.bib',
 'audit':HERE/'installed-audit.jsonl'}
RECEIPT=HERE/'installation-receipt.json'
IDENTITY=DATA/'data/form-identities.csv'
def sha(b):return hashlib.sha256(b).hexdigest()
def current(p):return sha(p.read_bytes()) if p.exists() else None

def prepare():
    for name,h in EXPECTED.items():assert current(HERE/name)==h,(name,'frozen source changed')
    assert current(PDF)==PDF_SHA
    report=json.loads(AUDIT_REVIEW.read_text())
    assert report['material_errors']==0 and report['actual_csv_rows_checked']==20
    assert report['frozen_inputs']['proposed.csv']==EXPECTED['proposed.csv']
    assert report['frozen_inputs']['proposal-audit.jsonl']==EXPECTED['proposal-audit.jsonl']
    rows=list(csv.reader((HERE/'proposed.csv').open()))
    assert len(rows)==104 and len({r[10] for r in rows})==104
    assert all(len(r)==15 and r[0]=='Cholanaickan' and r[7].startswith('nair2013cholanaickan[') for r in rows)
    keys={r[10] for r in rows}
    assert sum(bool(r[11]) for r in rows)==11 and all(not r[11] or r[11] in keys for r in rows)
    audit=[json.loads(x) for x in (HERE/'proposal-audit.jsonl').read_text().splitlines()]
    assert len(audit)==117
    assert {a['proposed_entry_key'] for a in audit if a.get('proposed_entry_key')}==keys
    accepted=[{**a,'accepted_for_source_stage':True} for a in audit]
    audit_bytes=''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in accepted).encode()
    config=HERE/'20260927-nair-cholanaickan.yaml'
    meta=SourceMeta([config])
    assert 'nair2013cholanaickan' in meta.sources
    existing_lang=TARGETS['languages'].read_bytes()
    lang_rows=list(csv.DictReader(io.StringIO(existing_lang.decode())))
    assert not any(r['ID']=='Cholanaickan' for r in lang_rows)
    proposed_lang=list(csv.reader((HERE/'proposed-language.csv').open()))
    assert len(proposed_lang)==2 and proposed_lang[0]==list(lang_rows[0].keys()) and proposed_lang[1][0]=='Cholanaickan'
    lang_bytes=existing_lang
    extra=io.StringIO(newline='');csv.writer(extra).writerow(proposed_lang[1]);lang_bytes+=extra.getvalue().encode()
    existing_bib=TARGETS['bib'].read_bytes()
    before=parse_string(existing_bib.decode(),'bibtex').entries
    new=(HERE/'proposed-reference.bib').read_bytes();after=parse_string((existing_bib+b'\n'+new).decode(),'bibtex').entries
    assert set(after)-set(before)=={'nair2013cholanaickan','muralidharan1988cholanaikka'}
    assert all(after[k]==v for k,v in before.items())
    planned={'csv':(HERE/'proposed.csv').read_bytes(),'yaml':config.read_bytes(),'languages':lang_bytes,
             'bib':existing_bib+b'\n'+new,'audit':audit_bytes}
    baselines={role:current(path) for role,path in TARGETS.items()}
    assert baselines['csv'] is None and baselines['yaml'] is None and baselines['audit'] is None
    return planned,baselines,current(IDENTITY),{'languages':existing_lang,'bib':existing_bib}

def main(install=False):
    if RECEIPT.exists():
        receipt=json.loads(RECEIPT.read_text())
        assert all(current(TARGETS[k])==h for k,h in receipt['outputs'].items())
        assert current(IDENTITY)==receipt['durable_identity_sha256']
        print(json.dumps({'status':'already installed, exact byte no-op','rows':104}));return
    planned,baselines,identity,baseline_bytes=prepare()
    print(json.dumps({'status':'validated source-stage proposal','install':install,'rows':104,'audit_units':117,
                      'variant_edges':11,'baselines':baselines},indent=2))
    if not install:return
    # All dependencies and baseline hashes are checked before the first write.
    assert all(current(p)==baselines[k] for k,p in TARGETS.items())
    written=[]
    try:
        for role,path in TARGETS.items():
            assert current(path)==baselines[role]
            path.parent.mkdir(parents=True,exist_ok=True)
            temp=path.with_name(path.name+'.cholanaickan-install-tmp')
            assert not temp.exists()
            temp.write_bytes(planned[role]);os.replace(temp,path);written.append(role)
        assert all(current(p)==sha(planned[k]) for k,p in TARGETS.items()),'post-write target drift'
        assert current(IDENTITY)==identity,'durable identity drift'
        receipt={'status':'source-stage installed; full CLDF/browser gates deferred','rows':104,'audit_units':117,
                 'variant_edges':11,'frozen_inputs':EXPECTED,'source_pdf_sha256':PDF_SHA,
                 'outputs':{k:sha(planned[k]) for k in TARGETS},'baselines':baselines,'durable_identity_sha256':identity}
        temp=RECEIPT.with_suffix('.tmp');assert not temp.exists();temp.write_text(json.dumps(receipt,indent=2)+'\n');os.replace(temp,RECEIPT)
    except BaseException:
        # Restore only targets still equal to our proposed bytes; refuse to
        # overwrite a concurrent edit and leave evidence for manual recovery.
        for role in reversed(written):
            path=TARGETS[role]
            if current(path)!=sha(planned[role]):continue
            if baselines[role] is None:path.unlink()
            else:
                original=baseline_bytes.get(role)
                if original is None:raise AssertionError('Missing rollback baseline')
                temp=path.with_name(path.name+'.cholanaickan-rollback-tmp');temp.write_bytes(original);os.replace(temp,path)
        raise
    print(json.dumps({'status':'installed','rows':104,'audit_units':117,'variant_edges':11}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--install',action='store_true');main(parser.parse_args().install)
