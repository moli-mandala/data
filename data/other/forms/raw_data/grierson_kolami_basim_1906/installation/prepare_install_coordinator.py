"""Generate a disabled six-target coordinator from the reviewed Mullu transaction core.

No assets or shared baselines are captured here. Final pins/tests require accepted
whole-output review. Generated code is standalone and reviewable before activation.
"""
from pathlib import Path
import hashlib
HERE=Path(__file__).resolve().parent
DATA=next(p for p in HERE.parents if (p/'source_meta.py').exists())
TEMPLATE=DATA/'data/other/forms/raw_data/nair_mullu_kurumba_2013/installation/install_source.py'
HEADER='''"""Disabled six-target Kolami/Basim source-stage transaction. Root activates exact pins.
Dry run is default; no build, durable-ID mutation, remote operation, or publication.
"""
import argparse,csv,hashlib,io,json,sys
from pathlib import Path
import yaml
from pybtex.database import parse_string
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
DATA=next(p for p in HERE.parents if (p/'source_meta.py').exists())
sys.path.insert(0,str(DATA))
from source_meta import SourceMeta
STEM='20260927-grierson-kolami-basim'
APPROVAL=HERE/'root-acceptance.json'
TARGETS={'csv':f'data/other/forms/{STEM}.csv','yaml':f'data/other/forms/{STEM}.yaml',
 'audit':'data/other/forms/raw_data/grierson_kolami_basim_1906/installed-audit.jsonl',
 'profile':'conversion/grierson-kolami-basim-1906.txt','bib':'cldf/sources.bib','dialects':'cldf/dialects.csv'}
def digest(content):return hashlib.sha256(content).hexdigest()
def file_hash(path):
    if not path.exists():return None
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def contained(root,name):
    path=(root/name).resolve();assert path.is_relative_to(root.resolve()),('escaping path',name)
    return path
def registry(content):
    reader=csv.DictReader(io.StringIO(content.decode()));rr=list(reader)
    assert all(None not in r and r.get('ID') for r in rr)
    assert len({r['ID'] for r in rr})==len(rr)
    return reader.fieldnames,{r['ID']:r for r in rr}
def validate_inputs(check_draft=False,check_identity=True):
    a=json.loads((HERE/'root-acceptance-draft.json' if check_draft else APPROVAL).read_text())
    assert a['installation_approved'] is (False if check_draft else True)
    assert check_draft or a['result']=='pass'
    assert a.get('shared_baseline_captured') is True and a.get('audit_acceptance_ready') is True
    assert set(a['assets'])==set(TARGETS) and set(a['canonical_baselines'])==set(TARGETS.values())
    assert 'installation/install_source.py' in a['files']
    assert {'installation/baseline-bibliography.bib','installation/baseline-dialects.csv','installation/staged-references.bib','installation/staged-dialects.csv'}<=set(a['files'])
    assert a['files']['installation/baseline-bibliography.bib']==a['canonical_baselines'][TARGETS['bib']]
    assert a['files']['installation/baseline-dialects.csv']==a['canonical_baselines'][TARGETS['dialects']]
    for name,pin in a['files'].items():assert file_hash(contained(PACKAGE,name))==pin,('approved input drift',name)
    for role,asset in a['assets'].items():
        assert asset['target']==TARGETS[role] and asset['input'] in a['files']
        assert asset['output_sha256']==a['files'][asset['input']]
    for name,pin in a['original_inputs'].items():assert file_hash(contained(DATA.parent,name))==pin,('original drift',name)
    if check_identity:assert file_hash(DATA/'data/form-identities.csv')==a['durable_identity_sha256']
    if check_identity:
        for name,pin in a['canonical_reuse_inputs'].items():assert file_hash(contained(DATA,name))==pin,('reuse input drift',name)
    # Later unrelated source additions may change a reused source's whole file;
    # the three approved reused witnesses themselves must stay exactly intact.
    for witness in a['canonical_reuse_rows']:
        with contained(DATA,witness['canonical_path']).open() as stream:
            matches=[r for r in csv.reader(stream) if r[10]==witness['entry_key']]
        assert matches==[witness['canonical_row']],('canonical reused witness drift',witness['entry_key'])
    return a
def validate_assets(a,outputs):
    rr=list(csv.reader(io.StringIO(outputs['csv'].decode())))
    assert len(rr)==a['rows']==827 and all(len(r)==15 and r[0] and r[2] and r[10] for r in rr)
    keys={r[10] for r in rr};assert len(keys)==827
    assert all(not r[4] and not r[5] for r in rr)
    assert sum(bool(r[11]) for r in rr)==15 and sum(bool(r[13]) for r in rr)==2
    assert all(k in keys for r in rr for k in [r[11],*r[13].split('|')] if k)
    adjacency={r[10]:[k for k in [r[11],*r[13].split('|')] if k] for r in rr}
    active=set();done=set()
    def visit(k):
        assert k not in active,('cycle',k)
        if k in done:return
        active.add(k)
        for target in adjacency[k]:visit(target)
        active.remove(k);done.add(k)
    for k in keys:visit(k)
    audit=list(map(json.loads,outputs['audit'].decode().splitlines()))
    assert len(audit)==a['audit_units']==1570
    assert all(r['accepted_for_installation'] is True and 'source_record' in r for r in audit)
    assert {k for r in audit for k in r['entry_keys']}==keys|set(a['reused_existing_keys'])
    assert len(a['reused_existing_keys'])==3 and not set(a['reused_existing_keys'])&keys
    settings=yaml.safe_load(outputs['yaml'])
    assert settings['file']==STEM+'.csv'
    assert settings['defaults']['transcription']=={'profile':'grierson-kolami-basim-1906','input':'form','preserve_hyphens':True}
    assert settings['defaults']['identity']['dedupe_by_entry_key'] is True
    assert settings['defaults']['importer']['commands']==[['data/other/forms/raw_data/grierson_kolami_basim_1906/installation/install_source.py','--install']]
    assert settings['sources']['grierson1906kolamibasim']['forms']['split_alternates'] is False
    assert settings['sources']['grierson1906kolamibasim']['gloss']['grammar_tags'] is False
    refs=parse_string((HERE/'staged-references.bib').read_text(),'bibtex').entries
    assert set(settings['sources'])==set(refs)=={'grierson1906kolamibasim','haig1897gondi_kolami'}
    for k,v in settings['sources'].items():
        assert v['reference']=={'editor':'OpenAI Codex','ocr':k=='grierson1906kolamibasim','etymology_provenance':'source' if k=='grierson1906kolamibasim' else 'none'}
    meta=SourceMeta([HERE/(STEM+'.yaml')]);assert set(meta.sources)==set(refs)
    bib=parse_string(outputs['bib'].decode(),'bibtex').entries
    assert all(bib[k]==v for k,v in refs.items())
    prior=parse_string((HERE/'baseline-bibliography.bib').read_text(),'bibtex').entries
    assert not set(prior)&set(refs) and set(bib)==set(prior)|set(refs)
    assert all(bib[k]==v for k,v in prior.items()),'unrelated bibliography lost or changed'
    _,dialects=registry(outputs['dialects']);_,expected=registry((HERE/'staged-dialects.csv').read_bytes())
    assert len(expected)==2 and all(dialects[k]==v for k,v in expected.items())
    header,prior_dialects=registry((HERE/'baseline-dialects.csv').read_bytes())
    output_header,_=registry(outputs['dialects'])
    assert output_header==header and not set(prior_dialects)&set(expected)
    assert set(dialects)==set(prior_dialects)|set(expected)
    assert all(dialects[k]==v for k,v in prior_dialects.items()),'unrelated dialect lost or changed'
def shared_membership():
    expected=parse_string((HERE/'staged-references.bib').read_text(),'bibtex').entries
    actual=parse_string((DATA/TARGETS['bib']).read_text(),'bibtex').entries
    assert all(k in actual and actual[k]==v for k,v in expected.items()),'shared bibliography contribution drift'
    _,expected_dialects=registry((HERE/'staged-dialects.csv').read_bytes())
    _,actual_dialects=registry((DATA/TARGETS['dialects']).read_bytes())
    assert all(k in actual_dialects and actual_dialects[k]==v for k,v in expected_dialects.items()),'shared dialect contribution drift'
    a=json.loads(APPROVAL.read_text())
    assert file_hash(DATA/TARGETS['profile'])==a['assets']['profile']['output_sha256'],'own profile drift'
def plan(a):
    outputs={role:contained(PACKAGE,asset['input']).read_bytes() for role,asset in a['assets'].items()}
    validate_assets(a,outputs)
    assert all(digest(outputs[role])==a['assets'][role]['output_sha256'] for role in TARGETS)
    return outputs
'''
def main():
    source=TEMPLATE.read_text()
    core=source[source.index('def journal_state('):]
    core=core.replace('.mullu-', '.kolami-basim-').replace("'rows': 135", "'rows': 827").replace("'audit_units': 65", "'audit_units': 1570").replace("'targets': 4", "'targets': 6")
    # Own-byte no-op must include the profile; shared Bib/dialects use membership.
    core=core.replace("for role in ('csv', 'yaml', 'audit'):","for role in ('csv', 'yaml', 'audit', 'profile'):")
    output=HERE/'install_source.py';output.write_text(HEADER+'\n'+core)
    (HERE/'transaction-template-pin.json').write_text(__import__('json').dumps({'template':str(TEMPLATE),'sha256':hashlib.sha256(source.encode()).hexdigest(),'generated_coordinator_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'installation_approved':False},indent=2)+'\n')
    print('Disabled coordinator generated; final approvals/assets/baselines and isolated tests pending.')
if __name__=='__main__':main()
