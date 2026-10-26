"""Disabled six-target Kolami/Basim source-stage transaction. Root activates exact pins.
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

def journal_state(approval, outputs):
    journal_path = HERE / 'installation-transaction.json'
    if journal_path.exists():
        journal = json.loads(journal_path.read_text())
        assert journal['approval_sha256'] == file_hash(APPROVAL)
        assert set(journal['targets']) == set(TARGETS.values())
        for role, name in TARGETS.items():
            record = journal['targets'][name]
            assert record['output_sha256'] == digest(outputs[role])
            if approval['canonical_baselines'][name] is not None:
                backup = contained(HERE, record['backup'])
                assert backup.is_relative_to((HERE / 'baselines').resolve())
                assert file_hash(backup) == approval['canonical_baselines'][name]
            assert file_hash(DATA / name) in {approval['canonical_baselines'][name], digest(outputs[role])}, ('partial transaction drift', name)
        return journal
    assert all(file_hash(DATA / name) == approval['canonical_baselines'][name] for name in TARGETS.values()), 'canonical baseline drift'
    return None


def main(install=False, rollback=False, check_draft=False):
    assert not check_draft or not install and not rollback, 'Draft validation cannot mutate'
    receipt_path = HERE / 'installation-receipt.json'
    # A completed source installation must remain reusable after later builds
    # allocate other durable IDs. Initial/recovery/rollback transactions still
    # require the exact pre-install identity ledger and never modify it.
    approval = validate_inputs(check_draft, check_identity=not (receipt_path.exists() and not rollback))
    outputs = plan(approval)
    assert not (HERE / 'rollback-receipt.json').exists(), 'Rollback requires new root authorization'
    expected = {name: digest(outputs[role]) for role, name in TARGETS.items()}
    if receipt_path.exists() and not rollback:
        receipt = json.loads(receipt_path.read_text())
        assert receipt['approval_sha256'] == file_hash(APPROVAL) and receipt['installed_sha256'] == expected
        for role in ('csv', 'yaml', 'audit', 'profile'):
            assert file_hash(DATA / TARGETS[role]) == expected[TARGETS[role]]
        shared_membership()
        result = {'status': 'no-op; own bytes and current shared membership verified', 'canonical_mutations': False}
        print(json.dumps(result)); return result
    journal = journal_state(approval, outputs)
    if rollback:
        assert journal is not None, 'No installed transaction to roll back'
        if install:
            for role, name in TARGETS.items():
                journal_state(approval, outputs)
                path = DATA / name
                baseline = approval['canonical_baselines'][name]
                if file_hash(path) == baseline:
                    continue
                if baseline is None:
                    path.unlink()
                else:
                    content = contained(HERE, journal['targets'][name]['backup']).read_bytes()
                    temporary = path.with_name(path.name + '.kolami-basim-rollback-tmp')
                    temporary.write_bytes(content); temporary.replace(path)
            assert all(file_hash(DATA / name) == approval['canonical_baselines'][name] for name in TARGETS.values())
            (HERE / 'rollback-receipt.json').write_text(json.dumps({'approval_sha256': file_hash(APPROVAL), 'status': 'exact baselines restored'}) + '\n')
        result = {'status': 'rollback completed' if install else 'rollback dry run validated', 'targets': 6}
        print(json.dumps(result)); return result
    if install:
        if journal is None:
            records = {}
            for role, name in TARGETS.items():
                path = DATA / name
                assert file_hash(path) == approval['canonical_baselines'][name]
                backup_name = None
                if path.exists():
                    backup = HERE / 'baselines' / (role + '.asset')
                    backup.parent.mkdir(exist_ok=True)
                    assert not backup.exists() or file_hash(backup) == file_hash(path)
                    backup.write_bytes(path.read_bytes())
                    backup_name = str(backup.relative_to(HERE))
                records[name] = {'output_sha256': digest(outputs[role]), 'backup': backup_name}
            # Recheck all current baselines before committing recovery authority.
            assert all(file_hash(DATA / name) == approval['canonical_baselines'][name] for name in TARGETS.values())
            journal = {'approval_sha256': file_hash(APPROVAL), 'targets': records}
            temp = HERE / 'installation-transaction.tmp'
            temp.write_text(json.dumps(journal, indent=2) + '\n'); temp.replace(HERE / 'installation-transaction.json')
        for role, name in TARGETS.items():
            journal_state(approval, outputs)
            path, content = DATA / name, outputs[role]
            if file_hash(path) == digest(content):
                continue
            assert file_hash(path) == approval['canonical_baselines'][name]
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + '.kolami-basim-install-tmp')
            assert not temporary.exists() or file_hash(temporary) == digest(content)
            temporary.write_bytes(content); temporary.replace(path)
        assert all(file_hash(DATA / name) == expected[name] for name in TARGETS.values()), 'post-write drift'
        assert file_hash(DATA / 'data/form-identities.csv') == approval['durable_identity_sha256']
        shared_membership()
        result = {'status': 'source-stage installed; compiled/browser gates deferred', 'installed_sha256': expected,
            'approval_sha256': file_hash(APPROVAL), 'rows': 827, 'audit_units': 1570, 'canonical_mutations': True}
        temporary = HERE / 'installation-receipt.tmp'
        temporary.write_text(json.dumps(result, indent=2) + '\n'); temporary.replace(receipt_path)
    else:
        result = {'status': 'disabled draft checked' if check_draft else 'approved dry-run checked',
            'rows': 827, 'audit_units': 1570, 'canonical_mutations': False, 'output_sha256': expected}
    print(json.dumps(result)); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--rollback', action='store_true')
    parser.add_argument('--check-draft', action='store_true')
    args = parser.parse_args()
    main(args.install, args.rollback, args.check_draft)
