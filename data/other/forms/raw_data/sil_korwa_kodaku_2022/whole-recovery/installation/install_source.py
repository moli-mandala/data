"""Draft Behera whole-report installer: read-only unless explicitly --install.

Reuses the tested Aranadan journal/atomic-replacement engine. Root must supply
an exact-hash acceptance manifest; this script never creates approval. No builds.
"""
import argparse
import csv
import importlib.util
import io
import json
from pathlib import Path
import hashlib
import yaml
from pybtex.database import parse_string

HERE = Path(__file__).resolve().parents[1]  # whole-recovery
DATA = next(p for p in HERE.parents if (p/'tags.py').exists())
APPROVAL = HERE/'installation/root-acceptance.json'
ENGINE = DATA/'data/other/forms/raw_data/nair_aranadan_2013/import_source.py'
OLD = '20260828-sil-korwa-kodaku'
NEW = '20260927-behera-whole-additions'
SOURCE = 'behera2022korwakodaku'
TARGETS = {
    'legacy_csv': f'data/other/forms/{OLD}.csv',
    'addition_csv': f'data/other/forms/{NEW}.csv',
    'legacy_yaml': f'data/other/forms/{OLD}.yaml',
    'addition_yaml': f'data/other/forms/{NEW}.yaml',
    'languages': 'cldf/languages.csv',
    'dialects': 'cldf/dialects.csv',
    'bibliography': 'cldf/sources.bib',
    'audit': 'data/other/forms/raw_data/sil_korwa_kodaku_2022/whole-source-audit.jsonl',
}
IDENTITY = 'data/form-identities.csv'

def digest(content): return hashlib.sha256(content).hexdigest()
def file_hash(path): return digest(path.read_bytes()) if path.exists() else None
def contained(root, name):
    path=(root/name).resolve()
    assert path.is_relative_to(root.resolve()), ('Escaping path', name)
    return path

def approval():
    a=json.loads(APPROVAL.read_text())
    assert a['installation_approved'] is True and a['result']=='pass'
    assert a['review_scope']=='whole-report'
    assert a['engine_sha256']==file_hash(ENGINE), 'Changed transaction engine'
    assert set(a['assets'])==set(TARGETS)
    assert set(a['canonical_baselines'])==set(TARGETS.values())
    for name in TARGETS.values():contained(DATA,name)
    assert 'installation/install_source.py' in a['files']
    for name, expected in a['files'].items():
        assert file_hash(contained(HERE,name))==expected, ('Changed approved input',name)
    for asset in a['assets'].values():
        assert asset['input'] in a['files']
        assert asset['output_sha256']==a['files'][asset['input']]
    # Optional original inputs outside the package are read-only and still
    # restricted to this Jambu workspace. They are never copied or deleted.
    for name, expected in a.get('original_inputs',{}).items():
        assert file_hash(contained(DATA.parent,name))==expected, ('Changed original',name)
    assert file_hash(DATA/IDENTITY)==a['durable_identity_sha256']
    return a

def validate_transaction_state(a):
    expected={TARGETS[role]:asset['output_sha256'] for role,asset in a['assets'].items()}
    journal=HERE/'installation-transaction.json'
    if journal.exists():
        j=json.loads(journal.read_text())
        assert j['approval_sha256']==file_hash(APPROVAL)
        assert set(j['targets'])==set(expected)
        for name,entry in j['targets'].items():
            assert entry['output_sha256']==expected[name], ('Unapproved recovery output',name)
            stage=contained(HERE,entry['staged_file'])
            assert stage.is_relative_to((HERE/'installation-stage').resolve())
            assert file_hash(stage)==expected[name]
    receipt=HERE/'installation-receipt.json'
    if receipt.exists():
        r=json.loads(receipt.read_text())
        assert r['approval_sha256']==file_hash(APPROVAL)
        assert r['installed_sha256']==expected, 'Receipt outputs differ from approved assets'

def csv_rows(content): return list(csv.reader(io.StringIO(content.decode())))
def registry(content):
    reader=csv.DictReader(io.StringIO(content.decode())); rows=list(reader)
    assert all(None not in r and r.get('ID') for r in rows)
    out={r['ID']:r for r in rows};assert len(out)==len(rows)
    return reader.fieldnames,out

def validate_assets(a, planned):
    values={role:planned[DATA/name] for role,name in TARGETS.items()}
    legacy=csv_rows(values['legacy_csv']);extra=csv_rows(values['addition_csv'])
    assert len(legacy)==a['legacy_rows'] and len(extra)==a['addition_rows']
    assert len(legacy)+len(extra)==a['rows']
    assert all(len(r)==15 and r[0] and r[2] and r[10] for r in legacy+extra)
    keys={r[10] for r in legacy+extra};assert len(keys)==a['rows']
    before=csv_rows((DATA/TARGETS['legacy_csv']).read_bytes())
    old={r[10]:r for r in before};after={r[10]:r for r in legacy}
    assert len(old)==len(before) and set(old)<=set(after)
    changed={k for k in old if old[k]!=after[k]}
    assert changed==set(a['legacy_changed_keys'])
    assert set(after)-set(old)==set(a['legacy_added_keys'])
    assert all(not r[4] and not r[5] for r in extra), 'New literal route must not invent Native/Phonemic'
    for r in legacy+extra:
        for target in [r[11],r[12],*filter(None,r[13].split('|'))]:
            assert not target or target in keys, ('Missing graph endpoint',r[10],target)
    old_yaml=yaml.safe_load(values['legacy_yaml']);new_yaml=yaml.safe_load(values['addition_yaml'])
    assert old_yaml['file']==OLD+'.csv' and new_yaml['file']==NEW+'.csv'
    assert SOURCE in old_yaml['sources'] and SOURCE not in new_yaml.get('sources',{})
    source=old_yaml['sources'][SOURCE]
    assert not source.get('transcription')
    rule=old_yaml['defaults']['transcription'];rule=rule[0] if isinstance(rule,list) else rule
    assert rule=={'profile':'sil-korwa-kodaku'}, 'Old conversion must remain unchanged'
    assert source['forms']['split_alternates'] is False
    assert source['notes']['audit_only'] is True
    assert source['reference'] == {
        'editor': 'OpenAI Codex', 'ocr': False, 'etymology_provenance': 'source'}
    assert source['identity']['dedupe_by_entry_key'] is True
    command=[str((HERE/'installation/install_source.py').relative_to(DATA)),'--install']
    for settings in (old_yaml,new_yaml):
        assert settings['defaults']['importer']['commands']==[command]
    rule=new_yaml['defaults']['transcription'];rule=rule[0] if isinstance(rule,list) else rule
    assert rule.get('convert') is False and rule.get('preserve_hyphens') is True
    for role in ('languages','dialects'):
        headers,old_reg=registry((DATA/TARGETS[role]).read_bytes())
        new_headers,new_reg=registry(values[role])
        assert headers==new_headers and all(new_reg.get(k)==v for k,v in old_reg.items())
        assert set(new_reg)-set(old_reg)==set(a['registry_added_ids'][role])
    old_bib=parse_string((DATA/TARGETS['bibliography']).read_text(),'bibtex')
    new_bib=parse_string(values['bibliography'].decode(),'bibtex')
    assert set(old_bib.entries)<=set(new_bib.entries)
    assert {k for k,v in old_bib.entries.items() if new_bib.entries[k]!=v}==set(a['bibliography_changed_keys'])
    assert set(a['bibliography_changed_keys'])<={SOURCE}
    assert set(new_bib.entries)-set(old_bib.entries)==set(a['bibliography_added_keys'])
    audit=[json.loads(line) for line in values['audit'].decode().splitlines()]
    schema=a['audit_schema'];assert schema['key_field'] in ('audit_key','source_unit_key')
    assert schema['entry_keys_field'] in ('entry_keys','proposed_entry_keys')
    assert len(audit)==a['audit_units']==len({r[schema['key_field']] for r in audit})
    assert all(r['accepted_for_installation'] is True for r in audit)
    audit_keys={k for r in audit for k in r[schema['entry_keys_field']]}
    assert audit_keys==keys, 'Final audit must cover exactly every installed key'

def prepare():
    a=approval()
    planned={DATA/TARGETS[role]:contained(HERE,asset['input']).read_bytes() for role,asset in a['assets'].items()}
    for path in planned:
        assert file_hash(path)==a['canonical_baselines'][str(path.relative_to(DATA))], ('Canonical drift',str(path))
    validate_assets(a,planned)
    return a,planned,DATA/IDENTITY

def backup_path(name): return HERE/'installation/baselines'/f'{digest(name.encode())}.asset'

def preserve_baselines(a):
    records={}
    for name, expected in a['canonical_baselines'].items():
        path=contained(DATA,name);assert file_hash(path)==expected
        backup=backup_path(name)
        if expected is not None:
            backup.parent.mkdir(parents=True,exist_ok=True)
            assert not backup.exists() or file_hash(backup)==expected
            if not backup.exists(): backup.write_bytes(path.read_bytes())
            assert file_hash(backup)==expected
        records[name]={'sha256':expected,'backup':str(backup.relative_to(HERE)) if expected is not None else None}
    content=json.dumps({'approval_sha256':file_hash(APPROVAL),'targets':records},indent=2)+'\n'
    path=HERE/'installation/baseline-backups.json'
    assert not path.exists() or path.read_text()==content
    if not path.exists():path.write_text(content)

def rollback_plan(a):
    backup=json.loads((HERE/'installation/baseline-backups.json').read_text())
    assert backup['approval_sha256']==file_hash(APPROVAL)
    assert set(backup['targets'])==set(TARGETS.values())
    plan={}
    outputs={TARGETS[role]:asset['output_sha256'] for role,asset in a['assets'].items()}
    for name,record in backup['targets'].items():
        path=contained(DATA,name);expected=a['canonical_baselines'][name]
        assert record['sha256']==expected
        assert file_hash(path) in {expected,outputs[name]}, ('Rollback would overwrite unexpected edit',name)
        if expected is None:plan[path]=None
        else:
            stored=contained(HERE,record['backup']);assert stored==backup_path(name)
            assert file_hash(stored)==expected
            plan[path]=stored.read_bytes()
    return plan

def main(install=False,rollback=False):
    a=approval()
    validate_transaction_state(a)
    if rollback:
        plan=rollback_plan(a)
        if install:
            for path,content in plan.items():
                expected=a['canonical_baselines'][str(path.relative_to(DATA))]
                if file_hash(path)==expected:continue
                # Recheck immediately before each replacement; never erase edits.
                rollback_plan(a)
                if content is None:path.unlink()
                else:
                    temp=path.with_name(path.name+'.behera-rollback-tmp')
                    assert not temp.exists() or file_hash(temp)==digest(content)
                    if not temp.exists():temp.write_bytes(content)
                    temp.replace(path)
            assert all(file_hash(p)==a['canonical_baselines'][str(p.relative_to(DATA))] for p in plan)
            (HERE/'installation/rollback-receipt.json').write_text(json.dumps({'approval_sha256':file_hash(APPROVAL),'status':'rolled back to exact approved baselines'},indent=2)+'\n')
        print(json.dumps({'status':'rollback performed' if install else 'rollback plan validated; no writes','targets':len(plan)}));return
    assert not (HERE/'installation/rollback-receipt.json').exists(), 'Rolled-back transaction requires new root adjudication'
    if install and not (HERE/'installation-transaction.json').exists() and not (HERE/'installation-receipt.json').exists():
        prepare()  # Validate every asset before even staging baseline backups.
        preserve_baselines(a)
    spec=importlib.util.spec_from_file_location('behera_aranadan_engine',ENGINE)
    engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
    engine.HERE=HERE;engine.DATA=DATA;engine.APPROVAL=APPROVAL;engine.prepare=prepare
    engine.main(install=install)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--install',action='store_true')
    p.add_argument('--rollback',action='store_true',help='Validate rollback; add --install to execute it.')
    args=p.parse_args();main(args.install,args.rollback)
