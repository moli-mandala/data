"""Six-target source-stage coordinator patterned on Kolami's journal transaction.

Disabled until root approves exact final assets and contemporary shared baselines.
Per-file replacement is atomic; six-file interruption uses a durable recovery journal.
No database, identity allocation, network operation or publication is performed.
"""
import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
import yaml
from pybtex.database import parse_string

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
DATA = next(p for p in HERE.parents if (p / 'source_meta.py').exists())
sys.path.insert(0, str(DATA))
from source_meta import SourceMeta

STEM = '20260927-grierson-ahirwati'
APPROVAL = HERE / 'root-acceptance.json'
TARGETS = {'csv': f'data/other/forms/{STEM}.csv', 'yaml': f'data/other/forms/{STEM}.yaml',
           'audit': 'data/other/forms/raw_data/grierson_ahirwati_1908/installed-audit.jsonl',
           'profile': 'conversion/grierson-ahirwati-1908.txt',
           'bib': 'cldf/sources.bib', 'dialects': 'cldf/dialects.csv'}

def file_hash(path):
    if not path.exists(): return None
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def digest(data): return hashlib.sha256(data).hexdigest()

def contained(root, name):
    result = (root / name).resolve()
    assert result.is_relative_to(root.resolve()), ('escaping path', name)
    return result

def registry(content):
    reader = csv.DictReader(io.StringIO(content.decode()))
    records = list(reader)
    assert all(None not in r and r.get('ID') for r in records)
    assert len({r['ID'] for r in records}) == len(records)
    return reader.fieldnames, {r['ID']: r for r in records}

def validate_inputs(check_identity=True):
    a = json.loads(APPROVAL.read_text())
    assert a['installation_approved'] is True and a['result'] == 'pass', 'root acceptance required'
    assert a['shared_baseline_captured'] is True and a['audit_acceptance_ready'] is True
    assert set(a['assets']) == set(TARGETS)
    assert set(a['canonical_baselines']) == set(TARGETS.values())
    assert 'installation/install_source.py' in a['files']
    for name, pin in a['files'].items():
        assert file_hash(contained(PACKAGE, name)) == pin, ('approved input drift', name)
    for role, asset in a['assets'].items():
        assert asset['target'] == TARGETS[role] and asset['input'] in a['files']
        assert asset['output_sha256'] == a['files'][asset['input']]
    for name, pin in a['original_inputs'].items():
        assert file_hash(contained(DATA.parent, name)) == pin, ('original drift', name)
    for role, name in [('bib', 'baseline-bibliography.bib'), ('dialects', 'baseline-dialects.csv')]:
        assert a['files']['installation/' + name] == a['canonical_baselines'][TARGETS[role]]
    assert {'installation/staged-references.bib', 'installation/staged-dialects.csv'} <= set(a['files'])
    if check_identity:
        assert file_hash(DATA / 'data/form-identities.csv') == a['durable_identity_sha256']
    return a

def validate_assets(a, outputs):
    records = list(csv.reader(io.StringIO(outputs['csv'].decode())))
    assert len(records) == a['rows'] and all(len(r) == 15 and r[0] and r[2] and r[10] for r in records)
    keys = {r[10] for r in records}
    assert len(keys) == len(records)
    for r in records:
        assert all(k in keys for k in [r[11], r[12], *r[13].split('|')] if k)
    adjacency = {r[10]: [k for k in [r[11], r[12], *r[13].split('|')] if k] for r in records}
    active, done = set(), set()
    def visit(k):
        assert k not in active, ('graph cycle', k)
        if k in done: return
        active.add(k)
        for parent in adjacency[k]: visit(parent)
        active.remove(k); done.add(k)
    for k in keys: visit(k)
    for field, index in [('variant_edges', 11), ('borrowed_edges', 12), ('component_children', 13)]:
        assert sum(bool(r[index]) for r in records) == a[field]
    audit = [json.loads(line) for line in outputs['audit'].decode().splitlines()]
    assert len(audit) == a['audit_units']
    assert all(x['accepted_for_installation'] is True and 'source_record' in x for x in audit)
    # Acceptance here includes documentary/held audit decisions, not lexical emission.
    assert {k for x in audit for k in x.get('entry_keys', [])} == keys
    settings = yaml.safe_load(outputs['yaml'])
    assert settings['file'] == STEM + '.csv'
    assert settings['defaults']['identity']['dedupe_by_entry_key'] is True
    assert settings['defaults']['importer']['commands'] == [[
        'data/other/forms/raw_data/grierson_ahirwati_1908/installation/install_source.py', '--install']]
    rule = settings['defaults']['transcription']
    assert rule == {'profile': 'grierson-ahirwati-1908', 'input': 'form', 'preserve_hyphens': True}
    for source in settings['sources'].values():
        assert source['forms']['split_alternates'] is False
        assert source['gloss']['grammar_tags'] is False
        reference = source['reference']
        assert reference['editor'] == 'OpenAI Codex' and isinstance(reference['ocr'], bool)
        assert reference['etymology_provenance'] in {'source', 'source-mapped', 'jambu', 'mixed', 'none', ''}
    meta = SourceMeta([contained(PACKAGE, a['assets']['yaml']['input'])])
    assert set(meta.sources) == set(settings['sources'])
    refs = parse_string((HERE / 'staged-references.bib').read_text(), 'bibtex').entries
    actual = parse_string(outputs['bib'].decode(), 'bibtex').entries
    prior = parse_string((HERE / 'baseline-bibliography.bib').read_text(), 'bibtex').entries
    assert set(refs) == set(a['own_reference_keys'])
    assert set(prior) & set(refs) == set(a['approved_reference_replacements'])
    assert set(actual) == set(prior) | set(refs)
    assert all(actual[k] == v for k,v in refs.items())
    assert all(actual[k] == v for k,v in prior.items() if k not in refs), 'unrelated bibliography changed'
    header, prior_dialects = registry((HERE / 'baseline-dialects.csv').read_bytes())
    own_header, own_dialects = registry((HERE / 'staged-dialects.csv').read_bytes())
    actual_header, actual_dialects = registry(outputs['dialects'])
    assert header == own_header == actual_header
    assert set(prior_dialects) & set(own_dialects) == set(a['approved_dialect_replacements'])
    assert set(actual_dialects) == set(prior_dialects) | set(own_dialects)
    assert all(actual_dialects[k] == v for k,v in own_dialects.items())
    assert all(actual_dialects[k] == v for k,v in prior_dialects.items() if k not in own_dialects), 'unrelated dialect changed'

def shared_membership():
    own = parse_string((HERE / 'staged-references.bib').read_text(), 'bibtex').entries
    current = parse_string((DATA / TARGETS['bib']).read_text(), 'bibtex').entries
    assert all(k in current and current[k] == v for k,v in own.items()), 'own bibliography drift'
    _, own_dialects = registry((HERE / 'staged-dialects.csv').read_bytes())
    _, current_dialects = registry((DATA / TARGETS['dialects']).read_bytes())
    assert all(k in current_dialects and current_dialects[k] == v for k,v in own_dialects.items()), 'own dialect drift'

def journal_state(a, outputs):
    path = HERE / 'installation-transaction.json'
    if not path.exists():
        assert all(file_hash(DATA / n) == pin for n,pin in a['canonical_baselines'].items()), 'canonical baseline drift'
        return None
    journal = json.loads(path.read_text())
    assert journal['approval_sha256'] == file_hash(APPROVAL)
    assert set(journal['targets']) == set(TARGETS.values())
    for role, name in TARGETS.items():
        record = journal['targets'][name]
        assert record['output_sha256'] == digest(outputs[role])
        assert file_hash(DATA / name) in {a['canonical_baselines'][name], digest(outputs[role])}, ('partial transaction drift', name)
        if a['canonical_baselines'][name] is not None:
            backup = contained(HERE, record['backup'])
            assert backup.is_relative_to((HERE / 'baselines').resolve())
            assert file_hash(backup) == a['canonical_baselines'][name]
    return journal

def main(install=False, rollback=False):
    receipt_path = HERE / 'installation-receipt.json'
    a = validate_inputs(check_identity=not (receipt_path.exists() and not rollback))
    outputs = {role: contained(PACKAGE, asset['input']).read_bytes() for role,asset in a['assets'].items()}
    validate_assets(a, outputs)
    expected = {TARGETS[role]: digest(content) for role,content in outputs.items()}
    assert all(expected[TARGETS[role]] == a['assets'][role]['output_sha256'] for role in TARGETS)
    assert not (HERE / 'rollback-receipt.json').exists(), 'new root authorization required after rollback'
    if receipt_path.exists() and not rollback:
        receipt = json.loads(receipt_path.read_text())
        assert receipt['approval_sha256'] == file_hash(APPROVAL) and receipt['installed_sha256'] == expected
        assert all(file_hash(DATA / TARGETS[role]) == expected[TARGETS[role]] for role in ['csv','yaml','audit','profile'])
        shared_membership()
        return {'status': 'no-op; own assets and semantic shared memberships unchanged', 'canonical_mutations': False}
    journal = journal_state(a, outputs)
    if rollback:
        assert journal is not None
        if install:
            for role, name in TARGETS.items():
                journal_state(a, outputs)
                path = DATA / name
                if file_hash(path) == a['canonical_baselines'][name]: continue
                if a['canonical_baselines'][name] is None: path.unlink()
                else:
                    content = contained(HERE, journal['targets'][name]['backup']).read_bytes()
                    temp = path.with_name(path.name + '.ahirwati-rollback-tmp')
                    temp.write_bytes(content); temp.replace(path)
            assert all(file_hash(DATA / n) == pin for n,pin in a['canonical_baselines'].items())
            (HERE / 'rollback-receipt.json').write_text(json.dumps({'approval_sha256': file_hash(APPROVAL), 'status': 'baselines restored'}) + '\n')
        return {'status': 'rollback completed' if install else 'rollback dry-run checked', 'canonical_mutations': install}
    if not install: return {'status': 'approved dry-run checked', 'canonical_mutations': False, 'targets': 6}
    if journal is None:
        records = {}
        for role, name in TARGETS.items():
            path = DATA / name
            assert file_hash(path) == a['canonical_baselines'][name]
            backup_name = None
            if path.exists():
                backup = HERE / 'baselines' / (role + '.asset')
                backup.parent.mkdir(exist_ok=True)
                assert not backup.exists() or file_hash(backup) == file_hash(path)
                backup.write_bytes(path.read_bytes()); backup_name = str(backup.relative_to(HERE))
            records[name] = {'output_sha256': digest(outputs[role]), 'backup': backup_name}
        assert all(file_hash(DATA / n) == pin for n,pin in a['canonical_baselines'].items())
        temp = HERE / 'installation-transaction.tmp'
        temp.write_text(json.dumps({'approval_sha256': file_hash(APPROVAL), 'targets': records}, indent=2) + '\n')
        temp.replace(HERE / 'installation-transaction.json')
    for role, name in TARGETS.items():
        journal_state(a, outputs)
        path, content = DATA / name, outputs[role]
        if file_hash(path) == digest(content): continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + '.ahirwati-install-tmp')
        assert not temp.exists() or file_hash(temp) == digest(content)
        temp.write_bytes(content); temp.replace(path)
    assert all(file_hash(DATA / n) == pin for n,pin in expected.items()), 'post-write drift'
    assert file_hash(DATA / 'data/form-identities.csv') == a['durable_identity_sha256']
    shared_membership()
    result = {'status': 'source-stage installed; compiled/browser gates deferred', 'canonical_mutations': True,
              'rows': a['rows'], 'audit_units': a['audit_units'], 'installed_sha256': expected,
              'approval_sha256': file_hash(APPROVAL), 'durable_identity_sha256': a['durable_identity_sha256']}
    temp = HERE / 'installation-receipt.tmp'
    temp.write_text(json.dumps(result, indent=2) + '\n'); temp.replace(receipt_path)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--rollback', action='store_true')
    args = parser.parse_args()
    print(json.dumps(main(args.install, args.rollback)))
