"""Guarded six-target Malaveda installer. Dry run by default; root activates it.

Approved assets are immutable. A journal permits exact interrupted recovery;
rollback restores original assets only while no unexpected edits have occurred.
Completed no-ops verify own bytes and shared contribution membership, allowing
later unrelated source installations. No builds or publication occur here.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import sys
import unicodedata
from pathlib import Path

import yaml
from pybtex.database import parse_string

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
DATA = next(p for p in HERE.parents if (p / 'source_meta.py').exists())
sys.path.insert(0, str(DATA))
from source_meta import SourceMeta

STEM = '20260927-nair-malaveda'
APPROVAL = HERE / 'root-acceptance.json'
TARGETS = {'csv': f'data/other/forms/{STEM}.csv', 'yaml': f'data/other/forms/{STEM}.yaml',
    'audit': 'data/other/forms/raw_data/nair_malaveda_2013/installed-audit.jsonl',
    'languages': 'cldf/languages.csv', 'dialects': 'cldf/dialects.csv', 'bib': 'cldf/sources.bib'}
digest = lambda content: hashlib.sha256(content).hexdigest()


def file_hash(path):
    return digest(path.read_bytes()) if path.exists() else None


def contained(root, name):
    path = (root / name).resolve()
    assert path.is_relative_to(root.resolve()), ('escaping path', name)
    return path


def validate_inputs(check_draft=False, check_identity=True):
    path = HERE / 'root-acceptance-draft.json' if check_draft else APPROVAL
    approval = json.loads(path.read_text())
    assert approval['installation_approved'] is (False if check_draft else True)
    if not check_draft:
        assert approval['result'] == 'pass'
    assert set(approval['assets']) == set(TARGETS)
    assert set(approval['canonical_baselines']) == set(TARGETS.values())
    assert 'installation/install_source.py' in approval['files']
    for name, expected in approval['files'].items():
        assert file_hash(contained(PACKAGE, name)) == expected, ('approved input drift', name)
    for role, asset in approval['assets'].items():
        assert asset['target'] == TARGETS[role]
        assert asset['input'] in approval['files']
        assert asset['output_sha256'] == approval['files'][asset['input']]
    assert file_hash(DATA.parent / 'tmp/pdfs/tribal-languages-kerala-nair2013.pdf') == approval['source_pdf_sha256']
    if check_identity:
        assert file_hash(DATA / 'data/form-identities.csv') == approval['durable_identity_sha256']
    return approval


def registry(content):
    reader = csv.DictReader(io.StringIO(content.decode()))
    rows = list(reader)
    assert all(None not in r and r.get('ID') for r in rows)
    assert len({r['ID'] for r in rows}) == len(rows)
    return reader.fieldnames, {r['ID']: r for r in rows}


def validate_assets(approval, planned):
    rows = list(csv.reader(io.StringIO(planned['csv'].decode())))
    assert len(rows) == approval['rows'] == 223
    assert all(len(r) == 15 and r[0] == 'Malaveda' and r[2] and r[10] for r in rows)
    keys = {r[10] for r in rows}
    assert len(keys) == 223
    for row in rows:
        assert row[1] == row[4] == row[5] == row[8] == ''
        assert unicodedata.is_normalized('NFC', row[2])
        assert all(unicodedata.category(c) != 'Co' and c != '\ufffd' for c in row[2])
        assert row[7].startswith('nair2013malaveda[p. ')
        assert not row[12] and not row[13]
        assert not row[11] or row[11] in keys and row[11] != row[10]
    assert sum(bool(r[11]) for r in rows) == approval['variants'] == 12
    audit = list(map(json.loads, planned['audit'].decode().splitlines()))
    assert len(audit) == len({a['audit_key'] for a in audit}) == approval['audit_units'] == 240
    audit_keys = [k for a in audit for k in a['entry_keys']]
    assert len(audit_keys) == len(set(audit_keys)) == 223 and set(audit_keys) == keys
    assert all(a['accepted_for_installation'] is True and 'source_record' in a for a in audit)
    assert sum(not a['entry_keys'] for a in audit) == 17
    settings = yaml.safe_load(planned['yaml'])
    assert settings['file'] == STEM + '.csv'
    assert settings['defaults']['transcription'] == {'convert': False, 'preserve_hyphens': True}
    assert settings['defaults']['identity']['dedupe_by_entry_key'] is True
    assert settings['defaults']['importer']['commands'] == [['data/other/forms/raw_data/nair_malaveda_2013/installation/install_source.py', '--install']]
    refs = parse_string((PACKAGE / 'metadata/proposed-references.bib').read_text(), 'bibtex').entries
    assert set(settings['sources']) == set(refs) and len(refs) == 7
    for key, source in settings['sources'].items():
        assert source['reference'] == {'editor': 'OpenAI Codex', 'ocr': False,
            'etymology_provenance': 'source' if key == 'nair2013malaveda' else 'none'}
    meta = SourceMeta([PACKAGE / 'installation/20260927-nair-malaveda.yaml'])
    assert set(meta.sources) == set(refs)
    for role, proposal in [('languages', 'metadata/proposed-language.csv'), ('dialects', 'metadata/proposed-dialect.csv')]:
        header, installed = registry(planned[role])
        expected = list(csv.DictReader((PACKAGE / proposal).open(newline='')))
        assert len(expected) == 1 and list(expected[0]) == header
        assert installed[expected[0]['ID']] == expected[0]
    installed_bib = parse_string(planned['bib'].decode(), 'bibtex').entries
    assert all(installed_bib[k] == v for k, v in refs.items())


def shared_membership():
    for role, proposal in [('languages', 'metadata/proposed-language.csv'), ('dialects', 'metadata/proposed-dialect.csv')]:
        expected = list(csv.DictReader((PACKAGE / proposal).open(newline='')))[0]
        _, actual = registry((DATA / TARGETS[role]).read_bytes())
        assert actual.get(expected['ID']) == expected, ('shared contribution drift', role)
        if role == 'dialects':
            assert sum(r['Tag'] == expected['Tag'] for r in actual.values()) == 1
            assert sum(r['Source_Language_ID'] == expected['Source_Language_ID'] for r in actual.values()) == 1
    expected = parse_string((PACKAGE / 'metadata/proposed-references.bib').read_text(), 'bibtex').entries
    actual = parse_string((DATA / TARGETS['bib']).read_text(), 'bibtex').entries
    assert all(k in actual and actual[k] == v for k, v in expected.items()), 'shared bibliography contribution drift'


def plan(approval):
    outputs = {role: contained(PACKAGE, a['input']).read_bytes() for role, a in approval['assets'].items()}
    validate_assets(approval, outputs)
    for role, content in outputs.items():
        assert digest(content) == approval['assets'][role]['output_sha256']
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
        for role in ('csv', 'yaml', 'audit'):
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
                    temporary = path.with_name(path.name + '.malaveda-rollback-tmp')
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
            temporary = path.with_name(path.name + '.malaveda-install-tmp')
            assert not temporary.exists() or file_hash(temporary) == digest(content)
            temporary.write_bytes(content); temporary.replace(path)
        assert all(file_hash(DATA / name) == expected[name] for name in TARGETS.values()), 'post-write drift'
        assert file_hash(DATA / 'data/form-identities.csv') == approval['durable_identity_sha256']
        shared_membership()
        result = {'status': 'source-stage installed; compiled/browser gates deferred', 'installed_sha256': expected,
            'approval_sha256': file_hash(APPROVAL), 'rows': 223, 'audit_units': 240, 'canonical_mutations': True}
        temporary = HERE / 'installation-receipt.tmp'
        temporary.write_text(json.dumps(result, indent=2) + '\n'); temporary.replace(receipt_path)
    else:
        result = {'status': 'disabled draft checked' if check_draft else 'approved dry-run checked',
            'rows': 223, 'audit_units': 240, 'canonical_mutations': False, 'output_sha256': expected}
    print(json.dumps(result)); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--rollback', action='store_true')
    parser.add_argument('--check-draft', action='store_true')
    args = parser.parse_args()
    main(args.install, args.rollback, args.check_draft)
