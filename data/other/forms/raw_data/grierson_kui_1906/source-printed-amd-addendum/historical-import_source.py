"""Install independently approved source-stage assets; never build the database.

Default checks the complete proposed transaction without modifying canonical files.
Approval pins reviewed inputs and canonical baselines. Unexpected edits cause a stop.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import yaml
from pybtex.database import parse_file, parse_string

HERE = Path(__file__).resolve().parent
DATA = next(p for p in HERE.parents if (p / 'tags.py').exists())
STEM = '20260927-grierson-kui'
PROFILE = 'grierson-kui-1906'
APPROVAL = HERE / 'root-installation-approval.json'
INPUTS = {
    'csv': 'integration/combined-candidate.csv',
    'audit': 'integration/combined-audit.jsonl',
    'yaml': 'integration/install-settings-proposed.yaml',
    'dialects': 'integration/proposed-dialects.csv',
    'profile': 'integration/literal-preservation.txt',
    'bib': 'proposed-references.bib',
}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def file_hash(path):
    return digest(path.read_bytes()) if path.exists() else None


def prepare():
    approval = json.loads(APPROVAL.read_text())
    assert approval['installation_approved'] is True
    assert approval['result'] == 'pass'
    assert set(INPUTS.values()) <= set(approval['files'])
    assert 'import_source.py' in approval['files']
    for name, expected in approval['files'].items():
        path = (HERE / name).resolve()
        assert path.is_relative_to(HERE)
        assert file_hash(path) == expected, ('Changed approved input', name)
    rows = list(csv.reader((HERE / INPUTS['csv']).open(newline='')))
    assert len(rows) == approval['rows']
    assert all(len(row) == 15 and row[2] and row[10] for row in rows)
    assert len({row[10] for row in rows}) == len(rows)
    audit = [json.loads(line) for line in (HERE / INPUTS['audit']).read_text().splitlines()]
    assert len(audit) == approval['audit_units']
    assert len({r['source_unit_key'] for r in audit}) == len(audit)
    keys = {row[10] for row in rows}
    for record in audit:
        if record['disposition'] != 'reused-existing-occurrence':
            assert set(record['entry_keys']) <= keys

    settings_bytes = (HERE / INPUTS['yaml']).read_bytes()
    settings = yaml.safe_load(settings_bytes)
    assert settings['file'] == STEM + '.csv'
    assert settings['defaults']['transcription']['profile'] == PROFILE
    assert settings['defaults']['identity']['dedupe_by_entry_key'] is True
    assert settings['defaults']['forms']['split_alternates'] is False
    assert settings['sources']['grierson1906kui']['reference']['etymology_provenance'] == 'source'

    bibliography = DATA / 'cldf/sources.bib'
    old_bib = parse_file(str(bibliography))
    additions = parse_file(str(HERE / INPUTS['bib']))
    assert 'grierson1906kui' in additions.entries
    existing_keys = set(additions.entries) & set(old_bib.entries)
    assert not existing_keys, ('Proposed reference key already exists; reconcile explicitly', sorted(existing_keys))
    bib_bytes = bibliography.read_bytes().rstrip() + b'\n\n' + (HERE / INPUTS['bib']).read_bytes()
    merged = parse_string(bib_bytes.decode(), 'bibtex')
    assert len(merged.entries) == len(old_bib.entries) + len(additions.entries)
    assert all(merged.entries[key] == value for key, value in old_bib.entries.items())

    dialect_path = DATA / 'cldf/dialects.csv'
    with dialect_path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        old_dialects = list(reader)
    proposed = list(csv.DictReader((HERE / INPUTS['dialects']).open(newline='')))
    existing = {r['ID']: r for r in old_dialects}
    assert len(existing) == len(old_dialects)
    assert len({r['ID'] for r in proposed}) == len(proposed)
    for row in proposed:
        assert set(row) == set(fields)
        assert row['ID'] not in existing or existing[row['ID']] == row
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writerows(row for row in proposed if row['ID'] not in existing)
    dialect_bytes = dialect_path.read_bytes()
    if buffer.getvalue():
        if dialect_bytes and not dialect_bytes.endswith(b'\n'):
            dialect_bytes += b'\n'
        dialect_bytes += buffer.getvalue().encode()

    planned = {
        DATA / f'data/other/forms/{STEM}.csv': (HERE / INPUTS['csv']).read_bytes(),
        DATA / f'data/other/forms/{STEM}.yaml': settings_bytes,
        DATA / f'conversion/{PROFILE}.txt': (HERE / INPUTS['profile']).read_bytes(),
        HERE / 'installed-audit.jsonl': (HERE / INPUTS['audit']).read_bytes(),
        bibliography: bib_bytes,
        dialect_path: dialect_bytes,
    }
    assert set(approval['canonical_baselines']) == {str(p.relative_to(DATA)) for p in planned}
    for path in planned:
        assert file_hash(path) == approval['canonical_baselines'][str(path.relative_to(DATA))], ('Canonical input changed', str(path))
    identity = DATA / 'data/form-identities.csv'
    assert file_hash(identity) == approval['durable_identity_sha256']
    return approval, planned, identity


def main(install=False):
    # A repeated invocation succeeds only if the approved inputs and all installed
    # outputs still match the original transaction receipt. It never rewrites them.
    receipt_path = HERE / 'installation-receipt.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert file_hash(APPROVAL) == receipt['approval_sha256']
        approval = json.loads(APPROVAL.read_text())
        assert set(receipt['installed_sha256']) == set(approval['canonical_baselines'])
        for name, expected in approval['files'].items():
            assert file_hash(HERE / name) == expected, name
        for name, expected in receipt['installed_sha256'].items():
            assert file_hash(DATA / name) == expected, name
        assert file_hash(DATA / 'data/form-identities.csv') == receipt['durable_identity_sha256']
        print(json.dumps({'status': 'already installed; all approved inputs and outputs unchanged'}))
        return
    journal_path = HERE / 'installation-transaction.json'
    if journal_path.exists():
        journal = json.loads(journal_path.read_text())
        assert file_hash(APPROVAL) == journal['approval_sha256']
        approval = json.loads(APPROVAL.read_text())
        assert approval['installation_approved'] is True and approval['result'] == 'pass'
        for name, expected in approval['files'].items():
            path = (HERE / name).resolve()
            assert path.is_relative_to(HERE)
            assert file_hash(path) == expected, name
        assert set(journal['targets']) == set(approval['canonical_baselines'])
        planned = {}
        for name, entry in journal['targets'].items():
            path = (DATA / name).resolve()
            stage = (HERE / entry['staged_file']).resolve()
            assert path.is_relative_to(DATA) and stage.is_relative_to(HERE / 'installation-stage')
            assert file_hash(stage) == entry['output_sha256']
            assert file_hash(path) in {approval['canonical_baselines'][name], entry['output_sha256']}, ('Unexpected partial-transaction edit', name)
            planned[path] = stage.read_bytes()
        identity = DATA / 'data/form-identities.csv'
        assert file_hash(identity) == approval['durable_identity_sha256']
    else:
        approval, planned, identity = prepare()
    report = {'status': 'validated source-stage installation plan', 'rows': approval['rows'],
              'audit_units': approval['audit_units'],
              'installed_sha256': {str(p.relative_to(DATA)): digest(b) for p, b in planned.items()},
              'approval_sha256': file_hash(APPROVAL), 'durable_identity_sha256': file_hash(identity),
              'deferred': ['database/full CLDF build', 'full test suite', 'compiled graph and browser QA', 'publication']}
    if install:
        if not journal_path.exists():
            stage_dir = HERE / 'installation-stage'
            stage_dir.mkdir(exist_ok=True)
            targets = {}
            for number, (path, content) in enumerate(planned.items()):
                staged = stage_dir / f'{number:02d}.asset'
                assert not staged.exists() or file_hash(staged) == digest(content), staged
                if not staged.exists():
                    staged.write_bytes(content)
                targets[str(path.relative_to(DATA))] = {'staged_file': str(staged.relative_to(HERE)), 'output_sha256': digest(content)}
            journal = {'approval_sha256': file_hash(APPROVAL), 'targets': targets}
            journal_tmp = journal_path.with_suffix('.tmp')
            assert not journal_tmp.exists() or json.loads(journal_tmp.read_text()) == journal
            journal_tmp.write_text(json.dumps(journal, indent=2) + '\n')
            journal_tmp.replace(journal_path)
        for path, content in planned.items():
            expected = digest(content)
            if file_hash(path) == expected:
                continue
            assert file_hash(path) == approval['canonical_baselines'][str(path.relative_to(DATA))]
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + '.kui-install-tmp')
            assert not temporary.exists() or file_hash(temporary) == expected, temporary
            if not temporary.exists():
                temporary.write_bytes(content)
            temporary.replace(path)
        assert file_hash(identity) == report['durable_identity_sha256']
        report['status'] = 'source-stage installed; compiled gates deferred'
        # The validated journal remains the recovery authority until the complete
        # receipt is atomically installed. A partial receipt must never mask it.
        receipt_tmp = receipt_path.with_suffix('.tmp')
        receipt_tmp.write_text(json.dumps(report, indent=2) + '\n')
        receipt_tmp.replace(receipt_path)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    main(parser.parse_args().install)
