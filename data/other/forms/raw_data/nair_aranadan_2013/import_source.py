"""Install approved whole-chapter Aranadan assets locally; never build data.

The transaction engine follows the independently tested Kui installer. Default is
read-only validation; --install requires pinned root approval and canonical baselines.
"""
import argparse,csv,hashlib,json,unicodedata
from pathlib import Path
import yaml
from pybtex.database import parse_file,parse_string
HERE=Path(__file__).resolve().parent
DATA=next(p for p in HERE.parents if (p/'tags.py').exists())
STEM='20260927-nair-aranadan'
APPROVAL=HERE/'root-installation-approval.json'
INPUTS={'csv':'integration/review-candidate.csv','audit':'integration/accepted-audit.jsonl',
        'yaml':'root-review/proposed-settings.yaml','bib':'root-review/proposed-references.bib'}
def digest(content):return hashlib.sha256(content).hexdigest()
def file_hash(path):return digest(path.read_bytes()) if path.exists() else None
def prepare():
    approval=json.loads(APPROVAL.read_text())
    assert approval['installation_approved'] is True and approval['result']=='pass'
    assert set(INPUTS.values())|{'import_source.py'} <= set(approval['files'])
    for name,expected in approval['files'].items():
        path=(HERE/name).resolve()
        assert path.is_relative_to(HERE) and file_hash(path)==expected,name
    rows=list(csv.reader((HERE/INPUTS['csv']).open(newline='')))
    assert len(rows)==approval['rows']==102
    assert all(len(r)==15 and r[2] and r[10] for r in rows)
    keys={r[10] for r in rows}
    assert len(keys)==len(rows)
    assert all(not any(unicodedata.category(c)=='Co' or c=='\ufffd' for c in r[2]) for r in rows)
    assert all(not r[4] and not r[5] for r in rows)
    assert all(not r[11] or r[11] in keys for r in rows)
    audit=[json.loads(line) for line in (HERE/INPUTS['audit']).read_text().splitlines()]
    assert len(audit)==len({r['source_unit_key'] for r in audit})==approval['audit_units']==97
    assert {k for r in audit for k in r['entry_keys']}==keys
    assert all(r['accepted_for_installation'] is True for r in audit)
    settings_bytes=(HERE/INPUTS['yaml']).read_bytes()
    settings=yaml.safe_load(settings_bytes)
    assert settings['file']==STEM+'.csv'
    assert settings['defaults']['transcription']['convert'] is False
    assert settings['defaults']['transcription']['preserve_hyphens'] is True
    assert settings['defaults']['identity']['dedupe_by_entry_key'] is True
    assert settings['defaults']['forms']['split_alternates'] is False
    assert settings['sources']['nair2013aranadan']['reference']['ocr'] is False
    bibliography=DATA/'cldf/sources.bib'
    old=parse_file(str(bibliography))
    added=parse_file(str(HERE/INPUTS['bib']))
    assert set(added.entries)=={'nair2013aranadan','usha1992aranadanmar'}
    assert not set(added.entries)&set(old.entries)
    bib_bytes=bibliography.read_bytes().rstrip()+b'\n\n'+(HERE/INPUTS['bib']).read_bytes()
    merged=parse_string(bib_bytes.decode(),'bibtex')
    assert len(merged.entries)==len(old.entries)+2
    assert all(merged.entries[k]==v for k,v in old.entries.items())
    planned={DATA/f'data/other/forms/{STEM}.csv':(HERE/INPUTS['csv']).read_bytes(),
             DATA/f'data/other/forms/{STEM}.yaml':settings_bytes,
             HERE/'installed-audit.jsonl':(HERE/INPUTS['audit']).read_bytes(),
             bibliography:bib_bytes}
    assert set(approval['canonical_baselines'])=={str(p.relative_to(DATA)) for p in planned}
    for p in planned:
        assert file_hash(p)==approval['canonical_baselines'][str(p.relative_to(DATA))],str(p)
    identity=DATA/'data/form-identities.csv'
    assert file_hash(identity)==approval['durable_identity_sha256']
    return approval,planned,identity

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
            temporary = path.with_name(path.name + '.aranadan-install-tmp')
            assert not temporary.exists() or file_hash(temporary) == expected, temporary
            if not temporary.exists():
                temporary.write_bytes(content)
            temporary.replace(path)
        assert file_hash(identity) == report['durable_identity_sha256']
        # A later target write must not hide intervening edits to an earlier
        # target. Never certify a transaction whose final bytes have drifted.
        for path, content in planned.items():
            assert file_hash(path) == digest(content), ('Post-write target drift', str(path))
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
