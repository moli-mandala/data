"""Prepare reviewed Mannan assets and a disabled exact installation draft."""
import argparse
import csv
import hashlib
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

STEM = '20260927-nair-mannan'
digest = lambda b: hashlib.sha256(b).hexdigest()


def sha(path):
    if not path.exists():
        return None
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main(local_only=False):
    assert not (HERE / 'root-acceptance.json').exists(), 'Activated assets are immutable'
    freeze_path = PACKAGE / 'review-freeze.json'
    assert sha(freeze_path) == '8fe6620df601435704e65f2f66b1382f34dbb507392123b3ee48ad30cea4829a'
    freeze = json.loads(freeze_path.read_text())
    for name, value in freeze['files_sha256'].items():
        assert sha(PACKAGE / name) == value, name
    review_path = PACKAGE / 'root-original-review-final-20260927452.json'
    assert sha(review_path) == 'e9c2fee8953367c73c3b8d04a12a6e03d9746c3fdb4741a0e3cb930a4673b482'
    review = json.loads(review_path.read_text())
    assert review['material_errors'] == 0 and review['source_stage_accepted'] is True
    settings = yaml.safe_load((PACKAGE / 'proposed-settings.yaml').read_text())
    settings['defaults']['importer'] = {
        'commands': [['data/other/forms/raw_data/nair_mannan_2013/installation/install_source.py', '--install']],
        'note': 'Guarded complete reviewed chapter only; historical literal generator followed by metadata overlay remains archival. Full compiled/browser gates separate.'}
    settings_path = HERE / (STEM + '.yaml')
    settings_path.write_text(yaml.safe_dump(settings, allow_unicode=True, sort_keys=False))
    assert len(SourceMeta([settings_path]).sources) == 2
    audit = []
    for raw in map(json.loads, (PACKAGE / 'proposal-audit.jsonl').read_text().splitlines()):
        audit.append(dict(source_unit_key=raw['source_unit_key'], audit_key=raw['source_unit_key'],
            printed_pages=raw['printed_pages'], entry_keys=raw['proposed_entry_keys'],
            disposition=raw['disposition'], accepted_for_installation=True,
            acceptance_scope='root-accepted complete source stage; canonical transaction requires separate activation',
            acceptance_evidence=review_path.name, acceptance_evidence_sha256=sha(review_path), source_record=raw))
    assert len(audit) == len({a['audit_key'] for a in audit}) == 140
    rows = list(csv.reader((PACKAGE / 'proposed.csv').open(newline='')))
    keys = [k for a in audit for k in a['entry_keys']]
    assert len(keys) == len(set(keys)) == len(rows) == 154 and set(keys) == {r[10] for r in rows}
    assert sum(len(a['source_record']['witness_dispositions']) for a in audit) == 208
    accepted_path = HERE / 'accepted-audit-proposal.jsonl'
    accepted_path.write_text(''.join(json.dumps(a, ensure_ascii=False) + '\n' for a in audit))
    raw_bib = (PACKAGE / 'proposed-references.bib').read_text()
    old_provenance = 'data/other/forms/raw_data/nair_mannan_2013/physical-units-first-reading.jsonl; data/other/forms/raw_data/nair_mannan_2013/proposal-audit.jsonl'
    assert raw_bib.count(old_provenance) == 1
    new_provenance = 'data/other/forms/20260927-nair-mannan.csv; data/other/forms/raw_data/nair_mannan_2013/installed-audit.jsonl; ' + old_provenance
    staged_bib = raw_bib.replace(old_provenance, new_provenance)
    raw_entries = parse_string(raw_bib, 'bibtex').entries
    staged_entries = parse_string(staged_bib, 'bibtex').entries
    assert set(raw_entries) == set(staged_entries) == {'nair2013mannan', 'joseph1980devikolam'}
    assert raw_entries['joseph1980devikolam'] == staged_entries['joseph1980devikolam']
    assert raw_entries['nair2013mannan'].persons == staged_entries['nair2013mannan'].persons
    assert {k: v for k, v in raw_entries['nair2013mannan'].fields.items() if k != 'provenance'} == {k: v for k, v in staged_entries['nair2013mannan'].fields.items() if k != 'provenance'}
    (HERE / 'staged-references.bib').write_text(staged_bib)
    assets = HERE / 'assets'
    assets.mkdir(exist_ok=True)
    planned = {'csv': (PACKAGE / 'proposed.csv').read_bytes(), 'yaml': settings_path.read_bytes(), 'audit': accepted_path.read_bytes()}
    for role, content in planned.items():
        (assets / (role + '.asset')).write_bytes(content)
    if local_only:
        print(json.dumps({'status': 'three own assets staged; shared baseline and disabled draft await Kakkala', 'rows': 154, 'audit': 140}))
        return
    targets = {'csv': f'data/other/forms/{STEM}.csv', 'yaml': f'data/other/forms/{STEM}.yaml',
        'audit': 'data/other/forms/raw_data/nair_mannan_2013/installed-audit.jsonl', 'bib': 'cldf/sources.bib'}
    old_bytes = (DATA / targets['bib']).read_bytes()
    old = parse_string(old_bytes.decode(), 'bibtex').entries
    extra = (HERE / 'staged-references.bib').read_bytes()
    added = parse_string(extra.decode(), 'bibtex').entries
    assert set(added) == {'nair2013mannan', 'joseph1980devikolam'} and not set(added) & set(old)
    planned['bib'] = old_bytes + (b'\n' if old_bytes.endswith(b'\n') else b'\n\n') + extra
    merged = parse_string(planned['bib'].decode(), 'bibtex').entries
    assert set(merged) - set(old) == set(added)
    assert all(merged[k] == v for k, v in old.items())
    descriptions = {}
    for role, content in planned.items():
        path = assets / (role + '.asset')
        path.write_bytes(content)
        descriptions[role] = dict(target=targets[role], input=str(path.relative_to(PACKAGE)), output_sha256=digest(content))
    files = dict(freeze['files_sha256'])
    for name in ['review-freeze.json', review_path.name, 'installation/prepare_installer.py', 'installation/install_source.py',
                 'installation/prepare_assets.py', 'installation/prepare_review_checks.py', 'installation/check_fixture.py', 'installation/test_installer.py',
                 'installation/' + settings_path.name, 'installation/staged-references.bib', 'installation/accepted-audit-proposal.jsonl']:
        files[name] = sha(PACKAGE / name)
    if (HERE / 'actual-parser-check.json').exists():
        files['installation/actual-parser-check.json'] = sha(HERE / 'actual-parser-check.json')
    files.update({a['input']: a['output_sha256'] for a in descriptions.values()})
    assert all(files.values())
    draft = dict(installation_approved=False, result='disabled; root activation required', rows=154, audit_units=140,
        witness_records=208, target_rows=123, comparison_rows=31, audit_only_witness_records=54, variants=0,
        files=files, assets=descriptions, source_pdf_sha256=review['source_pdf_sha256'],
        canonical_baselines={name: sha(DATA / name) for name in targets.values()},
        durable_identity_sha256=sha(DATA / 'data/form-identities.csv'),
        required_independent_report=review_path.name, required_independent_report_sha256=sha(review_path),
        existing_registry='Mannan/mann1246 and Malayalam already registered; no registry or dialect edits',
        deferred=['full CLDF/database build', 'full suite', 'compiled identity/graph/reference survival', 'browser QA/publication'])
    (HERE / 'root-acceptance-draft.json').write_text(json.dumps(draft, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'disabled four-target draft', 'rows': 154, 'audit': 140, 'files': len(files),
        'draft_sha256': sha(HERE / 'root-acceptance-draft.json')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-only', action='store_true')
    main(parser.parse_args().local_only)
