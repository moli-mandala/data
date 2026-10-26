"""Prepare disabled installation assets outside the immutable reviewed freeze."""
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

STEM = '20260927-nair-malaveda'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
digest = lambda b: hashlib.sha256(b).hexdigest()


def main():
    assert not (HERE / 'root-acceptance.json').exists(), 'Do not replace activated assets'
    freeze = json.loads((PACKAGE / 'review-freeze.json').read_text())
    assert sha(PACKAGE / 'review-freeze.json') == 'de21cde19b8a61715b1096e454b266e82fedd6e2fdecc2204164e1fd8d27b1a9'
    for name, value in freeze['files_sha256'].items():
        assert sha(PACKAGE / name) == value, name
    review = PACKAGE / 'independent-review/independent-original-output-review-pass2.json'
    assert sha(review) == '946669c47f834a96e95aa88aba62efabbff304b376291f05c62855a906c3210b'
    r = json.loads(review.read_text())
    assert r['material_errors'] == 0 and r['overall_passes'] == 20
    assets = HERE / 'assets'
    assets.mkdir(exist_ok=True)
    settings = yaml.safe_load((PACKAGE / 'metadata/proposed-settings.yaml').read_text())
    settings['defaults']['importer'] = {
        'commands': [['data/other/forms/raw_data/nair_malaveda_2013/installation/install_source.py', '--install']],
        'note': 'Guarded complete chapter installation only; frozen lexical inputs and independent pass2 required. Full compiled/browser gates are separate.'}
    bib = parse_string((PACKAGE / 'metadata/proposed-references.bib').read_text(), 'bibtex').entries
    assert len(bib) == 7
    for key in bib:
        settings['sources'].setdefault(key, {})['reference'] = {
            'editor': 'OpenAI Codex', 'ocr': False,
            'etymology_provenance': 'source' if key == 'nair2013malaveda' else 'none'}
    settings_path = HERE / (STEM + '.yaml')
    settings_path.write_text(yaml.safe_dump(settings, allow_unicode=True, sort_keys=False))
    meta = SourceMeta([settings_path])
    assert set(meta.sources) == set(bib)
    assert meta.file_flag(STEM + '.csv', 'identity', 'dedupe_by_entry_key') is True
    for key in bib:
        assert meta.flag(key, 'reference', 'ocr') is False
    audit = []
    for raw in map(json.loads, (PACKAGE / 'proposal-audit.jsonl').read_text().splitlines()):
        record = dict(source_unit_key=raw['source_unit_key'],
            audit_key=raw.get('entry_key') or raw.get('witness', {}).get('witness_key') or raw['source_unit_key'],
            printed_page=raw['printed_page'], entry_keys=[raw['entry_key']] if raw.get('entry_key') else [],
            disposition='eligible reviewed lexical witness' if raw.get('entry_key') else raw['status'],
            accepted_for_installation=True, acceptance_scope='source extraction and independent review; root activation required before installation',
            acceptance_evidence='independent-review/independent-original-output-review-pass2.json',
            acceptance_evidence_sha256=sha(review), source_record=raw)
        audit.append(record)
    rows = list(csv.reader((PACKAGE / 'proposed.csv').open(newline='')))
    assert len(audit) == len({a['audit_key'] for a in audit}) == 240
    assert {k for a in audit for k in a['entry_keys']} == {r[10] for r in rows}
    assert len(rows) == 223 and sum(bool(a['entry_keys']) for a in audit) == 223
    accepted = HERE / 'accepted-audit-proposal.jsonl'
    accepted.write_text(''.join(json.dumps(a, ensure_ascii=False) + '\n' for a in audit))
    targets = {
        'csv': f'data/other/forms/{STEM}.csv', 'yaml': f'data/other/forms/{STEM}.yaml',
        'audit': 'data/other/forms/raw_data/nair_malaveda_2013/installed-audit.jsonl',
        'languages': 'cldf/languages.csv', 'dialects': 'cldf/dialects.csv', 'bib': 'cldf/sources.bib'}
    planned = {'csv': (PACKAGE / 'proposed.csv').read_bytes(), 'yaml': settings_path.read_bytes(), 'audit': accepted.read_bytes()}
    for role, source in [('languages', 'metadata/proposed-language.csv'), ('dialects', 'metadata/proposed-dialect.csv')]:
        current = (DATA / targets[role]).read_bytes()
        reader = csv.DictReader(io.StringIO(current.decode()))
        existing = list(reader)
        proposal = list(csv.DictReader((PACKAGE / source).open(newline='')))
        assert len(proposal) == 1 and list(proposal[0]) == reader.fieldnames
        added = proposal[0]
        assert all(a['ID'] != added['ID'] for a in existing), (role, 'duplicate ID')
        if role == 'dialects':
            assert all(a['Tag'] != added['Tag'] and a['Source_Language_ID'] != added['Source_Language_ID'] for a in existing)
        assert not added['Glottocode'] and not added['Latitude'] and not added['Longitude'] and not added['Quality']
        stream = io.StringIO(newline='')
        csv.writer(stream).writerow([added[f] for f in reader.fieldnames])
        planned[role] = current + (b'' if current.endswith(b'\n') else b'\n') + stream.getvalue().encode()
    old_bytes = (DATA / targets['bib']).read_bytes()
    old_bib = parse_string(old_bytes.decode(), 'bibtex').entries
    assert not set(bib) & set(old_bib)
    planned['bib'] = old_bytes + (b'\n' if old_bytes.endswith(b'\n') else b'\n\n') + (PACKAGE / 'metadata/proposed-references.bib').read_bytes()
    merged = parse_string(planned['bib'].decode(), 'bibtex').entries
    assert set(merged) - set(old_bib) == set(bib)
    assert all(merged[k] == v for k, v in old_bib.items())
    descriptions = {}
    for role, content in planned.items():
        path = assets / (role + '.asset')
        path.write_bytes(content)
        descriptions[role] = {'target': targets[role], 'input': str(path.relative_to(PACKAGE)), 'output_sha256': digest(content)}
    files = dict(freeze['files_sha256'])
    files.update({name: sha(PACKAGE / name) for name in [
        'review-freeze.json', 'independent-review/independent-original-output-review-pass2.json',
        'independent-review/selection-repaired-20260927.json', 'installation/prepare_assets.py',
        'installation/install_source.py', 'installation/check_fixture.py', 'installation/test_installer.py',
        'installation/20260927-nair-malaveda.yaml', 'installation/accepted-audit-proposal.jsonl']})
    files.update({a['input']: a['output_sha256'] for a in descriptions.values()})
    assert all(files.values())
    draft = dict(installation_approved=False, result='disabled; root activation required',
        rows=223, audit_units=240, variants=12, source_pdf_sha256=freeze['source_pdf_sha256'],
        files=files, assets=descriptions,
        canonical_baselines={name: sha(DATA / name) for name in targets.values()},
        durable_identity_sha256=sha(DATA / 'data/form-identities.csv'),
        required_independent_report='independent-review/independent-original-output-review-pass2.json',
        required_independent_report_sha256=sha(review),
        accepted_audit_semantics='All240 source records retained;223 emitted keys and17 explicit control/documentary records. Historical proposal flags nested in source_record; audit review acceptance is not canonical installation.',
        deferred=['full CLDF/database build', 'full test suite', 'compiled identity/graph and browser QA', 'publication'],
        shared_baseline_notice='Reprepare after the shared write window releases if registry/Bib baselines changed. No canonical mutation is performed by this preparer.')
    (HERE / 'root-acceptance-draft.json').write_text(json.dumps(draft, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'disabled approval draft prepared', 'rows': 223, 'audit': 240, 'files': len(files), 'assets': 6,
        'draft_sha256': sha(HERE / 'root-acceptance-draft.json')}))


if __name__ == '__main__':
    main()
