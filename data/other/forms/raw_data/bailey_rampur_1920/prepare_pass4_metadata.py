"""Reconcile reviewed whole-source metadata from actual legacy/output differences."""
import csv
import json
from pathlib import Path
import yaml
from freeze_full_reread_for_audit import sha

P = Path(__file__).resolve().parent
D = P.parents[4]


def read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.reader(stream))


def main():
    freeze_path = P/'whole-paired-pass4-freeze-20260927.json'
    freeze = json.loads(freeze_path.read_text())
    audit_path = P/'root-independent-whole-output-audit-20260927-pass4.json'
    audit = json.loads(audit_path.read_text())
    assert audit['material_errors'] == 0 and audit['sample_size'] >= 20
    assert audit['freeze_sha256'] == sha(freeze_path)
    reports = {}
    languages_path = D/'cldf/languages.csv'
    with languages_path.open(newline='') as stream:
        languages = {r['ID']: r for r in csv.DictReader(stream)}
    assert {key: languages[key]['Name'] for key in ['ramp', 'ba', 'kc']} == {
        'ramp': 'Rampuri', 'ba': 'Baghi', 'kc': 'Kochi'}
    for name, expected_count in [('rampur',647), ('baghi',698)]:
        package = P.parent/f'bailey_{name}_1920'
        for filename, digest in freeze['packages'][name].items():
            assert sha(package/filename) == digest
        legacy = read_rows(package/'legacy-before-whole-recovery.csv')
        rows = read_rows(package/'pass4-preview.csv')
        by_key = {row[10]: row for row in rows}
        assert len(rows) == len(by_key) == expected_count
        assert all(row[10] in by_key for row in legacy)
        changes = [dict(entry_key=row[10], changed_columns=[i for i,(a,b) in enumerate(zip(row,by_key[row[10]])) if a != b])
                   for row in legacy if row != by_key[row[10]]]
        stem = f'20260925-bailey-{name}'
        canonical_yaml = D/f'data/other/forms/{stem}.yaml'
        meta = yaml.safe_load(canonical_yaml.read_text())
        meta['defaults']['transcription'].update(input='form', preserve_hyphens=True)
        source = f'bailey1920{name}'
        assert meta['sources'][source]['identity']['dedupe_by_entry_key'] is True
        assert meta['sources'][source]['forms']['split_alternates'] is False
        intro = '116' if name == 'rampur' else '116–117'
        scope = (f'Shared introduction pp.113–115; lect-specific introduction pp.{intro}; grammar pp.131–140; '
                 '22 complete translated responses pp.141–142; commentary pp.142–143; glossary pp.144–147.')
        counts = (f'{len(rows)} rows retain all {len(legacy)} legacy keys; {len(changes)} legacy rows have '
                  f'explicit reviewed field corrections, {len(legacy)-len(changes)} remain unchanged; '
                  f'{len(rows)-len(legacy)} additional rows recover the full source scope.')
        attribution = ('Four generic Kochi witnesses are carried once under existing kc with explicit attribution uncertainty.'
                       if name == 'rampur' else 'Generic Kochi witnesses are carried in the companion Rampuri package only.')
        note = f'{scope} {counts} {attribution} Literal notation, source-qualified uncertainties and full expressions are retained.'
        # The final coordinated installer will replace this historical command;
        # leave it explicit in the plan rather than publishing a nonexistent route.
        meta['defaults']['importer']['note'] = note
        proposed_yaml = package/'pass4-proposed.yaml'
        proposed_yaml.write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))
        provenance = '; '.join([
            f'data/other/forms/{stem}.csv', f'data/other/forms/raw_data/bailey_{name}_1920/audit.jsonl',
            'data/other/forms/raw_data/bailey_rampur_1920/whole-paired-pass4-freeze-20260927.json',
            'data/other/forms/raw_data/bailey_rampur_1920/root-independent-whole-output-audit-20260927-pass4.json',
            'data/other/forms/raw_data/bailey_rampur_1920/whole-legacy-full-rereading-20260927.json'])
        reference = dict(citation_id=source, changed_fields=dict(included=note, provenance=provenance),
                         preserve_other_fields=True)
        (package/'pass4-proposed-reference-fields.json').write_text(json.dumps(reference, ensure_ascii=False, indent=2)+'\n')
        reports[name] = dict(rows=len(rows), legacy_rows=len(legacy), changed_legacy_rows=changes,
                             new_rows=len(rows)-len(legacy), current_yaml_sha256=sha(canonical_yaml),
                             proposed_yaml_sha256=sha(proposed_yaml), reference=reference,
                             importer_route_status='Historical glossary route retained in proposal only; replace with verified whole-source route before installation')
    report = dict(status='current reviewed metadata reconciliation; not installed', packages=reports,
                  freeze_sha256=sha(freeze_path), accepted_audit_sha256=sha(audit_path),
                  registry_sha256=sha(languages_path), registry_changes=[], coordinate_changes=[],
                  canonical_changes=False, pending=['Reproducible guarded whole-source importer route',
                      'Coordinated CSV/audit/profile/YAML/bibliography installation and focused checks',
                      'Full build, graph and browser gates deferred under user no-build instruction'])
    (P/'pass4-metadata-reconciliation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    for name, values in reports.items():
        print(name, values['rows'], 'rows;', len(values['changed_legacy_rows']), 'changed legacy rows;', values['new_rows'], 'new rows')


if __name__ == '__main__':
    main()
