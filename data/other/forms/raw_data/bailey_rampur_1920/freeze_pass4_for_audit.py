"""Freeze pass-four artifacts and draw an independent sample excluding prior audits."""
import csv
import json
import random
from pathlib import Path
from freeze_full_reread_for_audit import sha

P = Path(__file__).resolve().parent


def main():
    previous_freeze = P/'whole-paired-full-reread-freeze-20260927.json'
    prior = json.loads(previous_freeze.read_text())
    assert sha(Path.cwd()/prior['source_pdf']) == prior['source_pdf_sha256']
    packages, hashes = {}, {}
    for name in ['rampur', 'baghi']:
        folder = P.parent/f'bailey_{name}_1920'
        for filename, digest in prior['packages'][name].items():
            assert sha(folder/filename) == digest
        filenames = ['pass4-preview.csv', 'pass4-preview-audit.jsonl', 'pass4-profile.txt',
                     'pass4-preview-report.json', 'pass4-parser-validation.json']
        hashes[name] = {filename: sha(folder/filename) for filename in filenames}
        with (folder/'pass4-preview.csv').open(newline='') as stream:
            rows = list(csv.reader(stream))
        audit = [json.loads(line) for line in (folder/'pass4-preview-audit.jsonl').read_text().splitlines()]
        packages[name] = rows, audit
    freeze = P/'whole-paired-pass4-freeze-20260927.json'
    manifest = dict(status='frozen for independent review; not installed',
                    source_pdf=prior['source_pdf'], source_pdf_sha256=prior['source_pdf_sha256'],
                    prior_freeze_sha256=sha(previous_freeze), packages=hashes,
                    input_sha256={filename: sha(P/filename) for filename in
                                  ['prepare_pass4_preview.py', 'validate_pass4_preview.py',
                                   'freeze_pass4_for_audit.py', 'root-independent-whole-output-audit-20260927-pass3.json']})
    freeze.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    previous = set()
    for suffix in ['', '-pass2', '-pass3']:
        previous.update(unit['audit_index'] for unit in json.loads(
            (P/f'root-frozen-sample-20260927{suffix}.json').read_text())['selected_units'])
    rng = random.Random('root-independent-paired-output-20260927-v4')
    audit = packages['rampur'][1]
    selected = []
    for kind, count in [('glossary', 4), ('grammar', 8), ('sentence', 3), ('introduction', 3), ('commentary', 2)]:
        pool = [i for i, unit in enumerate(audit) if i not in previous
                and unit.get('source_kind', 'glossary') == kind
                and (unit.get('exported_entry_keys') or unit.get('entry_keys')
                     or packages['baghi'][1][i].get('exported_entry_keys'))]
        for index in rng.sample(pool, count):
            paired = {}
            for name, (rows, units) in packages.items():
                unit = units[index]
                keys = unit.get('exported_entry_keys', unit.get('entry_keys', []))
                actual = [row for row in rows if row[10] in keys]
                assert len(actual) == len(keys)
                paired[name] = dict(audit_unit=unit, actual_csv_rows=actual)
            selected.append(dict(audit_index=index, printed_page=audit[index]['printed_page'],
                                 source_kind=kind, packages=paired, verdict='pending original-image review'))
    output = dict(status='fresh sample selected; no verdict', freeze_sha256=sha(freeze),
                  excluded_prior_indices=sorted(previous), selected_units=selected)
    (P/'root-frozen-sample-20260927-pass4.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    for unit in selected:
        print(unit['audit_index'], unit['printed_page'], unit['source_kind'])


if __name__ == '__main__':
    main()
