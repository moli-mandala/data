"""Select fresh exact-output units from Koya's corrected metadata revision."""
import csv
import hashlib
import json
import random
from pathlib import Path

P = Path(__file__).resolve().parent
R = P/'whole-source-recovery'
F = R/'literal-integration-metadata-r2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest_path = F/'review-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for filename,digest in manifest['outputs_sha256'].items():
        assert sha(F/filename) == digest,filename
    for filename,digest in manifest['scripts_sha256'].items():
        assert sha(R/filename) == digest,filename
    old = json.loads((R/'literal-integration/independent-output-audit-20260927-pass1.json').read_text())
    earlier = json.loads((R/'independent-whole-output-audit-20260927.json').read_text())
    excluded = {unit['source_unit_key'] for unit in old['per_unit']}
    excluded.update(unit['source_unit_key'] for unit in earlier['findings'])
    data = {}
    for kind in ['wordlist','narrative','comparative']:
        with (F/f'proposed-{kind}.csv').open(newline='') as stream:
            for row in csv.reader(stream):
                assert row[10] not in data
                data[row[10]] = (kind,row)
    pools = {kind:[] for kind in ['wordlist','narrative','comparative']}
    units = [json.loads(line) for line in (F/'proposed-whole-audit.jsonl').read_text().splitlines()]
    for unit in units:
        if unit['source_unit_key'] in excluded:
            continue
        keys = unit.get('entry_keys',unit.get('proposed_entry_keys',[]))
        if not keys:
            continue
        actual = [data[key] for key in keys]
        kinds = {kind for kind,_ in actual}
        assert len(kinds) == 1
        kind = next(iter(kinds))
        pools[kind].append(dict(source_unit_key=unit['source_unit_key'],kind=kind,
                               audit_unit=unit,actual_rows=[row for _,row in actual],
                               verdict='pending original-page comparison'))
    rng = random.Random(20260927319)
    selected = []
    for kind,count in [('wordlist',12),('narrative',6),('comparative',2)]:
        selected.extend(rng.sample(pools[kind],count))
    sample = dict(status='fresh selection; no audit verdict',seed=20260927319,
                  manifest_sha256=sha(manifest_path),excluded_prior_source_units=sorted(excluded),selected_units=selected)
    (F/'root-audit-selection-20260927.json').write_text(json.dumps(sample,ensure_ascii=False,indent=2)+'\n')
    for unit in selected:
        print(unit['source_unit_key'],[(row[2],row[3],row[14]) for row in unit['actual_rows']])


if __name__ == '__main__':
    main()
