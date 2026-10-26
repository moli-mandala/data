"""Append researched proposals without modifying accepted sidecars."""
import csv, gzip, json, sys
from pathlib import Path
root = Path.cwd()
p = root / 'data/curation/etymology-lab/pattapu-20260922'
sys.path.insert(0, str(root / 'data'))
from etymology_assignments import read_assignments
z = json.load(open(p / 'proposed-decisions.json'))
seen = {a['record']['ID'] for a in z['accepted'] + z['held']} | {a['Form_ID'] for a in read_assignments()}
rules = {
 'kolupu': ('fat (organic substance)', 'd2146', 'Tamil koḻuppu denotes fat and grease; Malayalam koḻuppu denotes fatness and solidity of broth or curry. Pattapu kolupu preserves this suffixed noun with plain l and single p in the source; it is compared with the attested noun rather than assigned an invented synchronic derivation.'),
 'guⁿdekāi': ('heart', 'd1693', 'Telugu guṇḍe-kāya, Tulu guṇḍe-kāyi and Kannada guṇḍi-kāyi explicitly denote the physical heart. Pattapu guⁿdekāi matches this full formation. Ordered components are DEDR1693 heart/chest plus DEDR1459 kāy/kāya/kāyi fruit/nut; the latter also includes Kodagu kāy kidneys. Source nasal notation and final diphthong are retained. These are component links, not two alternative whole-word etymologies.'),
}
for r in json.load(open(p / 'inventory.json')):
    if r['ID'] in seen or r['Form'] not in rules:
        continue
    gloss, parent, evidence = rules[r['Form']]
    assert r['Gloss'] == gloss, r
    z['accepted'].append(dict(record=r, parent=parent, kind='reflex', citation=f'dedr[entry {parent[1:]}];' + r['Source'], evidence=f'DEDR{parent[1:]}: ' + evidence + ' Whole-article family membership is proposed; inheritance versus local transmission remains subject to review.', status='pending-review'))
for a in z['accepted']:
    if a['record']['Form'] == 'guⁿdekāi' and a['record']['Gloss'] == 'heart':
        a['kind'] = 'component'
        a['components'] = ['d1693', 'd1459']
        a['citation'] = 'dedr[entry 1693, entry 1459];' + a['record']['Source']
(p / 'proposed-decisions.json').write_text(json.dumps(z, ensure_ascii=False, indent=2) + '\n')
used = {d for a in z['accepted'] for d in a.get('components', [a['parent']])}
articles = {r['Entry_ID']: r for r in csv.DictReader(gzip.open(root / 'data/data/dedr/entry-texts-audit.csv.gz', 'rt')) if r['Entry_ID'] in used and r['Status'] != 'duplicate-excluded'}
(p / 'primary-evidence.json').write_text(json.dumps(articles, ensure_ascii=False, indent=2) + '\n')
