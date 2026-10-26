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
 'siɹiki': ('girl', 'd1594', 'Tamil ciṟukki and Malayalam ceṟukki explicitly denote girl; Pattapu siɹiki preserves the sibilant-rhotic-k formation with raised vowels and an ungeminated k. The full female noun is compared, rather than assigning the record to a bare small adjective or inventing a new derivational suffix node.'),
 'paii': ('boy', 'd3939', 'Tamil paiyaṉ/paiyal denote boy, while Tulu paiyya denotes child and paiyyè child or boy. Pattapu paii is proposed as a contracted member of this family, with its written vowel sequence retained. The absence of an overt glide or final consonant is an explicit limitation; no spelling correction or exact-form claim is made.'),
}
for r in json.load(open(p / 'inventory.json')):
    if r['ID'] in seen or r['Form'] not in rules:
        continue
    gloss, parent, evidence = rules[r['Form']]
    assert r['Gloss'] == gloss, r
    z['accepted'].append(dict(record=r, parent=parent, kind='reflex', citation=f'dedr[entry {parent[1:]}];' + r['Source'], evidence=f'DEDR{parent[1:]}: ' + evidence + ' Whole-article family membership is proposed; inheritance versus local transmission remains subject to review.', status='pending-review'))
(p / 'proposed-decisions.json').write_text(json.dumps(z, ensure_ascii=False, indent=2) + '\n')
used = {d for a in z['accepted'] for d in a.get('components', [a['parent']])}
articles = {r['Entry_ID']: r for r in csv.DictReader(gzip.open(root / 'data/data/dedr/entry-texts-audit.csv.gz', 'rt')) if r['Entry_ID'] in used and r['Status'] != 'duplicate-excluded'}
(p / 'primary-evidence.json').write_text(json.dumps(articles, ensure_ascii=False, indent=2) + '\n')
