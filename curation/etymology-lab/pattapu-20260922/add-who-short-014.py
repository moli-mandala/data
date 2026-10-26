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
 'd̪āɹu': ('who', 'd5151', 'Irula dāru explicitly means who; Kannada dāru and Kodagu dā·rï occur in the same interrogative article. Pattapu d̪āɹu closely matches the initial dental, long vowel and rhotic formation. This specific d-initial member supports the whole-article link even though Tamil yā/e- forms dominate the head.'),
 'kuɹasa': ('short', 'd1851', 'Telugu kuṟuca explicitly means short, dwarfish, small, and Kannada kuṟucu/kuṟaci denotes smallness or dwarfishness. Pattapu kuɹasa is compared with this fuller sibilant-bearing formation rather than only the bare kuṟu stem. Its medial a and s in place of the Telugu u and c remain explicit sound differences; no unsupported suffix node is introduced.'),
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
