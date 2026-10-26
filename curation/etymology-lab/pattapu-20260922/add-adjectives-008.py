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
 'pucu': ('new', 'd4275', 'Tamil/Malayalam putu and Kota pud denote new; Kannada/Tulu posa show another medial consonant outcome in this family. Pattapu pucu retains the p-u stem but its medial c is not an exact match to the quoted forms. This qualified comparison requires review of the local palatal outcome and does not silently respell pucu as putu.'),
 'nelid̪u': ('good', 'd3610', 'Tamil/Malayalam nallatu means that which is good, Kodagu nallë means good, and Gondi (M.) nelā has an e-vowel. Pattapu nelid̪u is proposed within this nal-/nel- family with a final dental formation comparable to nallatu. Its vowels and single lateral remain explicit differences; no new suffix node or synchronic derivation is asserted.'),
 'nēt̪u': ('yesterday', 'd3758', 'Irula nä·ṯṯu denotes yesterday and Kota nē·r has the oblique stem nē·ṯ-. Pattapu nēt̪u matches the long front vowel and dental-stop formation particularly closely. The homographic-looking long adjective nētu is a different source record and is excluded.'),
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
