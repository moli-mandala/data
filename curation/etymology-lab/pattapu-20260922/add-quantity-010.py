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
 'alāmu': ('all', 'd844', 'Tamil ellām and Malayalam ellām/ellāvum denote all, alongside Kannada/Telugu ella. Pattapu alāmu preserves the lateral-long-vowel-m formation with an initial a, a single lateral and final u. Those source differences remain explicit rather than normalizing the word to ellām.'),
 'koⁿcev': ('few', 'd2041', 'Telugu koncemu explicitly means a little or a few, and Kannada koñce a little; Tamil/Malayalam koñcam gives the related quantity noun. Pattapu koⁿcev matches the nasal-affricate sequence but has final v rather than the quoted m(u). This is a qualified family proposal requiring review of the final consonant, not an exact-form claim.'),
 'ṣēpu': ('red', 'd1931', 'Tamil cēppu is listed among redness forms and Malayalam cōppu means red/red colour. Pattapu ṣēpu preserves a long-vowel sibilant-p formation with plain single p; its retroflex sibilant remains unchanged. The article explicitly identifies Kodagu cōpï and related forms as Malayalam loans, so those are not treated as independent proof of inheritance or as an established Pattapu donor route.'),
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
