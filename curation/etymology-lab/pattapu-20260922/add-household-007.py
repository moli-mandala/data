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
 'volaka': ('pestle', 'd672', 'Kannada olake and Malayalam ulakka denote pestle; Kodagu oḷake denotes a wooden pestle. Pattapu volaka preserves the l-k sequence with initial v and final a. Initial v is an explicit comparative difference, as in its pending numeral one proposal, not silently removed. The article’s questioned Sanskrit comparison does not establish a Sanskrit donor.'),
 'voli': ('path or road', 'd5297', 'Tamil/Malayalam vaḻi and Kannada baḻi denote way, path or road. Pattapu voli is proposed with a rounded first vowel and plain lateral retained. These differences require review; superficially similar Tamil oli entries denote sound, luxuriance or washing and do not support the path sense.'),
 'kaʷɹu': ('rope', 'd1254', 'Tamil kayiṟu and Malayalam kayaṟu denote rope; Kodagu kē·rï shows a shorter vocalic formation. Pattapu kaʷɹu retains initial k and the rhotic with final u, but its superscript rounded glide differs from the Tamil/Malayalam medial sequence. This is a qualified family comparison, not a claimed exact sound match or transcription correction.'),
 'çedi': ('tree', 'd1941', 'Malayalam ceṭi denotes shrub or small tree, Kannada ceṭṭu includes tree in general, and Telugu ceṭṭu denotes tree, plant, bush or creeper. Pattapu çedi fits this plant family with its palatal fricative and voiced non-retroflex stop retained. The source’s broad tree gloss is supported by members of the whole article even though Tamil ceṭi is glossed shrub/bush.'),
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
