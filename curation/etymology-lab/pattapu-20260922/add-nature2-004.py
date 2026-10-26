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
 'poga': ('smoke (exhaust)', 'd4240', 'Telugu poga denotes smoke, alongside Malayalam puka and Kannada/Kodagu poge. Pattapu poga matches the noun directly; tobacco and verbal derivatives in the article are not substituted for its source meaning.'),
 'θōka': ('tail', 'd3538', 'Malayalam tōka, Kannada tōke and Telugu tō̃ka denote tail, alongside Tamil tōkai. Pattapu θōka retains the long vowel and medial k, with an initial dental fricative in place of the comparanda’s stop; the transcription remains unchanged.'),
 'kāθu': ('wind', 'd1481', 'Tamil/Malayalam kāṟṟu, Kota ka·ṯ and Kodagu ka·tï denote wind. Pattapu kāθu preserves the long-vowel k-initial stem with a dental fricative corresponding to the dental/alveolar stop series. The article’s kāl noun belongs to this same family and does not require changing the displayed head.'),
 'pold̪u': ('sun', 'd4559', 'Tulu poḷtu denotes time, daylight and sun; Kannada portu and Telugu poddu/proddu likewise include sun. Pattapu pold̪u preserves a lateral-plus-dental cluster with a short vowel. Its plain l is not normalized to ḷ and the sun sense is explicit within this whole article.'),
 'θōlu': ('leather or hide', 'd3559', 'Telugu tōlu and Kannada tōl(u) explicitly include leather as well as skin/hide, alongside Tamil tōl. Pattapu θōlu retains the long-vowel lateral stem with final u and an initial dental fricative; this is the attested material sense, not an inferred change from bark.'),
 'muta': ('egg', 'd4939', 'Malayalam muṭṭa/moṭṭa, Tamil muṭṭai and Kodagu muṭṭe denote egg. Pattapu muta matches the rounded-vowel dental-stop stem, but its ungeminated non-retroflex t remains an explicit source difference rather than a corrected spelling.'),
}
for r in json.load(open(p / 'inventory.json')):
    if r['ID'] in seen or r['Form'] not in rules:
        continue
    gloss, parent, evidence = rules[r['Form']]
    assert r['Gloss'] == gloss, r
    z['accepted'].append(dict(record=r, parent=parent, kind='reflex', citation=f'dedr[entry {parent[1:]}];' + r['Source'], evidence=f'DEDR{parent[1:]}: ' + evidence + ' Whole-article family membership is proposed; inheritance versus local transmission remains subject to review.', status='pending-review'))
(p / 'proposed-decisions.json').write_text(json.dumps(z, ensure_ascii=False, indent=2) + '\n')
used = {a['parent'] for a in z['accepted']}
articles = {r['Entry_ID']: r for r in csv.DictReader(gzip.open(root / 'data/data/dedr/entry-texts-audit.csv.gz', 'rt')) if r['Entry_ID'] in used and r['Status'] != 'duplicate-excluded'}
(p / 'primary-evidence.json').write_text(json.dumps(articles, ensure_ascii=False, indent=2) + '\n')
