"""Extend pending research only; never write accepted sidecars."""
import csv, gzip, json, sys
from pathlib import Path
root = Path.cwd()
p = root / 'data/curation/etymology-lab/pattapu-20260922'
sys.path.insert(0, str(root / 'data'))
from etymology_assignments import read_assignments
z = json.load(open(p / 'proposed-decisions.json'))
seen = {a['record']['ID'] for a in z['accepted'] + z['held']} | {a['Form_ID'] for a in read_assignments()}
rules = {
 ('mala', 'rain (precipitation)'): ('d4753', 'Tamil/Malayalam maḻai/maḻa and Kodagu/Tulu maḷe denote rain. Pattapu mala preserves the two-syllable lateral stem; its plain l is retained rather than silently normalized to a retroflex segment.'),
 ('kalu', 'stone or rock'): ('d1298', 'Kannada kalu/kallu, Telugu kallu and Gondi (Ma.) kalu denote stone, alongside Tamil kal. Pattapu kalu has a direct segmental and semantic comparison within this article.'),
 ('yala', 'leaf'): ('d497', 'Kannada ela/ele and Malayalam ila denote leaf, alongside Tamil ilai. Pattapu yala is proposed with its initial glide and a vowel explicitly retained; this comparison does not establish a general sound law from a single form.'),
 ('mola', 'breast'): ('d4985', 'Malayalam mula and Kannada/Kodagu mole denote breast, alongside Tamil mulai. Pattapu mola closely matches this simple stem. The separate ed̪uɹumāɹu breast response is not assigned by this analysis; the article’s explicit Badaga-to-Kota loan statement does not establish a Pattapu donor.'),
 ('maⁱru', 'hair (head)'): ('d4707', 'Tamil/Malayalam mayir denote hair, with Irula moyiru/megaru and Kota/Toda mīr in the same article. Pattapu maⁱru preserves the diphthongal sequence and rhotic with final u; its superscript vowel remains as recorded.'),
 ('pēɹu', 'name'): ('d4410', 'Tamil/Malayalam pēr and Telugu pēru denote name. Pattapu pēɹu matches the long-vowel rhotic stem; the source’s rhotic transcription is preserved.'),
 ('kāya', 'fruit'): ('d1459', 'Kannada kāya/kāyi and Telugu kāya denote unripe or green fruit, alongside Tamil kāy. Pattapu kāya has the same long-vowel stem but its elicited gloss is the broader fruit; that semantic difference remains explicit. Compounds containing kāya are not included.'),
 ('āɹu', 'river'): ('d5159', 'Tamil yāṟu/āṟu and Malayalam āṟu denote river; Tulu ārů occurs in the same article. Pattapu āɹu matches the river stem. The homographic numeral six is a distinct record assigned separately to DEDR2485.'),
}
for r in json.load(open(p / 'inventory.json')):
    if r['ID'] in seen or (r['Form'], r['Gloss']) not in rules:
        continue
    parent, evidence = rules[r['Form'], r['Gloss']]
    z['accepted'].append(dict(record=r, parent=parent, kind='reflex', citation=f'dedr[entry {parent[1:]}];' + r['Source'], evidence=f'DEDR{parent[1:]}: ' + evidence + ' Whole-article family membership is proposed; inheritance versus local transmission remains subject to review.', status='pending-review'))
(p / 'proposed-decisions.json').write_text(json.dumps(z, ensure_ascii=False, indent=2) + '\n')
used = {a['parent'] for a in z['accepted']}
articles = {r['Entry_ID']: r for r in csv.DictReader(gzip.open(root / 'data/data/dedr/entry-texts-audit.csv.gz', 'rt')) if r['Entry_ID'] in used and r['Status'] != 'duplicate-excluded'}
(p / 'primary-evidence.json').write_text(json.dumps(articles, ensure_ascii=False, indent=2) + '\n')
