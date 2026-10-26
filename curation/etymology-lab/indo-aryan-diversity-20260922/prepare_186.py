"""Audit Varli animal and body-part responses against complete Turner articles."""
import csv
import json
import pickle
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_vmd7to6xvmg2m', '3277', 'Dungar kutra “dog” directly matches Marathi kutrā and Gujarati kutrɔ in Turner *kuttira.'),
    ('f_d3gkgq32l7jv2', '13384', 'Davar sī̃h “lion” matches Punjabi sī̃h in Turner siṁhá.'),
    ('f_fseifbkfctq3w', '13384', 'Dungar sĩhɔ “lion” matches the Turner siṁhá family, with Punjabi sī̃h and regional final vowels.'),
    ('f_le4sbts6owwke', '7627', 'Davar pãkh “wing” matches Gujarati pā̃kh and Marathi pā̃kh in Turner pakṣá; Turner notes possible influence from a distinct puṅkha family on nasalization.'),
    ('f_m5aexvngqtlv4', '7627', 'Dungar pā̃kh “wing” directly matches Gujarati/Marathi pā̃kh in Turner pakṣá; Turner notes possible puṅkha influence on nasalization.'),
    ('f_pbddkfpanzup2', '9982', 'Davar māss “flesh” matches Turner māṁsá, including Gujarati and Marathi mā̃s/mās; the source final doubled s is retained.'),
    ('f_yaolrvsthzvgo', '9982', 'Dungar mas “flesh” matches Turner māṁsá, including Marathi mās and European Romani mas.'),
    ('f_gjoow5wyolimk', '8056', 'Davar paĩ “foot” matches Kumaoni plural paĩ and Hindi pā̃i in Turner pā́da §1.'),
    ('f_3tqvi5tvrudlc', '8056', 'Dungar pā̃i “foot” directly matches Hindi pā̃i in Turner pā́da §1.'),
    ('f_ti7jfiyx4xwps', '14024', 'Davar hath “hand” matches Turner hásta, with Gujarati/Hindi hāth and western hatth reflexes.'),
    ('f_tecup7whfk6ny', '14024', 'Dungar hāth “hand” directly matches Gujarati/Hindi hāth in Turner hásta.'),
    ('f_yxcu4fsye5yja', '8056', 'The independently keyed Davar paĩ “leg” matches Turner pā́da §1; Hindi pā̃i has both foot and leg senses.'),
    ('f_np3iog23bqrmm', '8056', 'The independently keyed Dungar pā̃i “leg” matches Turner pā́da §1; Hindi pā̃i has both foot and leg senses.'),
]
ids = {fid for fid, _, _ in targets}
records = {r['ID']: r for r in csv.DictReader(open('cldf/forms.csv')) if r['ID'] in ids}
assert len(records) == len(ids)
linked = {r['Form_ID'] for r in read_assignments() if r['Status'] == 'accepted'}
linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank'] == '1')
audit_rows = [json.loads(x) for x in open('data/other/forms/raw_data/20260911-dadra-varli-audit.jsonl')]
accepted, audit = [], []
for fid, parent, evidence in targets:
    r = records[fid]
    assert fid not in linked and r['Language_ID'] in {'Bhili', 'Varli'} and r['Source'].startswith('pattanaik-koul2003varli[')
    lect = 'Davar' if r['Language_ID'] == 'Bhili' else 'Dungar'
    item = int(r['Source'].split('item ')[1].split(',')[0])
    page = int(r['Source'].split('[p. ')[1].split(',')[0])
    matches = [x for x in audit_rows if x['source'] == 'pattanaik-koul2003varli' and x['printed_page'] == page and x['item'] == item and x['lect'] == lect and x['gloss'] == r['Gloss'] and x['status'] == 'ingested']
    assert len(matches) == 1, (fid, matches)
    source_row = matches[0]
    assert source_row['review'] == 'manual-page-image' and all(not x['uncertainty'] for x in source_row['forms'])
    assert r['Original'] in [x['form'] for x in source_row['forms']]
    accepted.append({'record': {k: r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')}, 'parent': parent, 'kind': 'reflex', 'citation': f"{r['Source']};CDIAL[{parent}]", 'evidence': evidence + ' Existing manual page-image audit has empty reading uncertainty; exact response and compiled identity checked. No import or transcription change.'})
    audit.append({'id': fid, 'manual_page_audit': source_row, 'compiled_original': r['Original'], 'compiled_form': r['Form'], 'compiled_gloss': r['Gloss'], 'compiled_tags': r['Tags']})
wanted = {'3277', '13384', '7627', '9982', '8056', '14024'}
primary = {}
for page_num, html in enumerate(pickle.load(open('data/cdial/cdial.pickle', 'rb')), 1):
    if not any(f'<number>{n}</number>' in html for n in wanted - primary.keys()):
        continue
    for p in BeautifulSoup(html, 'html.parser').select('p'):
        number = p.find('number')
        if number and number.get_text(strip=True) in wanted:
            n = number.get_text(strip=True)
            primary[n] = {'page': page_num, 'url': f'https://dsal.uchicago.edu/cgi-bin/app/soas_query.py?page={page_num}', 'html': str(p), 'text': p.get_text(' ', strip=True)}
    if len(primary) == len(wanted):
        break
assert len(primary) == len(wanted)
assert "M. kutrā m." in primary['3277']['text']
assert "P. sī˜h" in primary['13384']['text']
assert 'G. pā̃kh' in primary['7627']['text'] and 'puṅkha' in primary['7627']['text']
assert 'G. mā̃s' in primary['9982']['text']
assert 'H. pā̃ , pā̃i' in primary['8056']['text']
assert 'H. Marw. G. hāth' in primary['14024']['text']
(base / 'varli-primary021.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')
(base / 'pass186-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
(base / 'pass186-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': [
    {'form': 'suno', 'gloss': 'dog', 'reason': 'Distinct Davar dog term, not bundled with Dungar kutra.'},
    {'form': 'rɔgɔt', 'gloss': 'blood', 'reason': 'Likely related to rakta, but the source r-g-t form needs a precise regional comparison.'},
]}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass186-review.md').write_text('\n'.join([
    '# Davar and Dungar Varli: dog, lion and body terms',
    '',
    'Thirteen manually image-audited survey records have direct regional comparanda in six complete Turner articles. Each lect and each separately elicited foot/leg key is preserved. The source audits are copied to `pass186-source-audit.json`, and the complete articles are in `varli-primary021.json`.',
    '',
    '| Survey responses | Turner parent | Evidence |',
    '|---|---|---|',
    '| Dungar **kutra** ‘dog’ | **\\*kuttira** 3277 | Marathi *kutrā*, Gujarati *kutrɔ*. |',
    '| Davar **sī̃h**, Dungar **sĩhɔ** ‘lion’ | **siṁhá** 13384 | Punjabi *sī̃h* and regional vocalic endings. |',
    '| Davar **pãkh**, Dungar **pā̃kh** ‘wing’ | **pakṣá** 7627 | Gujarati/Marathi *pā̃kh*; Turner’s nasalization caveat retained. |',
    '| Davar **māss**, Dungar **mas** ‘flesh’ | **māṁsá** 9982 | Gujarati/Marathi *mā̃s/mās*. |',
    '| Davar **paĩ**, Dungar **pā̃i** ‘foot’ and separately ‘leg’ | **pā́da** §1, 8056 | Hindi *pā̃i* has both senses. |',
    '| Davar **hath**, Dungar **hāth** ‘hand’ | **hásta** 14024 | Gujarati/Hindi *hāth*. |',
    '',
    'Thirteen `reflex` rows affect thirteen records. Davar **suno** dog and both **rɔgɔt** blood records remain held for distinct analysis. No source ingestion or rebuild.',
    '',
    'Validation pending.',
]) + '\n')
print('accepted', len(accepted))
