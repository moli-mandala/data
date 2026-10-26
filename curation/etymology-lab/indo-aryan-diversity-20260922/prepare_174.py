"""Audit Varli ashes, cold and blind responses against complete CDIAL heads."""
import csv
import json
import pickle
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_fqnakpmz34e3o', '10552', 'Davar Varli rākh “ashes” matches Gujarati and Marathi rākh under Turner rakṣā², CDIAL 10552.'),
    ('f_7ej3jd3c4s2z4', '10555', 'Dungar Varli rakhaṛi “ashes” matches Gujarati rākhɔṛī “layers of ashes, ashes” under Turner *rakṣāpuṭaka, CDIAL 10555.'),
    ('f_yb4batyluepku', '13676', 'Dungar Varli thɔṇḍa “cold” matches Marathi thaṇḍā and Hindi ṭhaṇḍā in Turner stabdha, CDIAL 13676, section 2. Turner notes influence from Dravidian cold vocabulary on the later Indo-Aryan forms; this link does not decide the deeper etymology alone.'),
    ('f_ydf3tt4entd42', '385', 'Davar Varli andhḍo “blind” belongs with Turner andhá, CDIAL 385; the additional source retroflex ending is retained without a stronger morphological claim.'),
    ('f_wzsf6gt7loc6g', '385', 'Dungar Varli andhla “blind” matches the expanded -la- branch of Turner andhá, CDIAL 385, with Punjabi andhlā and Gujarati ā̃dhḷũ.'),
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
wanted = {'385', '10552', '10555', '13676'}
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
assert 'G. M. rākh' in primary['10552']['text']
assert 'rākhɔṛɔ' in primary['10555']['text'] and 'ashes' in primary['10555']['text']
assert 'M. thaṇḍ' in primary['13676']['text'] and 'Drav.' in primary['13676']['text']
assert 'Ext. with - la -' in primary['385']['text'] and 'P. andhlā' in primary['385']['text']
(base / 'varli-primary013.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')
(base / 'pass174-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
(base / 'pass174-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': [
    {'form': 'lākhāṛo / arkhoṛo', 'gloss': 'ashes', 'reason': 'Additional Davar alternatives need distinct derivation and transcription comparison before linking.'},
    {'form': 'sī̃ / hyũ', 'gloss': 'cold', 'reason': 'Other cold responses lack a secure Turner branch.'},
    {'form': 'andher', 'gloss': 'darkness', 'reason': 'Could be recent Hindustani borrowing; transmission not established here.'},
]}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass174-review.md').write_text('\n'.join([
    '# Davar and Dungar Varli: ashes, cold and blind',
    '',
    'Five survey responses have exact western Indo-Aryan comparanda in complete Turner articles. The existing Varli ingestion audit records manual page-image review and no reading uncertainty for each response; selected audit objects are copied to `pass174-source-audit.json`. Complete Turner articles are in `varli-primary013.json`.',
    '',
    '| # | Varli survey | Proposed etymology | Evidence |',
    '|---|---|---|---|',
    '| 1 | **rākh** ‘ashes’ (Davar) | **rakṣā²**, CDIAL **10552** | Gujarati/Marathi *rākh* is explicit. |',
    '| 2 | **rakhaṛi** ‘ashes’ (Dungar) | **\\*rakṣāpuṭaka**, CDIAL **10555** | Gujarati *rākhɔṛī* denotes ash layers and ashes. |',
    '| 3 | **thɔṇḍa** ‘cold’ (Dungar) | **stabdha**, CDIAL **13676** | Turner lists Marathi *thaṇḍā* and notes possible Dravidian influence on the later cold word. |',
    '| 4–5 | **andhḍo** (Davar), **andhla** (Dungar) ‘blind’ | **andhá**, CDIAL **385** | The latter matches Turner’s -la- expansion; the former retains an unanalysed retroflex ending. |',
    '',
    'Five `reflex` rows affect five records. Other ash and cold alternatives and **andher** darkness remain held. No source ingestion or rebuild.',
    '',
    'Validation pending.',
]) + '\n')
print('accepted', len(accepted))
