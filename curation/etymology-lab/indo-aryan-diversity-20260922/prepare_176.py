"""Audit distinct Varli 'dumb' responses against two Turner heads."""
import csv
import json
import pickle
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_ysculjvll6ag4', '10231-6', 'Davar Varli muŋgɔ “dumb” matches Turner mū́ka CDIAL 10231 §6 *muṅga, with Gujarati mũgũ “dumb”; the nasal velar distinguishes this from the plain muka response.'),
    ('f_5jbcfeh7cbbcs', '4171', 'Dungar Varli guŋga “dumb” matches Turner *guṅga CDIAL 4171, with Gujarati gũgɔ “dumb”; Turner compares Persian gung without deciding the immediate source.'),
    ('f_us7z5pome7l6c', '10231-3', 'A second Dungar Varli response muka “dumb” matches Turner mū́ka CDIAL 10231 §3 *mukka, including Marathi mukā “dumb”.'),
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
wanted = {'4171', '10231'}
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
assert 'G. gũgɔ' in primary['4171']['text']
assert 'G. mũgũ' in primary['10231']['text'] and "M. mukā 'dumb'" in primary['10231']['text']
(base / 'varli-primary015.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')
(base / 'pass176-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
(base / 'pass176-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': []}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass176-review.md').write_text('\n'.join([
    '# Davar and Dungar Varli: distinct dumb words',
    '',
    'Three source-separated responses for “dumb” resolve to two Turner entries and distinct subsections. Existing ingestion audit objects record manual page-image review and no reading uncertainty; they are copied to `pass176-source-audit.json`. Complete Turner articles are in `varli-primary015.json`. The possible older Persian connection mentioned by Turner for *guṅga* is not treated as an immediate borrowing claim.',
    '',
    '| # | Varli survey | Proposed etymology | Evidence |',
    '|---|---|---|---|',
    '| 1 | **muŋgɔ** (Davar) | **mū́ka** §6, CDIAL **10231-6** | Gujarati *mũgũ* “dumb” matches the nasal stem. |',
    '| 2 | **guŋga** (Dungar) | **\\*guṅga**, CDIAL **4171** | Gujarati *gũgɔ* “dumb” matches the initial g- stem. |',
    '| 3 | **muka** (Dungar) | **mū́ka** §3, CDIAL **10231-3** | Marathi *mukā* “dumb” matches the unnasalized response. |',
    '',
    'Three `reflex` rows affect three records. No source ingestion or rebuild.',
    '',
    'Validation pending.',
]) + '\n')
print('accepted', len(accepted))
