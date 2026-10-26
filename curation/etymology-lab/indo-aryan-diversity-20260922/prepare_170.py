import csv
import json
import pickle
from pathlib import Path

from bs4 import BeautifulSoup

base = Path(__file__).parent
records = {r['Form']: r for r in json.loads((base / 'andersen-records.json').read_text())}
child = records['pativāsin']
parents = [records['pati'], records['vāsin']]
assert child['ID'] == 'f_q7wohvq62iupa'
assert [r['ID'] for r in parents] == ['f_i3jc6qkafkgfu', 'f_be6moywh2zxci']
assigned = {r['Form_ID'] for r in csv.DictReader(open('data/other/forms/etymologies/20260804-andersen-mre.csv')) if r['Status'] == 'accepted'}
assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank'] == '1')
assert child['ID'] not in assigned
existing = {(r['Child_ID'], r['Parent_ID']) for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank'] == '1'}
assert (parents[0]['ID'], '8540') in existing
assert (parents[1]['ID'], '11605') in existing

wanted = {'8540', '11605'}
primary = {}
for page, html in enumerate(pickle.load(open('data/cdial/cdial.pickle', 'rb')), 1):
    if not any(f'<number>{n}</number>' in html for n in wanted - primary.keys()):
        continue
    for hw in BeautifulSoup(html, 'html.parser').find_all('hw'):
        number = hw.find('number')
        if number and number.get_text(strip=True) in wanted:
            n = number.get_text(strip=True)
            primary[n] = {'number': n, 'page': page, 'url': f'https://dsal.uchicago.edu/cgi-bin/app/soas_query.py?page={page}', 'text': hw.get_text(' ', strip=True), 'html': str(hw)}
    if len(primary) == len(wanted):
        break
assert set(primary) == wanted
(base / 'andersen-primary170.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')

evidence = "Andersen's printed pativāsin- 'neighbor' head compares Sanskrit prativāsin- and Pali pativissaka-. His pati preverb head on the same printed page cross-references pativāsin-, and the vāsin- 'living, staying' head on printed p. 169 also cross-references it. The installed pati and vāsin source heads already have accepted rank-1 Turner 8540 and 11605 ancestry, respectively. Preserve the two explicit lexical components rather than treating the whole compound as a reflex of either bare etymon."
(base / 'pass170-decisions.json').write_text(json.dumps({
    'accepted': [{'record': child, 'parent': parents[0]['ID'], 'components': [r['ID'] for r in parents], 'kind': 'component', 'citation': 'andersen1990[p. 155 (printed p. 157), p. 167 (printed p. 169)];CDIAL[8540];CDIAL[11605]', 'evidence': evidence}],
    'held': [{'record': records['māṇemadesa'], 'reason': "The printed p. 163 Māṇema-region entry cross-references the linked desa 'region' head, but Māṇema is a toponym without a separately installed constituent head. A lone desa component would misrepresent the compound analysis; hold pending a complete source-level representation."}]
}, ensure_ascii=False, indent=2) + '\n')
