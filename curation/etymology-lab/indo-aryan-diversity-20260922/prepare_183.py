import csv
import json
import pickle
from pathlib import Path

from bs4 import BeautifulSoup

base = Path(__file__).parent
records = json.loads((base / 'andersen-records.json').read_text())
by_form = {r['Form']: r for r in records if r['Form'] != 'su'}
hearing_su = next(r for r in records if r['Form'] == 'su' and r['Gloss'] == 'to hear')
prefix_su = next(r for r in records if r['Form'] == 'su' and r['Gloss'] == 'well, very')
targets = [by_form['i'], by_form['vas']]
assigned = {r['Form_ID'] for r in csv.DictReader(open('data/other/forms/etymologies/20260804-andersen-mre.csv')) if r['Status'] == 'accepted'}
assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank'] == '1')
assert all(r['ID'] not in assigned for r in targets)
assert prefix_su['ID'] != hearing_su['ID']
assert hearing_su['ID'] in assigned

wanted = {'2515', '11435'}
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
assert '√ i' in primary['2515']['text'] and 'Aś. vasati' in primary['11435']['text']
(base / 'andersen-primary183.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')

accepted = [
    {'record': by_form['i'], 'parent': '2515', 'kind': 'reflex',
     'citation': 'andersen1990[p. 140 (printed p. 142)];CDIAL[2515]',
     'evidence': "Andersen's i- 'to go' head explicitly compares RV eti and Pali eti. Turner 2515 ḗti 'goes' identifies √i, cites Pali ēti and exact Ashokan eti. The source head is the verbal root-stem label for this inherited family."},
    {'record': by_form['vas'], 'parent': '11435', 'kind': 'reflex',
     'citation': 'andersen1990[p. 167 (printed p. 169)];CDIAL[11435]',
     'evidence': "Andersen's vas- 'live, stay' head explicitly compares RV vasati and Pali vasati and cross-references vāsin-/vivas-. Turner 11435 vásati 'stays, dwells' identifies √vas, cites Pali vasati and exact Ashokan vasati. This is the same inherited verbal family."},
]
held = [{'record': prefix_su, 'reason': "Andersen's adjacent su- head is the unrelated prefix 'well, very' (RV/Pali su), not the hearing verb. Turner 12598 is excluded for this homograph; no exact standalone prefix parent is selected in this pass."}]
(base / 'pass183-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': held}, ensure_ascii=False, indent=2) + '\n')
