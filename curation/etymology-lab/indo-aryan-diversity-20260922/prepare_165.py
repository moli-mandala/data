import csv
import json
import pickle
from pathlib import Path

from bs4 import BeautifulSoup

base = Path(__file__).parent
records = {r['Form']: r for r in json.loads((base / 'andersen-records.json').read_text())}
parent = records['mahapta']
assert parent['ID'] == 'f_utjhvw2doia5y'
targets = [records['mahatva'], records['mahāpta']]
assigned = {r['Form_ID'] for r in csv.DictReader(open('data/other/forms/etymologies/20260804-andersen-mre.csv')) if r['Status'] == 'accepted'}
assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank'] == '1')
assert all(r['ID'] not in assigned for r in targets)
assert any(r['Child_ID'] == parent['ID'] and r['Parent_ID'] == '9942' and r['Rank'] == '1' for r in csv.DictReader(open('cldf/edges.csv')))

primary = None
for page, html in enumerate(pickle.load(open('data/cdial/cdial.pickle', 'rb')), 1):
    if '<number>9942</number>' not in html:
        continue
    for hw in BeautifulSoup(html, 'html.parser').find_all('hw'):
        number = hw.find('number')
        if number and number.get_text(strip=True) == '9942':
            primary = {'number': 9942, 'page': page, 'url': f'https://dsal.uchicago.edu/cgi-bin/app/soas_query.py?page={page}', 'text': hw.get_text(' ', strip=True), 'html': str(hw)}
            break
    if primary:
        break
assert primary and 'mahātman' in primary['text']
(base / 'andersen-primary165.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')

accepted = [
    {'record': records['mahatva'], 'parent': parent['ID'], 'kind': 'variant',
     'citation': 'andersen1990[p. 160 (printed p. 162)];CDIAL[9942]',
     'evidence': "The printed mahatva- entry is an explicit 'see mahapta-' cross-reference. Andersen's mahapta- head on the same page lists mahatva- at Bh, Bi, Ah among inscriptional alternants, with a Sanskrit mahat-tva- comparison. The source parent already has accepted rank-1 Turner 9942 ancestry; retain the attested regional spelling as a variant of that source head."},
    {'record': records['mahāpta'], 'parent': parent['ID'], 'kind': 'variant',
     'citation': 'andersen1990[p. 160 (printed p. 162), p. 161 (printed p. 163)];CDIAL[9942]',
     'evidence': "The printed mahāpta- entry on p. 163 explicitly says 'see mahapta-'; the mahapta- head on p. 162 lists mahāpta- at Br, Sd, Ni as its inscriptional spelling. The source parent already has accepted rank-1 Turner 9942 ancestry. This is an explicit source variant, not a fresh Sanskrit reflex assertion."},
]
held = [
    {'record': records['mahatata'], 'reason': "Andersen compares Skt mahātmatva- and Pali mahantatā- and cross-references maha-/mahapta-. Turner 9933 mahattva 'greatness' is suggestive, but the exact Ashokan mahatata formation and its relation to mahātmatva/mahantatā need separate review; do not collapse it into a direct reflex."},
    {'record': records['mahāpa'], 'reason': "Andersen's p. 163 cross-reference points to mahāpakāraṇa-, whose own compound ancestry remains unresolved. Hold this variant with its parent rather than install a terminal source-to-source link."},
]
(base / 'pass165-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': held}, ensure_ascii=False, indent=2) + '\n')
