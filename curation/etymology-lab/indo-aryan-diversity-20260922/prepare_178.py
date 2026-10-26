import csv
import json
import pickle
from pathlib import Path

from bs4 import BeautifulSoup

base = Path(__file__).parent
records = {r['Form']: r for r in json.loads((base / 'andersen-records.json').read_text())}
assigned = {r['Form_ID'] for r in csv.DictReader(open('data/other/forms/etymologies/20260804-andersen-mre.csv')) if r['Status'] == 'accepted'}
assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank'] == '1')
assert all(records[f]['ID'] not in assigned for f in ('as', 'gam'))
assert any(r['Child_ID'] == records['apa']['ID'] and r['Parent_ID'] == '1188' and r['Rank'] == '1' for r in csv.DictReader(open('cldf/edges.csv')))

wanted = {'977', '3955', '1188'}
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
assert '√ as' in primary['977']['text'] and '[√ gam ]' in primary['3955']['text']
(base / 'andersen-primary178.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')

accepted = [
    {'record': records['as'], 'parent': '977', 'kind': 'reflex',
     'citation': 'andersen1990[p. 137 (printed p. 139)];CDIAL[977]',
     'evidence': "Andersen's as- 'to exist, to be' head explicitly compares RV asti and Pali atthi and discusses first-person sumi. Turner 977 ásti 'is' names √as, gives Pali atthi and attested Ashokan asti/athi, and describes first-person asmi. This is the matching inherited verbal family; the lexical source head uses the root stem rather than the finite form."},
    {'record': records['gam'], 'parent': '3955', 'kind': 'reflex',
     'citation': 'andersen1990[p. 146 (printed p. 148)];CDIAL[3955]',
     'evidence': "Andersen's gam- 'to go' head explicitly compares RV gam- (gacchati) and Pali gacchati, cross-referencing adhigam-/upagata-. Turner 3955 gácchati 'goes' explicitly marks √gam, cites Pali gacchati and Ashokan gacheyaṁ/gachema/gachati. The source head is the root-stem label for this matching inherited verb, not a claim that gam- itself is an attested finite inscriptional spelling."},
]
held = [
    {'record': records['upagata'], 'reason': "Andersen p. 143 explicitly compares RV/Pali upagata and cross-references upa + gam. Upa is compiled-linked and gam is proposed in this pass, but the shared saver requires existing compiled ancestry for source component parents. Hold the compound until gam's link is incorporated in the compiled graph; no rebuild in this task."},
    {'record': records['apacāy'], 'reason': "The source p. 138 compares ŚB/Pali apacāyati and cross-references apa + cāy. The installed apa parent has a contaminated rank-1 compiled edge to Turner 1188 āpa 'obtaining', not the preverb apa 'away from, off'. Do not build a compound atop that incorrect edge."},
]
(base / 'pass178-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': held}, ensure_ascii=False, indent=2) + '\n')
