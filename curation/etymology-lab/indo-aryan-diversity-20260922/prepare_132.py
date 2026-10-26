"""Prepare three Khetrani basic verbs from the LSI comparative vocabulary."""
import csv
import json
import pickle
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_vlbazucdsstgi', '9871', 'Turner 9871 gives Lahnda maraṇ “to die” under máratē and many regional mar- stems. Grierson Khetrani mar “Die” matches the local stem and sense; the source citation is a survey elicitation form, not an asserted inflection.'),
    ('f_ptl7novwkratu', '6141', 'Turner 6141 gives the productive Indo-Aryan dē- “give” stem under dádāti, including Lahnda devaṇ and nearby direct dē- forms. Grierson Khetrani dē has the same give sense.'),
    ('f_5rxsgq6gnupz6', '10066', 'Turner 10066 gives Lahnda māraṇ and Punjabi mārnā “to kill, strike” under māráyati. Grierson Khetrani mār “Beat” matches the strike/beat branch; the distinct mar “die” stem is linked separately.'),
]
ids = {fid for fid, _, _ in targets}
records = {r['ID']: r for r in csv.DictReader(open('cldf/forms.csv')) if r['ID'] in ids}
assert len(records) == len(ids)
source_rows = list(csv.reader(open('data/other/forms/20260813-grierson-lsi.csv')))
linked = {r['Form_ID'] for r in read_assignments() if r['Status'] == 'accepted'}
linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank'] == '1')
accepted = []
audit = []
for fid, parent, evidence in targets:
    r = records[fid]
    assert fid not in linked and r['Language_ID'] == 'khet' and 'LSI-KHETRANI' in r['Tags']
    matches = [x for x in source_rows if len(x) > 10 and x[0] == 'LSI-KHETRANI' and x[7] == r['Source']]
    assert len(matches) == 1, (fid, len(matches))
    raw = matches[0]
    assert raw[3] == r['Gloss'] and unicodedata.normalize('NFC', raw[2]) == unicodedata.normalize('NFC', r['Original'])
    accepted.append({
        'record': {k: r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')},
        'parent': parent,
        'kind': 'reflex',
        'citation': f"{r['Source']};CDIAL[{parent}]",
        'evidence': evidence + ' Exact digital-edition key and original spelling checked; no import or transcription change.',
    })
    audit.append({'id': fid, 'raw_source_row': raw, 'compiled_original': r['Original'], 'compiled_form': r['Form'], 'source': r['Source']})

wanted = {parent for _, parent, _ in targets}
primary = {}
for page_num, html in enumerate(pickle.load(open('data/cdial/cdial.pickle', 'rb')), 1):
    if not any(f'<number>{n}</number>' in html for n in wanted - primary.keys()):
        continue
    soup = BeautifulSoup(html, 'html.parser')
    for paragraph in soup.select('p'):
        number = paragraph.find('number')
        if number and number.get_text(strip=True) in wanted:
            n = number.get_text(strip=True)
            primary[n] = {'page': page_num, 'url': f'https://dsal.uchicago.edu/cgi-bin/app/soas_query.py?page={page_num}', 'html': str(paragraph), 'text': paragraph.get_text(' ', strip=True)}
    if len(primary) == len(wanted):
        break
assert len(primary) == len(wanted)
for n, phrase in {'9871': 'L. maraṇ', '6141': 'L. devaṇ', '10066': 'L. māraṇ'}.items():
    assert unicodedata.normalize('NFD', phrase) in unicodedata.normalize('NFD', primary[n]['text']), (n, phrase)
(base / 'khetrani-primary006.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')
(base / 'pass132-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
held = [
    {'form': 'vanj', 'gloss': 'Go', 'reason': 'May be a compound or altered go stem; verify immediate history.'},
    {'form': 'jul', 'gloss': 'Go', 'reason': 'A separate go synonym with uncertain branch.'},
    {'form': 'catt', 'gloss': 'Go', 'reason': 'A separate go synonym with uncertain branch.'},
]
(base / 'pass132-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': held}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass132-review.md').write_text('\n'.join([
    '# Grierson Khetrani: die, give, and beat',
    '',
    'Three source-backed reflex records in the digital LSI comparative vocabulary. Exact original-spelling rows and source locators were checked; complete Turner articles are in `khetrani-primary006.json`. Original physical table images have not yet been independently checked.',
    '',
    '| # | Khetrani | Proposed etymology | Evidence |',
    '|---|---|---|---|',
    '| 1 | **mar** ‘die’ | **máratē**, CDIAL **9871** | Turner gives Lahnda *maraṇ* ‘to die’ and regional *mar-* stems. |',
    '| 2 | **dē** ‘give’ | **dádāti**, CDIAL **6141** | Turner gives Lahnda *devaṇ* and productive regional *dē-* stems. |',
    '| 3 | **mār** ‘beat’ | **māráyati**, CDIAL **10066** | Turner gives Lahnda *māraṇ* ‘to kill, strike’ and Punjabi *mārnā*. |',
    '',
    'Three proposed `reflex` rows affect three Khetrani records. The three printed **Go** synonyms remain held for separate stem/branch analysis. **mar** ‘die’ and **mār** ‘beat’ are kept distinct. No source ingestion or rebuild.',
    '',
    'The Jambu homepage service was unavailable locally; exact compiled Turner comparanda led to full cached primary articles. Validation pending.',
]) + '\n')
print('accepted', len(accepted), 'held', len(held))
