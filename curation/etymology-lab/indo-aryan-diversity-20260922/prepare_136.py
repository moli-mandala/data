"""Prepare two Grangali verbs from Buddruss's manually collated survey."""
import csv
import json
import pickle
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_rhbem6usgd2mk', '6141', 'Turner 6141 gives the old dā- and productive dē- “give” verb family, including Bashkarik da- and nearby Kohistani/Shina forms. Buddruss Grangali da- is the bare “give” stem. This attaches the broad verbal article without asserting an inflectional subdivision.'),
    ('f_bbzartfashray', '10066', 'Turner 10066 māráyati “kill, strike” gives Western Dardic mā́r- in Bashkarik, Gawarbati and Tirahi and Lahnda māraṇ. Buddruss Grangali mār- has the exact kill sense and local long-vowel stem.'),
]
ids = {fid for fid, _, _ in targets}
records = {r['ID']: r for r in csv.DictReader(open('cldf/forms.csv')) if r['ID'] in ids}
assert len(records) == len(ids)
source_rows = list(csv.reader(open('data/other/forms/20260819-buddruss-grangali.csv')))
linked = {r['Form_ID'] for r in read_assignments() if r['Status'] == 'accepted'}
linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank'] == '1')
accepted = []
audit = []
for fid, parent, evidence in targets:
    r = records[fid]
    assert fid not in linked and r['Language_ID'] == 'Gng' and r['Source'].startswith('buddruss-grangali1979[')
    matches = [x for x in source_rows if len(x) > 10 and x[0] == 'Gng' and x[7] == r['Source']]
    assert len(matches) == 1, (fid, len(matches))
    raw = matches[0]
    assert raw[3] == r['Gloss'] and unicodedata.normalize('NFC', raw[2]) == unicodedata.normalize('NFC', r['Original'])
    accepted.append({
        'record': {k: r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')},
        'parent': parent,
        'kind': 'reflex',
        'citation': f"{r['Source']};CDIAL[{parent}]",
        'evidence': evidence + ' Exact manually collated source key and original hyphenated stem checked; no import or transcription change.',
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
for n, phrase in {'6141': 'Bshk. da -', '10066': 'Bshk. mār -'}.items():
    assert unicodedata.normalize('NFD', phrase) in unicodedata.normalize('NFD', primary[n]['text']), (n, phrase)
(base / 'grangali-primary001.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')
(base / 'pass136-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
held = [
    {'form': 'zū', 'gloss': 'girl', 'reason': 'May continue the daughter word, but direct Grangali sense and the separate zū “louse” homophone need targeted comparison.'},
    {'form': 'lik', 'gloss': 'write', 'reason': 'No secure directly matching Turner head established; avoid a generic write-root guess.'},
    {'form': 'bādā́m', 'gloss': 'edible almond', 'reason': 'Source explicitly calls it a Persian loan; identify the immediate donor node before adding a borrowed edge.'},
]
(base / 'pass136-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': held}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass136-review.md').write_text('\n'.join([
    '# Buddruss Grangali: give and kill',
    '',
    'Two source-backed reflex records in the manually collated Grangali survey. Exact original hyphenated stems and source keys were checked against the installed raw table; complete Turner articles are in `grangali-primary001.json`. The Stanford ILL scan was collated during ingestion but is not redistributed or accessible in this workspace, so the original image was not rechecked for this batch.',
    '',
    '| # | Grangali | Proposed etymology | Evidence |',
    '|---|---|---|---|',
    '| 1 | **da-** ‘give’ | **dádāti**, CDIAL **6141** | Turner gives the regional *da-/dē-* verbal family, including Bashkarik *da-*. |',
    '| 2 | **mār-** ‘kill’ | **māráyati**, CDIAL **10066** | Turner gives western Dardic *mār-* and Lahnda *māraṇ* with the kill/strike sense. |',
    '',
    'Two proposed `reflex` rows affect two Grangali records. **zū** ‘girl’, **lik** ‘write’, and the explicit Persian loan **bādā́m** remain held for the reasons in the decisions file. No source ingestion or rebuild.',
    '',
    'The Jambu homepage service was unavailable locally; exact compiled Turner comparanda led to full cached primary articles. Validation pending.',
]) + '\n')
print('accepted', len(accepted), 'held', len(held))
