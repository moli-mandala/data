"""Prepare four Khetrani basic-vocabulary reflexes with local Turner matches."""
import csv
import json
import pickle
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_twbws6f3ljrrk', '4655', 'Turner 4655 lists Lahnda cār “four” under the catvā́ri subdivision of catvā́raḥ. Grierson Khetrani cār has the same numeral sense and exact form.'),
    ('f_pyrc4cgrxciru', '11616', 'Turner 11616 lists Lahnda vīh “20” under viṁśatí. Grierson Khetrani vīh exactly matches.'),
    ('f_4jo3273wijvma', '7756', 'Turner 7756 lists Lahnda pēr “foot” under *padara. Grierson Khetrani pēr exactly matches this local form and sense.'),
    ('f_qzcunr4bfauwe', '12196', 'Turner 12196 lists Lahnda vāt masculine “mouth” under the nominal branch of vyā́tta. Grierson Khetrani vāt has the same form and specific mouth sense.'),
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
for n, phrase in {'4655': 'L. P. Ku. N. cār', '11616': 'L. vīh', '7756': 'L. pair , pêr , pēr m.', '12196': 'L. vāt m.'}.items():
    text = primary[n]['text']
    assert unicodedata.normalize('NFD', phrase) in unicodedata.normalize('NFD', text), (n, phrase)
(base / 'khetrani-primary005.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')
(base / 'pass128-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
held = [
    {'form': 'cālī ta dak', 'gloss': 'Fifty', 'reason': 'Compound with an uncertain source transcription and arithmetic analysis; do not link wholesale to a single numeral.'},
]
(base / 'pass128-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': held}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass128-review.md').write_text('\n'.join([
    '# Grierson Khetrani: four, twenty, foot, and mouth',
    '',
    'Four source-backed reflex records in the digital LSI comparative vocabulary. Exact original-spelling rows and source locators were checked; complete Turner articles are in `khetrani-primary005.json`. Original physical table images have not yet been independently checked.',
    '',
    '| # | Khetrani | Proposed etymology | Evidence |',
    '|---|---|---|---|',
    '| 1 | **cār** ‘four’ | **catvā́raḥ**, CDIAL **4655** | Turner gives Lahnda *cār* under the *catvā́ri* branch. |',
    '| 2 | **vīh** ‘twenty’ | **viṁśatí**, CDIAL **11616** | Turner gives Lahnda *vīh* directly. |',
    '| 3 | **pēr** ‘foot’ | **\\*padara**, CDIAL **7756** | Turner gives Lahnda *pēr* directly. |',
    '| 4 | **vāt** ‘mouth’ | **vyā́tta**, CDIAL **12196** | Turner gives Lahnda *vāt* in the nominal mouth branch, distinct from *vāt* ‘news’. |',
    '',
    'Four proposed `reflex` rows affect four Khetrani records. **cālī ta dak** ‘fifty’ remains held because the whole phrase should not be flattened to one numeral ancestor. No source ingestion or rebuild.',
    '',
    'The Jambu homepage service was unavailable locally; exact compiled Turner comparanda led to full cached primary articles. Validation pending.',
]) + '\n')
print('accepted', len(accepted), 'held', len(held))
