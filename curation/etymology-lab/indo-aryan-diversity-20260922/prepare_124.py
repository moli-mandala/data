"""Prepare six Khetrani kinship, dog, and water reflexes from LSI."""
import csv
import json
import pickle
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_5cfsneik3ayny', '10016', 'Turner 10016 lists Lahnda mā “mother” under mātŕ̥, exactly matching Grierson Khetrani mā. The etymon is the inherited kinship stem as classified by Turner, not a claim that nursery renewal can be excluded.'),
    ('f_2sutky56clpbu', '6481', 'Turner 6481 lists Lahnda dhī “daughter” under duhitŕ̥, exactly matching Grierson Khetrani dʰī and its kinship sense.'),
    ('f_bcnocurfv7hke', '3275', 'Turner 3275 gives Lahnda kuttā masculine “dog” and feminine °tī under *kutta. Grierson Khetrani kuttā “Dog (male)” is an exact local match.'),
    ('f_6437q4dfbkjp4', '3275', 'Turner 3275 gives Lahnda kuttā masculine “dog” under *kutta. This second Grierson Khetrani kuttā is independently printed in the general Dog row.'),
    ('f_2wghp6ja34nhc', '3275', 'Turner 3275 lists the Lahnda feminine °tī beside kuttā under *kutta. Grierson Khetrani kuttī “Bitch” matches this feminine dog form.'),
    ('f_ejgim67ff5nti', '8082', 'Turner 8082 lists Lahnda pāṇī “water” under pānī́ya. Grierson Khetrani pānī has the same water sense; its source n is retained without silent retroflexion.'),
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
for n, phrase in {'10016': 'L. mā', '6481': 'L. dhī', '3275': 'kuttā', '8082': 'pāṇī'}.items():
    assert unicodedata.normalize('NFD', phrase) in unicodedata.normalize('NFD', primary[n]['text']), (n, phrase)
(base / 'khetrani-primary004.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')
(base / 'pass124-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
held = [
    {'form': 'kuttē', 'gloss': 'Dogs', 'reason': 'Inflected plural of kuttā; defer until the survey’s morphological treatment is checked.'},
    {'form': 'gʰōṛā', 'gloss': 'Horse', 'reason': 'Two survey slots and possible areal lexical spread; verify immediate history before linking.'},
    {'form': 'abbā', 'gloss': 'Father', 'reason': 'Likely nursery/contact word; do not flatten to inherited pitŕ̥.'},
]
(base / 'pass124-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': held}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass124-review.md').write_text('\n'.join([
    '# Grierson Khetrani: mother, daughter, dog, and water',
    '',
    'Six source-backed reflex records in the digital LSI comparative vocabulary. Exact original-spelling rows and source locators were checked; complete Turner articles are in `khetrani-primary004.json`. Original physical table images have not yet been independently checked.',
    '',
    '| # | Khetrani | Proposed etymology | Evidence |',
    '|---|---|---|---|',
    '| 1 | **mā** ‘mother’ | **mātŕ̥**, CDIAL **10016** | Turner lists Lahnda *mā* directly; this does not exclude nursery renewal. |',
    '| 2 | **dʰī** ‘daughter’ | **duhitŕ̥**, CDIAL **6481** | Turner lists Lahnda *dhī* directly. |',
    '| 3–4 | **kuttā** ‘dog’ in two survey slots | **\\*kutta**, CDIAL **3275** | Turner lists Lahnda *kuttā* masculine. Both source keys were checked. |',
    '| 5 | **kuttī** ‘bitch’ | **\\*kutta**, CDIAL **3275** | Turner explicitly pairs Lahnda feminine *°tī* with *kuttā*. |',
    '| 6 | **pānī** ‘water’ | **pānī́ya**, CDIAL **8082** | Turner lists Lahnda *pāṇī*; the survey’s plain *n* is retained. |',
    '',
    'Six proposed `reflex` rows affect six Khetrani records. **kuttē**, **gʰōṛā**, and **abbā** remain held as documented in the decisions file. No source ingestion or rebuild.',
    '',
    'The Jambu homepage service was unavailable locally; exact compiled Turner comparanda led to full cached primary articles. Validation pending.',
]) + '\n')
print('accepted', len(accepted), 'held', len(held))
