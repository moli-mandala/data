"""Prepare five Khetrani body, animal and material terms from LSI."""
import csv
import json
import pickle
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_krnwxbhq4yidm', '43', 'Turner 43 gives Lahnda akkh “eye” and Awankari akh with plural -iã̄ under ákṣi. Grierson Khetrani akʰī̃ has the same eye stem, with source final nasalization retained.'),
    ('f_j4rivhb5dcpvk', '5228', 'Turner 5228 gives Lahnda jibbh and Awankari jībh “tongue” under jihvā́. Grierson Khetrani jībʰ exactly matches the latter local consonant and vowel pattern.'),
    ('f_d7u33uvogw2oi', '11158', 'Turner 11158 gives Lahnda lohā “iron” and Awankari lōˋā under lōhá. Grierson Khetrani lōhā has the same material sense and stem.'),
    ('f_zsbm3urmt5z5e', '2387', 'Turner 2387 gives Lahnda uṭṭh “camel” under úṣṭra. Grierson Khetrani uṭʰ has the same local camel sense; the source does not mark the geminate.'),
    ('f_rpe7gaponyy74', '4054', 'Turner 4054 gives Punjabi gadhā “ass” beside Lahnda gaḍḍãh under gardabhá. Grierson Khetrani gadʰā shares the nearby Indo-Aryan donkey form and exact sense.'),
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
for n, phrase in {'43': 'awāṇ. akh', '5228': 'awāṇ. jībh', '11158': 'L. lohā', '2387': 'L. P. uṭṭh', '4054': 'P. gadhā'}.items():
    assert unicodedata.normalize('NFD', phrase) in unicodedata.normalize('NFD', primary[n]['text']), (n, phrase)
(base / 'khetrani-primary007.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')
(base / 'pass135-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
held = [
    {'form': 'gʰair', 'gloss': 'House', 'reason': 'Likely of the ghara family, but the source vowel sequence needs a closer immediate-form check.'},
    {'form': 'kukkur', 'gloss': 'Cock', 'reason': 'Do not confuse the bird word with the homophonous dog family; check the printed source.'},
    {'form': 'sunnā', 'gloss': 'Gold', 'reason': 'Turner Khetrani sunnā “beautiful” is a false friend for this source gloss.'},
]
(base / 'pass135-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': held}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass135-review.md').write_text('\n'.join([
    '# Grierson Khetrani: eye, tongue, iron, camel, and ass',
    '',
    'Five source-backed reflex records in the digital LSI comparative vocabulary. Exact original-spelling rows and source locators were checked; complete Turner articles are in `khetrani-primary007.json`. Original physical table images have not yet been independently checked.',
    '',
    '| # | Khetrani | Proposed etymology | Evidence |',
    '|---|---|---|---|',
    '| 1 | **akʰī̃** ‘eye’ | **ákṣi**, CDIAL **43** | Turner gives Lahnda *akkh* and Awankari *akh* with a plural in *-iã̄*. |',
    '| 2 | **jībʰ** ‘tongue’ | **jihvā́**, CDIAL **5228** | Turner gives local Awankari *jībh* directly. |',
    '| 3 | **lōhā** ‘iron’ | **lōhá**, CDIAL **11158** | Turner gives Lahnda *lohā* and Awankari *lōˋā*. |',
    '| 4 | **uṭʰ** ‘camel’ | **úṣṭra**, CDIAL **2387** | Turner gives Lahnda *uṭṭh* ‘camel’; source gemination is not supplied. |',
    '| 5 | **gadʰā** ‘ass’ | **gardabhá**, CDIAL **4054** | Turner gives neighbouring Punjabi *gadhā* and Lahnda *gaḍḍãh*. |',
    '',
    'Five proposed `reflex` rows affect five Khetrani records. **gʰair** ‘house’, **kukkur** ‘cock’, and **sunnā** ‘gold’ remain held as documented in the decisions file. No source ingestion or rebuild.',
    '',
    'The Jambu homepage service was unavailable locally; exact compiled Turner comparanda led to full cached primary articles. Validation pending.',
]) + '\n')
print('accepted', len(accepted), 'held', len(held))
