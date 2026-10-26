"""Prepare Khetrani horse and deer reflexes from the LSI comparative vocabulary."""
import csv
import json
import pickle
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_7z7rxfajltkmo', '4516', 'Turner 4516 gives Lahnda ghoṛā masculine “horse” under ghōṭa. Grierson Khetrani gʰōṛā “Horse (male)” matches form, gender and sense.'),
    ('f_dtgtje5xjndoe', '4516', 'Turner 4516 gives Lahnda ghoṛā masculine “horse” under ghōṭa. The second Grierson Khetrani gʰōṛā is a separately keyed general Horse survey row and is independently linked.'),
    ('f_htub76spczws4', '13982', 'Turner 13982 gives Ludhiana Punjabi haran masculine “deer” and Lahnda haraṇ under hariṇá. Grierson Khetrani haran “Male deer” shares the exact regional consonant stem and gender.'),
    ('f_boxtdm7l37q2q', '13982', 'Turner 13982 gives Ludhiana Punjabi harnī feminine “deer” and Lahnda harṇī under hariṇá. Grierson Khetrani harnī “Female deer” exactly matches the regional feminine form and sense.'),
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
for n, phrase in {'4516': 'L. P. ghoṛā', '13982': 'ludh. haran m., harnī f.'}.items():
    assert unicodedata.normalize('NFD', phrase) in unicodedata.normalize('NFD', primary[n]['text']), (n, phrase)
(base / 'khetrani-primary008.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')
(base / 'pass139-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
held = [
    {'form': 'taṭṭul', 'gloss': 'Bull', 'reason': 'The source animal form has no secure precise Turner parent yet.'},
    {'form': 'buj', 'gloss': 'He goat', 'reason': 'Possible buck/goat forms cross families; confirm the immediate branch.'},
]
(base / 'pass139-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': held}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass139-review.md').write_text('\n'.join([
    '# Grierson Khetrani: horse and deer',
    '',
    'Four source-backed reflex records in the digital LSI comparative vocabulary. Exact original-spelling rows and source locators were checked; complete Turner articles are in `khetrani-primary008.json`. Original physical table images have not yet been independently checked.',
    '',
    '| # | Khetrani | Proposed etymology | Evidence |',
    '|---|---|---|---|',
    '| 1–2 | **gʰōṛā** ‘horse’ (male and general survey slots) | **ghōṭa**, CDIAL **4516** | Turner gives Lahnda *ghoṛā* masculine directly; both source keys checked. |',
    '| 3 | **haran** ‘male deer’ | **hariṇá**, CDIAL **13982** | Turner gives Ludhiana Punjabi *haran* masculine. |',
    '| 4 | **harnī** ‘female deer’ | **hariṇá**, CDIAL **13982** | Turner gives Ludhiana Punjabi *harnī* feminine. |',
    '',
    'Four proposed `reflex` rows affect four Khetrani records. **taṭṭul** ‘bull’ and **buj** ‘he goat’ remain held for immediate branch uncertainty. No source ingestion or rebuild.',
    '',
    'The Jambu homepage service was unavailable locally; exact compiled Turner comparanda led to full cached primary articles. Validation pending.',
]) + '\n')
print('accepted', len(accepted), 'held', len(held))
