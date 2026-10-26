"""Prepare two direct Shumashti comparisons in Buddruss's survey."""
import csv
import json
import pickle
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_e6azn7np26hb2', '6793', 'Turner 6793 directly lists Shumashti dār and Ningalami dār in the hill/mountain branch of dhā́rā². Buddruss gives Shumashti dār “mountain” in the comparative row.'),
    ('f_jrvrfl7okgvaw', '6757', 'Turner 6757 directly lists Shumashti lyēmī “roof” under dhárman¹. Buddruss gives exactly lyēmī “roof” in the comparative row.'),
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
    assert fid not in linked and r['Language_ID'] == 'Shum' and r['Source'].startswith('buddruss-grangali1979[')
    matches = [x for x in source_rows if len(x) > 10 and x[0] == 'Shum' and x[7] == r['Source']]
    assert len(matches) == 1, (fid, len(matches))
    raw = matches[0]
    assert raw[3] == r['Gloss'] and unicodedata.normalize('NFC', raw[2]) == unicodedata.normalize('NFC', r['Original'])
    accepted.append({
        'record': {k: r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')},
        'parent': parent,
        'kind': 'reflex',
        'citation': f"{r['Source']};CDIAL[{parent}]",
        'evidence': evidence + ' Exact manually collated comparative source row and original spelling checked; no import or transcription change.',
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
for n, phrase in {'6793': 'Shum. Niṅg. dār', '6757': 'Shum. lyēmī'}.items():
    assert unicodedata.normalize('NFD', phrase) in unicodedata.normalize('NFD', primary[n]['text']), (n, phrase)
(base / 'shumashti-primary001.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')
(base / 'pass144-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
held = [
    {'form': 'lāk', 'gloss': 'grape', 'reason': 'Turner 6628 explicitly cites Shumashti lāk but marks it as a loan from an Indo-Aryan *drākkha form; identify the immediate donor before assigning a borrowed edge.'},
    {'form': 'zū', 'gloss': 'girl', 'reason': 'Turner 6481 cites Shumashti zū as daughter, but the source gloss is girl; review semantic scope and the distinct louse homonym.'},
]
(base / 'pass144-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': held}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass144-review.md').write_text('\n'.join([
    '# Buddruss Shumashti: mountain and roof',
    '',
    'Two directly attested Turner comparisons in the Shumashti comparison rows of Buddruss’s Grangali survey. Exact manually collated source rows and compiled identities were checked; full Turner articles are in `shumashti-primary001.json`. The Stanford ILL scan was checked during ingestion but is unavailable in this workspace for recheck.',
    '',
    '| # | Shumashti | Proposed etymology | Evidence |',
    '|---|---|---|---|',
    '| 1 | **dār** ‘mountain’ | **dhā́rā²**, CDIAL **6793** | Turner cites Shumashti *dār* directly in the mountain branch. |',
    '| 2 | **lyēmī** ‘roof’ | **dhárman¹**, CDIAL **6757** | Turner cites Shumashti *lyēmī* ‘roof’ directly. |',
    '',
    'Two proposed `reflex` rows affect two Shumashti records. **lāk** ‘grape’ is marked by Turner as borrowed from an Indo-Aryan source and awaits an immediate donor analysis; **zū** ‘girl’ awaits semantic/homophone review. No source ingestion or rebuild.',
    '',
    'The Jambu homepage service was unavailable locally; exact compiled Turner comparanda led to full cached primary articles. Validation pending.',
]) + '\n')
print('accepted', len(accepted), 'held', len(held))
