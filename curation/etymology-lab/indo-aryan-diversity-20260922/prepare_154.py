"""Audit two Shumashti comparanda, including the source w > compiled v spelling."""
import csv
import json
import pickle
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_zdl25v6pvmwdo', '8265', 'Turner 8265 explicitly lists Shumashti puλ “son” under putrá; Buddruss item 60 has the identical form and sense.'),
    ('f_ltxqwn5t2z6lu', '9028-3', 'Turner 9028 §3 explicitly lists Shumashti plōṷə “spleen” under *prahaka-. Buddruss item 129 writes plōwa; the compiled sound profile maps source w to v in plōva without changing the lexical identity.'),
]
ids = {fid for fid, _, _ in targets}
records = {r['ID']: r for r in csv.DictReader(open('cldf/forms.csv')) if r['ID'] in ids}
assert len(records) == len(ids)
raw = list(csv.reader(open('data/other/forms/20260819-buddruss-grangali.csv')))
linked = {r['Form_ID'] for r in read_assignments() if r['Status'] == 'accepted'}
linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank'] == '1')
accepted, audit = [], []
for fid, parent, evidence in targets:
    r = records[fid]
    assert fid not in linked and r['Language_ID'] == 'Shum' and r['Source'].startswith('buddruss-grangali1979[')
    matches = [x for x in raw if len(x) > 10 and x[0] == 'Shum' and x[7] == r['Source']]
    assert len(matches) == 1 and matches[0][2] == r['Original'] and matches[0][3] == r['Gloss']
    accepted.append({'record': {k: r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')}, 'parent': parent, 'kind': 'reflex', 'citation': f"{r['Source']};CDIAL[{parent}]", 'evidence': evidence + ' Raw Buddruss table and persistent compiled identity audited; no ingestion or transcription change.'})
    audit.append({'id': fid, 'raw_source_row': matches[0], 'compiled_original': r['Original'], 'compiled_form': r['Form'], 'compiled_gloss': r['Gloss'], 'source': r['Source']})
primary = {}
for page_num, html in enumerate(pickle.load(open('data/cdial/cdial.pickle', 'rb')), 1):
    if not any(f'<number>{n}</number>' in html for n in ('8265', '9028')):
        continue
    for p in BeautifulSoup(html, 'html.parser').select('p'):
        number = p.find('number')
        if number and number.get_text(strip=True) in {'8265', '9028'}:
            n = number.get_text(strip=True)
            primary[n] = {'page': page_num, 'url': f'https://dsal.uchicago.edu/cgi-bin/app/soas_query.py?page={page_num}', 'html': str(p), 'text': p.get_text(' ', strip=True)}
    if len(primary) == 2:
        break
assert len(primary) == 2
assert 'Shum. puλ' in primary['8265']['text']
assert 'Shum. plōṷə' in primary['9028']['text']
(base / 'shumashti-primary004.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')
(base / 'pass154-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
(base / 'pass154-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': [
    {'form': 'zū', 'gloss': 'girl', 'reason': 'No direct Turner Shumashti comparandum established for this source sense.'},
    {'form': 'yeman', 'gloss': 'winter', 'reason': 'Potential Iranian contact; immediate donor unverified.'},
    {'form': 'lāk', 'gloss': 'grape', 'reason': 'Turner labels this branch as a borrowing; immediate donor is unresolved.'},
]}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass154-review.md').write_text('\n'.join([
    '# Buddruss Shumashti: son and spleen',
    '',
    'Two direct Shumashti comparanda in full Turner articles. The exact manually collated Buddruss rows and compiled identities are audited in `pass154-source-audit.json`; the source *plōwa* is normalized to compiled *plōva*. Full Turner articles are in `shumashti-primary004.json`. The Stanford ILL scan was collated at ingestion but is unavailable in the workspace for this batch’s independent image recheck.',
    '',
    '| # | Shumashti | Proposed etymology | Evidence |',
    '|---|---|---|---|',
    '| 1 | **puλ** ‘son’ | **putrá**, CDIAL **8265** | Turner lists Shumashti *puλ* ‘son’ directly. |',
    '| 2 | **plōva** ‘spleen’ (source *plōwa*) | **\\*prahaka-**, CDIAL **9028-3** | Turner §3 lists Shumashti *plōṷə*; the selected subsection isolates the spleen branch. |',
    '',
    'Two `reflex` rows affect two records. **zū** ‘girl’, **yeman** ‘winter’ and **lāk** ‘grape’ remain held. No source ingestion or rebuild.',
    '',
    'Validation pending.',
]) + '\n')
print('accepted', len(accepted))
