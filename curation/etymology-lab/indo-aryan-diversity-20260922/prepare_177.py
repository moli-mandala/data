"""Audit Varli eclipse responses against CDIAL grahaṇa section 3."""
import csv
import json
import pickle
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_bey7k2h3utdu6', 'Davar Varli grihɔṇ “eclipse” matches Turner gráhaṇa CDIAL 4364 §3, alongside Lahnda girhāṇ and Gujarati gharaṇ. The local vowel shape is retained.'),
    ('f_qxrrfdufptlmq', 'Dungar Varli girhān “eclipse” closely matches Lahnda girhāṇ “eclipse” in Turner gráhaṇa CDIAL 4364 §3.'),
]
ids = {fid for fid, _ in targets}
records = {r['ID']: r for r in csv.DictReader(open('cldf/forms.csv')) if r['ID'] in ids}
assert len(records) == len(ids)
linked = {r['Form_ID'] for r in read_assignments() if r['Status'] == 'accepted'}
linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank'] == '1')
audit_rows = [json.loads(x) for x in open('data/other/forms/raw_data/20260911-dadra-varli-audit.jsonl')]
accepted, audit = [], []
for fid, evidence in targets:
    r = records[fid]
    assert fid not in linked and r['Language_ID'] in {'Bhili', 'Varli'} and r['Source'].startswith('pattanaik-koul2003varli[')
    lect = 'Davar' if r['Language_ID'] == 'Bhili' else 'Dungar'
    matches = [x for x in audit_rows if x['source'] == 'pattanaik-koul2003varli' and x['printed_page'] == 80 and x['item'] == 7 and x['lect'] == lect and x['gloss'] == 'eclipse' and x['status'] == 'ingested']
    assert len(matches) == 1, (fid, matches)
    source_row = matches[0]
    assert source_row['review'] == 'manual-page-image' and all(not x['uncertainty'] for x in source_row['forms'])
    assert r['Original'] in [x['form'] for x in source_row['forms']]
    accepted.append({'record': {k: r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')}, 'parent': '4364', 'kind': 'reflex', 'citation': f"{r['Source']};CDIAL[4364]", 'evidence': evidence + ' The installed CDIAL head is unsplit, so section 3 is recorded in the note. Existing manual page-image audit has empty reading uncertainty; exact response and compiled identity checked. No import or transcription change.'})
    audit.append({'id': fid, 'manual_page_audit': source_row, 'compiled_original': r['Original'], 'compiled_form': r['Form'], 'compiled_gloss': r['Gloss'], 'compiled_tags': r['Tags']})
primary = None
for page_num, html in enumerate(pickle.load(open('data/cdial/cdial.pickle', 'rb')), 1):
    if '<number>4364</number>' not in html:
        continue
    for p in BeautifulSoup(html, 'html.parser').select('p'):
        number = p.find('number')
        if number and number.get_text(strip=True) == '4364':
            primary = {'page': page_num, 'url': f'https://dsal.uchicago.edu/cgi-bin/app/soas_query.py?page={page_num}', 'html': str(p), 'text': p.get_text(' ', strip=True)}
            break
    if primary:
        break
assert primary and "3. Pk. gahaṇa - n. 'eclipse'" in primary['text'] and 'L. girhāṇ' in primary['text'] and 'G. gharaṇ' in primary['text']
(base / 'varli-primary016.json').write_text(json.dumps({'4364': primary}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass177-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
(base / 'pass177-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': []}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass177-review.md').write_text('\n'.join([
    '# Davar and Dungar Varli eclipse',
    '',
    'Both lects have survey responses for “eclipse” that match Turner gráhaṇa §3. The installed CDIAL parent 4364 is unsplit, and the relevant section is specified in each note. The existing ingestion audit records manual page-image review and no reading uncertainty; the selected audit objects are copied to `pass177-source-audit.json`. The complete Turner article is in `varli-primary016.json`.',
    '',
    '| # | Varli survey | Proposed etymology | Evidence |',
    '|---|---|---|---|',
    '| 1 | **grihɔṇ** (Davar) | **gráhaṇa** §3, CDIAL **4364** | Lahnda *girhāṇ* and Gujarati *gharaṇ* denote eclipse. |',
    '| 2 | **girhān** (Dungar) | **gráhaṇa** §3, CDIAL **4364** | Lahnda *girhāṇ* is an especially close form-and-sense match. |',
    '',
    'Two `reflex` rows affect two records. No source ingestion or rebuild.',
    '',
    'Validation pending.',
]) + '\n')
print('accepted', len(accepted))
