"""Audit Varli nephew and niece responses against Turner bhāginēya."""
import csv
import json
import pickle
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_4ndrtxaw6bhh6', 'Davar Varli bhaṇjɔ “nephew” matches the masculine bhāṇj- forms in Turner bhāginēya CDIAL 9433, especially Gujarati bhāṇɔ and Hindi bhā̃jā.'),
    ('f_cmm4tsncavwpi', 'Dungar Varli bhanja “nephew” matches the masculine bhāṇj- forms in Turner bhāginēya CDIAL 9433, including Punjabi bhāṇjā.'),
    ('f_57emvwnuqpwww', 'Davar Varli bhaṇji “niece” matches the feminine bhāṇjī in Turner bhāginēya CDIAL 9433, explicitly listed for Gujarati.'),
    ('f_gf5h7ajek4fda', 'Dungar Varli bhanji “niece” matches the feminine bhāṇjī in Turner bhāginēya CDIAL 9433, explicitly listed for Gujarati.'),
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
    item = int(r['Source'].split('item ')[1].split(',')[0])
    matches = [x for x in audit_rows if x['source'] == 'pattanaik-koul2003varli' and x['printed_page'] == 80 and x['item'] == item and x['lect'] == lect and x['gloss'] == r['Gloss'] and x['status'] == 'ingested']
    assert len(matches) == 1, (fid, matches)
    source_row = matches[0]
    assert source_row['review'] == 'manual-page-image' and all(not x['uncertainty'] for x in source_row['forms'])
    assert r['Original'] in [x['form'] for x in source_row['forms']]
    accepted.append({'record': {k: r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')}, 'parent': '9433', 'kind': 'reflex', 'citation': f"{r['Source']};CDIAL[9433]", 'evidence': evidence + ' The survey gives broad nephew/niece glosses, while the etymon specifically denotes a sister’s child; no local kinship restriction is inferred. Existing manual page-image audit has empty reading uncertainty; exact response and compiled identity checked. No import or transcription change.'})
    audit.append({'id': fid, 'manual_page_audit': source_row, 'compiled_original': r['Original'], 'compiled_form': r['Form'], 'compiled_gloss': r['Gloss'], 'compiled_tags': r['Tags']})
primary = None
for page_num, html in enumerate(pickle.load(open('data/cdial/cdial.pickle', 'rb')), 1):
    if '<number>9433</number>' not in html:
        continue
    for p in BeautifulSoup(html, 'html.parser').select('p'):
        number = p.find('number')
        if number and number.get_text(strip=True) == '9433':
            primary = {'page': page_num, 'url': f'https://dsal.uchicago.edu/cgi-bin/app/soas_query.py?page={page_num}', 'html': str(p), 'text': p.get_text(' ', strip=True)}
            break
    if primary:
        break
assert primary and 'G. bhāṇej' in primary['text'] and 'bhāṇjī' in primary['text'] and "sister's daughter" in primary['text']
(base / 'varli-primary017.json').write_text(json.dumps({'9433': primary}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass179-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
(base / 'pass179-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': [
    {'form': 'bhɔtrijo / bhɔtriji', 'gloss': 'nephew / niece', 'reason': 'Separate Davar alternatives likely follow a brother’s-child branch; the broad survey glosses do not establish the precise kin relation or stem analysis.'},
]}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass179-review.md').write_text('\n'.join([
    '# Davar and Dungar Varli nephew and niece',
    '',
    'Four survey responses match Turner bhāginēya and its masculine and feminine bhāṇj- descendants. The survey glosses “nephew” and “niece” broadly, so the etymon’s more specific sister’s-child sense is not imposed on the local use. The existing ingestion audit records manual page-image review and no reading uncertainty for all four responses; selected objects are in `pass179-source-audit.json`, and the full Turner article is in `varli-primary017.json`.',
    '',
    '| # | Varli survey | Proposed etymology | Evidence |',
    '|---|---|---|---|',
    '| 1–2 | **bhaṇjɔ** (Davar), **bhanja** (Dungar) ‘nephew’ | **bhāginēya**, CDIAL **9433** | Gujarati *bhāṇɔ* and Punjabi *bhāṇjā* are direct masculine comparisons. |',
    '| 3–4 | **bhaṇji** (Davar), **bhanji** (Dungar) ‘niece’ | **bhāginēya**, CDIAL **9433** | Gujarati *bhāṇjī* supplies the feminine branch. |',
    '',
    'Four `reflex` rows affect four records. The alternative Davar **bhɔtrijo/bhɔtriji** forms remain held for separate kinship analysis. No source ingestion or rebuild.',
    '',
    'Validation pending.',
]) + '\n')
print('accepted', len(accepted))
