"""Audit Davar and Dungar sister forms against Turner bhaginī."""
import csv
import json
import pickle
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_wc7c33ig2kiza', 'Davar Varli bɔhɔnīs “sister” has the b-h-n stem and feminine vowel of western bahiṇī/bahīṇ “sister” in Turner bhaginī CDIAL 9349. Source final -s is retained without a morphological analysis.'),
    ('f_wcwvwbjph7bwg', 'Dungar Varli bɔhɔnīs “sister” repeats the same b-h-n sister stem found in Turner bhaginī CDIAL 9349, with Gujarati bahεn and Marathi bahīṇ comparanda. Source final -s is retained without a morphological analysis.'),
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
    matches = [x for x in audit_rows if x['source'] == 'pattanaik-koul2003varli' and x['printed_page'] == 81 and x['item'] == 58 and x['lect'] == lect and x['gloss'] == 'sister' and x['status'] == 'ingested']
    assert len(matches) == 1, (fid, matches)
    source_row = matches[0]
    assert source_row['review'] == 'manual-page-image' and all(not x['uncertainty'] for x in source_row['forms'])
    assert r['Original'] in [x['form'] for x in source_row['forms']]
    accepted.append({'record': {k: r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')}, 'parent': '9349', 'kind': 'reflex', 'citation': f"{r['Source']};CDIAL[9349]", 'evidence': evidence + ' Existing manual page-image audit has empty reading uncertainty; exact response and compiled identity checked. No import or transcription change.'})
    audit.append({'id': fid, 'manual_page_audit': source_row, 'compiled_original': r['Original'], 'compiled_form': r['Form'], 'compiled_gloss': r['Gloss'], 'compiled_tags': r['Tags']})
primary = None
for page_num, html in enumerate(pickle.load(open('data/cdial/cdial.pickle', 'rb')), 1):
    if '<number>9349</number>' not in html:
        continue
    for p in BeautifulSoup(html, 'html.parser').select('p'):
        number = p.find('number')
        if number and number.get_text(strip=True) == '9349':
            primary = {'page': page_num, 'url': f'https://dsal.uchicago.edu/cgi-bin/app/soas_query.py?page={page_num}', 'html': str(p), 'text': p.get_text(' ', strip=True)}
            break
    if primary:
        break
assert primary and "G. bahεn , bεn 'sister'" in primary['text'] and 'M. bahīṇ' in primary['text']
(base / 'varli-primary020.json').write_text(json.dumps({'9349': primary}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass185-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
(base / 'pass185-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': [
    {'form': 'bābhis', 'gloss': 'sister-in-law', 'reason': 'Adjacent but distinct kinship word, with possible borrowed or derived transmission; not folded into bhaginī.'},
]}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass185-review.md').write_text('\n'.join([
    '# Davar and Dungar Varli sister',
    '',
    'Both lects give the same bɔhɔnīs sister form. The b-h-n-ī stem matches western Indo-Aryan bhaginī descendants in Turner, especially Gujarati bahεn and Marathi bahīṇ. The source final -s is not morphologically explained. The existing ingestion audit records manual page-image review and no reading uncertainty; selected audit objects are in `pass185-source-audit.json`, and the full Turner article is in `varli-primary020.json`.',
    '',
    '| # | Varli survey | Proposed etymology | Evidence |',
    '|---|---|---|---|',
    '| 1–2 | **bɔhɔnīs** ‘sister’ (Davar and Dungar, separate keys) | **bhaginī**, CDIAL **9349** | Gujarati *bahεn* and Marathi *bahīṇ* are close regional forms. |',
    '',
    'Two `reflex` rows affect two records. **bābhis** “sister-in-law” remains separate. No source ingestion or rebuild.',
    '',
    'Validation pending.',
]) + '\n')
print('accepted', len(accepted))
