"""Audit Varli bride, bridegroom and husband navr- survey slots."""
import csv
import json
import pickle
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_5mp476cl2llfy', 'Davar Varli nɔvri “bride” matches Turner *navavaraka CDIAL 7013, Marathi navrī “bride”.'),
    ('f_mj7e5j5jdcqpk', 'Dungar Varli nɔvɔri “bride” matches Turner *navavaraka CDIAL 7013, Marathi navrī “bride”, with an intervening local vowel retained.'),
    ('f_xsonoeaxcpgho', 'Davar Varli nɔvrɔ “bridegroom” matches Turner *navavaraka CDIAL 7013, Marathi navrā “bridegroom”.'),
    ('f_syg5wbdqbp74a', 'Dungar Varli nɔvɔra “bridegroom” matches Turner *navavaraka CDIAL 7013, Marathi navrā “bridegroom”.'),
    ('f_nab4caulqraqa', 'The same Davar Varli nɔvrɔ is independently elicited as “husband”; Turner *navavaraka CDIAL 7013 has the identical Marathi bridegroom form navrā. The broader local husband sense is preserved, not read into Turner.'),
    ('f_7xo6rt5xdepjk', 'The same Dungar Varli nɔvɔra is independently elicited as “husband”; Turner *navavaraka CDIAL 7013 has the related Marathi bridegroom form navrā. The broader local husband sense is preserved, not read into Turner.'),
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
    accepted.append({'record': {k: r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')}, 'parent': '7013', 'kind': 'reflex', 'citation': f"{r['Source']};CDIAL[7013]", 'evidence': evidence + ' Existing manual page-image audit has empty reading uncertainty; exact response and compiled identity checked. No import or transcription change.'})
    audit.append({'id': fid, 'manual_page_audit': source_row, 'compiled_original': r['Original'], 'compiled_form': r['Form'], 'compiled_gloss': r['Gloss'], 'compiled_tags': r['Tags']})
primary = None
for page_num, html in enumerate(pickle.load(open('data/cdial/cdial.pickle', 'rb')), 1):
    if '<number>7013</number>' not in html:
        continue
    for p in BeautifulSoup(html, 'html.parser').select('p'):
        number = p.find('number')
        if number and number.get_text(strip=True) == '7013':
            primary = {'page': page_num, 'url': f'https://dsal.uchicago.edu/cgi-bin/app/soas_query.py?page={page_num}', 'html': str(p), 'text': p.get_text(' ', strip=True)}
            break
    if primary:
        break
assert primary and "M. navrā , naurā m. 'bridegroom" in primary['text'] and "navrī f. 'bride" in primary['text']
(base / 'varli-primary018.json').write_text(json.dumps({'7013': primary}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass180-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
(base / 'pass180-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': [
    {'form': 'dhɔni', 'gloss': 'husband', 'reason': 'Distinct Davar alternative in the same item; no donor or Turner branch assigned here.'},
]}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass180-review.md').write_text('\n'.join([
    '# Davar and Dungar Varli bride and bridegroom forms',
    '',
    'Six source-defined slots carry the navr- family: bride, bridegroom and a repeated husband elicitation in each lect. Turner *navavaraka lists Marathi navrī “bride” and navrā “bridegroom”. The broader husband sense is recorded as the local survey gloss. Existing manual page-image source audits have no reading uncertainty and are copied to `pass180-source-audit.json`; the complete Turner article is in `varli-primary018.json`.',
    '',
    '| # | Varli survey | Proposed etymology | Evidence |',
    '|---|---|---|---|',
    '| 1–2 | **nɔvri** (Davar), **nɔvɔri** (Dungar) ‘bride’ | **\\*navavaraka**, CDIAL **7013** | Marathi *navrī* “bride” is explicit. |',
    '| 3–4 | **nɔvrɔ** (Davar), **nɔvɔra** (Dungar) ‘bridegroom’ | **\\*navavaraka**, CDIAL **7013** | Marathi *navrā* “bridegroom” is explicit. |',
    '| 5–6 | The same forms independently elicited as ‘husband’ | **\\*navavaraka**, CDIAL **7013** | Source meaning is broader than Turner’s bridegroom gloss. |',
    '',
    'Six `reflex` rows affect six records. The distinct Davar alternative **dhɔni** “husband” remains held. No source ingestion or rebuild.',
    '',
    'Validation pending.',
]) + '\n')
print('accepted', len(accepted))
