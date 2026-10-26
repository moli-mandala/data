"""Audit two Dungar Varli lexical responses against Turner."""
import csv
import json
import pickle
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_nta3cnavo7rfc', '5362', 'Dungar Varli jhāṛi “forest” matches Turner 5362 jhāṭa section 1, with Oriya jhāṛī “thicket, low jungle” and Hindi jhāṛī “bush”; the older stem is explicitly flagged as possibly Munda in Turner.'),
    ('f_sl3h56oshaefc', '6261', 'Dungar Varli dada “elder brother” matches Turner 6261 *dādda, including Marathi dādā “elder brother” and several neighboring Indo-Aryan elder-brother senses.'),
]
ids = {fid for fid, _, _ in targets}
records = {r['ID']: r for r in csv.DictReader(open('cldf/forms.csv')) if r['ID'] in ids}
assert len(records) == len(ids)
linked = {r['Form_ID'] for r in read_assignments() if r['Status'] == 'accepted'}
linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank'] == '1')
audit_rows = [json.loads(x) for x in open('data/other/forms/raw_data/20260911-dadra-varli-audit.jsonl')]
accepted, audit = [], []
for fid, parent, evidence in targets:
    r = records[fid]
    assert fid not in linked and r['Language_ID'] == 'Varli' and r['Source'].startswith('pattanaik-koul2003varli[')
    item = int(r['Source'].split('item ')[1].split(',')[0])
    page = int(r['Source'].split('[p. ')[1].split(',')[0])
    matches = [x for x in audit_rows if x['source'] == 'pattanaik-koul2003varli' and x['printed_page'] == page and x['item'] == item and x['lect'] == 'Dungar' and x['gloss'] == r['Gloss'] and x['status'] == 'ingested']
    assert len(matches) == 1, (fid, matches)
    source_row = matches[0]
    assert source_row['review'] == 'manual-page-image' and all(not x['uncertainty'] for x in source_row['forms'])
    assert r['Original'] in [x['form'] for x in source_row['forms']]
    accepted.append({'record': {k: r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')}, 'parent': parent, 'kind': 'reflex', 'citation': f"{r['Source']};CDIAL[{parent}]", 'evidence': evidence + ' Existing manual page-image audit has empty reading uncertainty; exact response and compiled identity checked. No import or transcription change.'})
    audit.append({'id': fid, 'manual_page_audit': source_row, 'compiled_original': r['Original'], 'compiled_form': r['Form'], 'compiled_gloss': r['Gloss'], 'compiled_tags': r['Tags']})
wanted = {'5362', '6261'}
primary = {}
for page_num, html in enumerate(pickle.load(open('data/cdial/cdial.pickle', 'rb')), 1):
    if not any(f'<number>{n}</number>' in html for n in wanted - primary.keys()):
        continue
    for p in BeautifulSoup(html, 'html.parser').select('p'):
        number = p.find('number')
        if number and number.get_text(strip=True) in wanted:
            n = number.get_text(strip=True)
            primary[n] = {'page': page_num, 'url': f'https://dsal.uchicago.edu/cgi-bin/app/soas_query.py?page={page_num}', 'html': str(p), 'text': p.get_text(' ', strip=True)}
    if len(primary) == len(wanted):
        break
assert len(primary) == len(wanted)
assert 'Or. jhāṛa' in primary['5362']['text'] and 'low jungle' in primary['5362']['text']
assert "M. dādā m. 'elder brother'" in primary['6261']['text']
(base / 'varli-primary014.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')
(base / 'pass175-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
(base / 'pass175-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': [
    {'form': 'jɔŋgɔḷ', 'gloss': 'forest', 'reason': 'The Davar source audit retains an unresolved capital-L/C transcription; likely Turner 5177 but held pending spelling check.'},
    {'form': 'moṭa bhās', 'gloss': 'elder brother', 'reason': 'Davar compound needs precise segmentation and etymology of bhās; not bundled with the separate Dungar dada response.'},
]}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass175-review.md').write_text('\n'.join([
    '# Dungar Varli forest and elder brother',
    '',
    'Two Dungar Varli survey responses have direct Turner comparisons. Their existing ingestion audit records manual page-image review and no reading uncertainty; selected audit objects are copied to `pass175-source-audit.json`. Complete Turner articles are in `varli-primary014.json`. Turner marks a possible earlier Munda source for the forest stem; this pass identifies the attested Indo-Aryan branch without resolving ultimate origin.',
    '',
    '| # | Varli | Proposed etymology | Evidence |',
    '|---|---|---|---|',
    '| 1 | **jhāṛi** ‘forest’ | **jhāṭa** §1, CDIAL **5362** | Oriya *jhāṛī* means thicket or low jungle. |',
    '| 2 | **dada** ‘elder brother’ | **\\*dādda**, CDIAL **6261** | Marathi *dādā* has the exact elder-brother sense. |',
    '',
    'Two `reflex` rows affect two records. Davar forest **jɔŋgɔḷ** retains a source transcription uncertainty, and Davar **moṭa bhās** requires separate compound analysis. No source ingestion or rebuild.',
    '',
    'Validation pending.',
]) + '\n')
print('accepted', len(accepted))
