"""Audit direct Shumashti horn and fingernail comparanda."""
import csv
import json
import pickle
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_z7r7npv2zgcp4', '12583', 'Turner 12583 explicitly lists Shumashti ṣīṅ “horn” under śŕ̥ṅga. Buddruss Shumashti ṣín is the same local horn stem, with the source’s final nasal notation preserved.'),
    ('f_kozn5nb4ouznw', '6914', 'Turner 6914 §1 explicitly lists Shumashti naũċikye-m “my fingernail” under nakhá. Buddruss independent naučik “fingernail” has the same unpossessed stem and the source č spelling is retained in Original, while the compiled sound profile writes c.'),
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
    matches = [x for x in raw if len(x) > 10 and x[0] == 'Shum' and x[7] == r['Source'] and unicodedata.normalize('NFC', x[2]) == unicodedata.normalize('NFC', r['Original'])]
    assert len(matches) == 1 and matches[0][3] == r['Gloss']
    accepted.append({'record': {k: r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')}, 'parent': parent, 'kind': 'reflex', 'citation': f"{r['Source']};CDIAL[{parent}]", 'evidence': evidence + ' Exact manually collated Buddruss row and persistent compiled identity audited; no source import change.'})
    audit.append({'id': fid, 'raw_source_row': matches[0], 'compiled_original': r['Original'], 'compiled_form': r['Form'], 'compiled_gloss': r['Gloss'], 'source': r['Source']})
wanted = {parent for _, parent, _ in targets}
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
assert 'Shum. ṣīṅ' in primary['12583']['text']
assert 'Shum. naũċikye-m' in primary['6914']['text']
(base / 'shumashti-primary005.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')
(base / 'pass164-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
(base / 'pass164-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': [
    {'form': 'baʦhār', 'gloss': 'leopard', 'reason': 'Turner 9341 cites Shumashti bac̣hā́r, but Buddruss explicitly marks the item a loanword; an immediate donor must be identified before a borrowed edge.'},
    {'form': 'xā̃ṭṭä', 'gloss': 'irrigation canal', 'reason': 'Turner 14345 gives similar Shumashti xā̃ṭa “rafter”, a false friend in the wrong sense.'},
]}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass164-review.md').write_text('\n'.join([
    '# Buddruss Shumashti: horn and fingernail',
    '',
    'Two direct Shumashti comparanda in full Turner articles. Exact manually collated Buddruss rows and compiled identities are in `pass164-source-audit.json`; Turner articles are in `shumashti-primary005.json`. The original Stanford ILL scan, collated at ingestion, is unavailable here for an independent image recheck.',
    '',
    '| # | Shumashti | Proposed etymology | Evidence |',
    '|---|---|---|---|',
    '| 1 | **ṣín** ‘horn’ | **śŕ̥ṅga**, CDIAL **12583** | Turner lists Shumashti *ṣīṅ* ‘horn’; final nasal notation differs. |',
    '| 2 | **naucik** ‘fingernail’ (source *naučik*) | **nakhá**, CDIAL **6914 §1** | Turner lists Shumashti *naũċikye-m* ‘my fingernail’; the source is the unpossessed stem. |',
    '',
    'Two `reflex` rows affect two records. The loan-marked **baʦhār** ‘leopard’ and canal **xā̃ṭṭä** remain held rather than asserting an inherited or false-sense edge. No source ingestion or rebuild.',
    '',
    'Validation pending.',
]) + '\n')
print('accepted', len(accepted))
