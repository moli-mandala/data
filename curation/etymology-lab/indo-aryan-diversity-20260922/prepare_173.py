"""Audit Varli earth and hill/mountain rows against full Turner subsections."""
import csv
import json
import pickle
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_ragltc7vpqmds', '9557', 'Turner 9557 explicitly lists Gujarati bhɔ̃/bhɔ̃y and Marathi bhuī̃/bhuī “earth” under bhū́mi. Dungar Varli bhũĩ matches this western earth word and source nasalization.'),
    ('f_lupp63ia2tafi', '5423-12', 'Turner 5423 §12 gives Gujarati ḍũgrī “hill” and Prakrit ḍuṁgara under *ḍuṅga-. Davar Varli ḍungri “hill” is the matching ḍu-n-g-r branch; the source writes Dungri in its own notation.'),
    ('f_lhy5zoucaela2', '5423-13', 'Turner 5423 §13 gives Marathi ḍõgar “hill” under *ḍōṅga-. Dungar Varli ḍoŋgɔr “hill” matches this rounded ḍō-ng-r regional form; original source notation DoMgOr is retained.'),
    ('f_idhj7jos54cnc', '5423-13', 'Turner 5423 §13 gives Marathi ḍõgar “hill” under *ḍōṅga-. A separate Dungar Varli survey key repeats ḍoŋgɔr with the broader “mountain” gloss and is linked independently.'),
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
    assert fid not in linked and r['Language_ID'] in {'Bhili', 'Varli'} and r['Source'].startswith('pattanaik-koul2003varli[')
    lect = 'Davar' if r['Language_ID'] == 'Bhili' else 'Dungar'
    item = int(r['Source'].split('item ')[1].split(',')[0])
    matches = [x for x in audit_rows if x['source'] == 'pattanaik-koul2003varli' and x['printed_page'] == 80 and x['item'] == item and x['lect'] == lect and x['gloss'] == r['Gloss'] and x['status'] == 'ingested']
    assert len(matches) == 1, (fid, matches)
    source_row = matches[0]
    assert source_row['raw'] == r['Original'] and source_row['review'] == 'manual-page-image' and all(not x['uncertainty'] for x in source_row['forms'])
    accepted.append({'record': {k: r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')}, 'parent': parent, 'kind': 'reflex', 'citation': f"{r['Source']};CDIAL[{parent}]", 'evidence': evidence + ' Existing manual page-image source audit has empty reading uncertainty; exact source and compiled identities checked. No import or transcription change.'})
    audit.append({'id': fid, 'manual_page_audit': source_row, 'compiled_original': r['Original'], 'compiled_form': r['Form'], 'compiled_gloss': r['Gloss'], 'compiled_tags': r['Tags']})
wanted = {'9557', '5423'}
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
assert 'M. bhuī˜ , bhuī' in primary['9557']['text']
assert 'G. ḍũgar m., ḍũgrī f.' in primary['5423']['text']
assert 'H. M. ḍõgar m.' in primary['5423']['text']
(base / 'varli-primary012.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')
(base / 'pass173-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
(base / 'pass173-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': [
    {'form': 'jɔmin', 'gloss': 'earth', 'reason': 'Distinct Persian/Urdu zamīn family; immediate donor not established.'},
    {'form': 'akās', 'gloss': 'sky', 'reason': 'Could be recent Sanskrit/Hindi learned borrowing; source transmission not established.'},
    {'form': 'dīs', 'gloss': 'sun', 'reason': 'Turner divasá primarily gives day/daylight senses; the Varli sun sense needs specific contextual support.'},
]}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass173-review.md').write_text('\n'.join([
    '# Davar and Dungar Varli: earth, hill and mountain',
    '',
    'Four source-defined records from the two Varli lects are compared with exact western Indo-Aryan Turner branches. The existing ingestion audit confirms manual page-image review with empty reading uncertainty for each selected response; the four audit objects are copied to `pass173-source-audit.json`. Complete Turner articles are in `varli-primary012.json`. Turner notes that the many hill forms may ultimately have a non-Aryan substrate origin; the selected links identify their attested Indo-Aryan branches, not a proven earlier substrate donor.',
    '',
    '| # | Varli survey | Proposed etymology | Evidence |',
    '|---|---|---|---|',
    '| 1 | **bhũĩ** ‘earth’ (Dungar) | **bhū́mi**, CDIAL **9557** | Turner lists Gujarati and Marathi nasalized *bhũĩ/bhuī* earth forms. |',
    '| 2 | **ḍungri** ‘hill’ (Davar) | **\\*ḍuṅga-**, CDIAL **5423-12** | Turner §12 gives Gujarati *ḍũgrī* ‘hill’, matching the source’s Dungri. |',
    '| 3–4 | **ḍoŋgɔr** ‘hill; mountain’ (Dungar, separate keys) | **\\*ḍōṅga-**, CDIAL **5423-13** | Turner §13 gives Marathi *ḍõgar*; both source senses are retained. |',
    '',
    'Four `reflex` rows affect four records. **jɔmin** ‘earth’, **akās** ‘sky’, and **dīs** ‘sun’ remain held over transmission or semantic scope. No source ingestion or rebuild.',
    '',
    'Validation pending.',
]) + '\n')
print('accepted', len(accepted))
