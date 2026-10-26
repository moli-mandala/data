"""Audit two distinct Varli father stems against Turner nursery-word heads."""
import csv
import json
import pickle
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_n2amzzzgrznxu', '9209', 'Davar Varli bapus “father” follows Turner *bāppa CDIAL 9209 §1, with Bengali bāpu “father” and widespread bāp/bāpu. The final source -s is retained without a specific morphological explanation.'),
    ('f_acgtkrtsoavvo', '9198', 'Dungar Varli bā “father” matches Turner *bā CDIAL 9198 §1, with Pashai, Shumashti and Nepali bā “father”; it is kept separate from the bāp family.'),
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
    matches = [x for x in audit_rows if x['source'] == 'pattanaik-koul2003varli' and x['printed_page'] == 80 and x['item'] == 40 and x['lect'] == lect and x['gloss'] == 'father' and x['status'] == 'ingested']
    assert len(matches) == 1, (fid, matches)
    source_row = matches[0]
    assert source_row['review'] == 'manual-page-image' and all(not x['uncertainty'] for x in source_row['forms'])
    assert r['Original'] in [x['form'] for x in source_row['forms']]
    accepted.append({'record': {k: r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')}, 'parent': parent, 'kind': 'reflex', 'citation': f"{r['Source']};CDIAL[{parent}]", 'evidence': evidence + ' The installed Turner head is unsplit; section 1 is specified in the note. Existing manual page-image audit has empty reading uncertainty; exact response and compiled identity checked. No import or transcription change.'})
    audit.append({'id': fid, 'manual_page_audit': source_row, 'compiled_original': r['Original'], 'compiled_form': r['Form'], 'compiled_gloss': r['Gloss'], 'compiled_tags': r['Tags']})
wanted = {'9198', '9209'}
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
assert "Shum. bā 'father'" in primary['9198']['text'] and "N. bā 'father'" in primary['9198']['text']
assert "B. bāp 'father', bāpu 'father" in primary['9209']['text']
(base / 'varli-primary019.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')
(base / 'pass182-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
(base / 'pass182-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': []}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass182-review.md').write_text('\n'.join([
    '# Davar and Dungar Varli father terms',
    '',
    'The two lects use different father stems. They are linked to Turner’s separate nursery-word heads, with no claim that one derives from the other. The existing ingestion audit records manual page-image review and no reading uncertainty; selected audit objects are in `pass182-source-audit.json`, and complete Turner articles are in `varli-primary019.json`.',
    '',
    '| # | Varli survey | Proposed etymology | Evidence |',
    '|---|---|---|---|',
    '| 1 | **bapus** (Davar) ‘father’ | **\\*bāppa** §1, CDIAL **9209** | Bengali *bāpu* and regional *bāp* are explicit; source final -s is not analysed. |',
    '| 2 | **bā** (Dungar) ‘father’ | **\\*bā** §1, CDIAL **9198** | Pashai, Shumashti and Nepali *bā* are exact. |',
    '',
    'Two `reflex` rows affect two records. No source ingestion or rebuild.',
    '',
    'Validation pending.',
]) + '\n')
print('accepted', len(accepted))
