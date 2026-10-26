"""Audit Khetrani one and ten with explicit regional Turner subsections."""
import csv
import json
import pickle
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_p6yauohl4ozp4', '2462-2', 'Turner 2462 §2 gives Sindhi heko “unique”, Lahnda hikk and Punjabi hik- “single” under *ēkka-, the reinforced branch of ḗka. Grierson Khetrani hik “One” fits the regional h-k form; the chosen second subsection distinguishes it from unsuffixed ḗka.'),
    ('f_zq5z27aokmjay', '6227', 'Turner 6227 gives Sindhi ḍ̠aha and Lahnda ḍ̠ā̆h “ten” under dáśa; the LSI digital Khetrani row prints ḍḍāh, with doubled initial retroflex. Turner elsewhere lists Khetrani dā, so the printed spelling is anomalous, but the numeral sense and regional ḍ-h shape support the family. No transcription emendation is made; physical scan recheck remains pending.'),
]
ids = {fid for fid, _, _ in targets}
records = {r['ID']: r for r in csv.DictReader(open('cldf/forms.csv')) if r['ID'] in ids}
assert len(records) == len(ids)
source_rows = list(csv.reader(open('data/other/forms/20260813-grierson-lsi.csv')))
linked = {r['Form_ID'] for r in read_assignments() if r['Status'] == 'accepted'}
linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank'] == '1')
accepted, audit = [], []
for fid, parent, evidence in targets:
    r = records[fid]
    assert fid not in linked and r['Language_ID'] == 'khet' and 'LSI-KHETRANI' in r['Tags']
    matches = [x for x in source_rows if len(x) > 10 and x[0] == 'LSI-KHETRANI' and x[7] == r['Source']]
    assert len(matches) == 1
    raw = matches[0]
    assert raw[3] == r['Gloss'] and unicodedata.normalize('NFC', raw[2]) == unicodedata.normalize('NFC', r['Original'])
    accepted.append({'record': {k: r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')}, 'parent': parent, 'kind': 'reflex', 'citation': f"{r['Source']};CDIAL[{parent}]", 'evidence': evidence + ' Exact LSI digital-edition key and original spelling checked; no import or transcription change.'})
    audit.append({'id': fid, 'raw_source_row': raw, 'compiled_original': r['Original'], 'compiled_form': r['Form'], 'source': r['Source']})
wanted = {'2462', '6227'}
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
for n, phrase in {'2462': 'L. hekk , hikk', '6227': 'khet. dā'}.items():
    assert unicodedata.normalize('NFD', phrase) in unicodedata.normalize('NFD', primary[n]['text']), (n, phrase)
(base / 'khetrani-primary010.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')
(base / 'pass163-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
(base / 'pass163-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': [
    {'form': 'cā', 'gloss': 'What?', 'reason': 'Turner kím has neighboring ki/kya, but the c-initial Khetrani form needs its own pronoun-branch evidence.'},
]}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass163-review.md').write_text('\n'.join([
    '# Grierson Khetrani: one and ten',
    '',
    'Two numeral records in the LSI digital edition are compared with precise western Indo-Aryan forms in Turner. Exact original-spelling rows and keys are in `pass163-source-audit.json`; full articles are in `khetrani-primary010.json`. The physical LSI table image has not been independently checked. The unusual **ḍḍāh** spelling is preserved without emendation and its difference from Turner’s Khetrani *dā* is documented.',
    '',
    '| # | Khetrani | Proposed etymology | Evidence |',
    '|---|---|---|---|',
    '| 1 | **hik** ‘one’ | **\\*ēkka-**, CDIAL **2462-2** | Turner lists Sindhi *heko*, Lahnda *hikk* and Punjabi *hik-* in this second branch. |',
    '| 2 | **ḍḍāh** ‘ten’ | **dáśa**, CDIAL **6227** | Sindhi *ḍ̠aha* and Lahnda *ḍ̠ā̆h* match the regional numeral; doubled onset awaits image verification. |',
    '',
    'Two `reflex` rows affect two records. **cā** ‘what’ remains held, and the exact transcription of **ḍḍāh** remains pending a physical scan check. No source ingestion or rebuild.',
    '',
    'Validation pending.',
]) + '\n')
print('accepted', len(accepted))
