"""Audit three Khetrani adjectives in the LSI digital survey."""
import csv
import json
import pickle
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_bfz3rfcztpdf4', '4564', 'Turner 4564 gives Lahnda and Punjabi caṅgā “good” under caṅga. Grierson’s Khetrani survey glosses the elicitation “Good man” but prints only the adjective caŋgā, matching this regional form.'),
    ('f_4j34vs3ts53na', '11225', 'Turner 11225 gives Lahnda vaḍḍā and western Punjabi vaḍ̠ā “big” under vaḍra, with a “high” use in a neighboring Romani branch. Grierson Khetrani vaḍā is elicited as “High”; this records a local big/high sense without asserting a separate height lexeme.'),
    ('f_oqxnalxoa3hkq', '1634', 'Turner 1634 gives Lahnda uccā “high” and related upper/above uses under ucca. Grierson Khetrani ucā, glossed “Up”, fits the same local upward/high adjective or adverb.'),
]
ids = {fid for fid, _, _ in targets}
records = {r['ID']: r for r in csv.DictReader(open('cldf/forms.csv')) if r['ID'] in ids}
assert len(records) == len(ids)
raw = list(csv.reader(open('data/other/forms/20260813-grierson-lsi.csv')))
linked = {r['Form_ID'] for r in read_assignments() if r['Status'] == 'accepted'}
linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank'] == '1')
accepted, audit = [], []
for fid, parent, evidence in targets:
    r = records[fid]
    assert fid not in linked and r['Language_ID'] == 'khet' and 'LSI-KHETRANI' in r['Tags']
    matches = [x for x in raw if len(x) > 10 and x[0] == 'LSI-KHETRANI' and x[7] == r['Source']]
    assert len(matches) == 1
    source_row = matches[0]
    assert source_row[3] == r['Gloss'] and unicodedata.normalize('NFC', source_row[2]) == unicodedata.normalize('NFC', r['Original'])
    accepted.append({'record': {k: r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')}, 'parent': parent, 'kind': 'reflex', 'citation': f"{r['Source']};CDIAL[{parent}]", 'evidence': evidence + ' Exact LSI digital-edition key and original spelling checked; no import or transcription change.'})
    audit.append({'id': fid, 'raw_source_row': source_row, 'compiled_original': r['Original'], 'compiled_form': r['Form'], 'source': r['Source']})
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
for n, phrase in {'4564': 'L. P. caṅgā', '11225': '(Ju.) vaḍ̠ā', '1634': 'L. uccā'}.items():
    assert unicodedata.normalize('NFD', phrase) in unicodedata.normalize('NFD', primary[n]['text']), (n, phrase)
(base / 'khetrani-primary011.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')
(base / 'pass168-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
(base / 'pass168-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': [
    {'form': 'gandā', 'gloss': 'Bad boy', 'reason': 'The survey prints an adjective, but a secure precise parent for this local bad/dirty sense remains undetermined.'},
    {'form': 'zimī', 'gloss': 'Down', 'reason': 'Potential locative/ground term; avoid a direct opposing derivation from “up” without a lexical source.'},
]}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass168-review.md').write_text('\n'.join([
    '# Grierson Khetrani: good, high and up',
    '',
    'Three adjective/adverb forms in the LSI digital edition are compared with western Indo-Aryan forms in full Turner articles. Exact source keys and original spellings are in `pass168-source-audit.json`, and primary articles in `khetrani-primary011.json`. Original physical table images have not been independently rechecked.',
    '',
    '| # | Khetrani | Proposed etymology | Evidence |',
    '|---|---|---|---|',
    '| 1 | **caŋgā** ‘good man’ | **caṅga**, CDIAL **4564** | Turner lists Lahnda/Punjabi *caṅgā* ‘good’; the source prints only the adjective. |',
    '| 2 | **vaḍā** ‘high’ | **vaḍra**, CDIAL **11225** | Lahnda/Punjabi *vaḍḍā/vaḍ̠ā* ‘big’ matches the form; “high” is the elicited local sense. |',
    '| 3 | **ucā** ‘up’ | **ucca**, CDIAL **1634** | Turner lists Lahnda *uccā* ‘high’ and wider upward uses. |',
    '',
    'Three `reflex` rows affect three records. **gandā** ‘bad boy’ and **zimī** ‘down’ remain held. No source ingestion or rebuild.',
    '',
    'Validation pending.',
]) + '\n')
print('accepted', len(accepted))
