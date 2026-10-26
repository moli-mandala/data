"""Audit three Darrai Pashai forms against western Dardic Turner comparanda."""
import csv
import json
import pickle
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_5lhcqerfqvxbe', '43', 'Turner 43 gives Pashai aᵉċ and neighboring Shumashti aĩċ “eye” under ákṣi. Lehr Darrai ainc preserves the same nasalized eye stem, with source n/c notation.'),
    ('f_vbibvkmj7pm6c', '9560', 'Turner 9560 gives neighboring Shumashti bō̃zil, Gawarbati banzíl and Waigali bōǰī́l “earthquake” under bhūmicala. Lehr Darrai bonjil has the diagnostic b-n-j-l sequence and exact sense.'),
    ('f_66ixtstmo7tho', '540', 'Turner 540 gives Pashai awatā́ and neighboring Shumashti awata “hungry” under abhukta, while explicitly allowing an ábhakta alternative. Lehr Darrai aoti has the same hungry stem; the deeper alternative remains unresolved.'),
]
ids = {fid for fid, _, _ in targets}
records = {r['ID']: r for r in csv.DictReader(open('cldf/forms.csv')) if r['ID'] in ids}
assert len(records) == len(ids)
raw = list(csv.reader(open('data/other/forms/20230705-pashai.csv')))
linked = {r['Form_ID'] for r in read_assignments() if r['Status'] == 'accepted'}
linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank'] == '1')
accepted, audit = [], []
for fid, parent, evidence in targets:
    r = records[fid]
    assert fid not in linked and r['Language_ID'] == 'Pas' and r['Source'] == 'lehr' and 'dialect:Pas:dar:' in r['Tags']
    matches = [x for x in raw if len(x) > 3 and x[0] == 'dar' and x[2:4] == [r['Form'], r['Gloss']]]
    assert len(matches) == 1, (fid, matches)
    accepted.append({'record': {k: r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')}, 'parent': parent, 'kind': 'reflex', 'citation': f'lehr;CDIAL[{parent}]', 'evidence': evidence + ' Exact raw Lehr row and persistent compiled identity checked; no import or transcription change.'})
    audit.append({'id': fid, 'raw_source_row': matches[0], 'compiled_original': r['Original'], 'compiled_form': r['Form'], 'compiled_gloss': r['Gloss'], 'compiled_tags': r['Tags']})
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
for n, phrase in {'43': 'Shum. aĩċ', '9560': 'Gaw. banzíl', '540': 'Paš. awatā́'}.items():
    assert phrase in primary[n]['text'], (n, phrase)
(base / 'pashai-primary024.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')
(base / 'pass169-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
(base / 'pass169-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': [
    {'form': 'aṇā', 'gloss': 'egg', 'reason': 'Turner 1111 identifies Pashai egg forms as Indic loans; the immediate donor is not established.'},
    {'form': 'ge', 'gloss': 'song', 'reason': 'Turner 4249 places a comparable form under gḗya but prefers gāthikā; the installed comparative edge follows 4249, so precise branch needs review.'},
]}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass169-review.md').write_text('\n'.join([
    '# Lehr Darrai Pashai: eye, earthquake and hungry',
    '',
    'Three raw Lehr Darrai rows match Pashai or adjacent western Dardic forms in full Turner articles. Exact raw and compiled records are audited in `pass169-source-audit.json`; full primary articles are in `pashai-primary024.json`. Lehr supplies no page locator.',
    '',
    '| # | Lehr Darrai | Proposed etymology | Evidence |',
    '|---|---|---|---|',
    '| 1 | **ainc** ‘eye’ | **ákṣi**, CDIAL **43** | Turner lists Pashai *aᵉċ* and nasalized Shumashti *aĩċ*. |',
    '| 2 | **bonjil** ‘earthquake’ | **bhūmicala**, CDIAL **9560** | Neighboring Shumashti *bō̃zil* and Gawarbati *banzíl* match the distinctive stem. |',
    '| 3 | **aoti** ‘hungry’ | **abhukta**, CDIAL **540** | Turner lists Pashai *awatā́* and Shumashti *awata*, with *ábhakta* also possible. |',
    '',
    'Three `reflex` rows affect three records. **aṇā** ‘egg’ is held for its immediate loan donor; **ge** ‘song’ is held for a competing Turner branch. No source ingestion or rebuild.',
    '',
    'Validation pending.',
]) + '\n')
print('accepted', len(accepted))
