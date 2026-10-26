"""Prepare direct Turner matches in Lehr's Darrai Pashai vocabulary."""
import csv
import json
import pickle
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_whdmafvplrm3k', '4286', 'Turner 4286 explicitly lists Pashai gō “lizard” under gōdhā́. Lehr Darrai gō matches form and exact sense.'),
    ('f_utsjjwk5azluu', '10412', 'Turner 10412 explicitly lists Darrai Pashai yan “mill” under yantrá. The raw Lehr row prints “yan, źan”; this record is the first compiled source variant.'),
    ('f_hmrnu25zq5yae', '10412', 'Turner 10412 lists Darrai yan and Lauṛ Pashai žantr for “mill” under yantrá. Raw Lehr “yan, źan” supplies this second palatal variant, which shares that mill etymon.'),
    ('f_v3kvzzklocb6o', '7603', 'Turner 7603 explicitly lists Pashai ne “not” under naiva. Lehr gives the same ne as a grammatical NEG marker; the compiled Gloss is blank but the grammatical Tags field retains neg.'),
    ('f_gfzgt72qvuti4', '2892', 'Turner 2892 explicitly lists Lauṛ Pashai lām and other regional λām “work” under kárman. Lehr Darrai Lām has the same work sense and lateral stem; its source capital L notation is retained.'),
]
ids = {fid for fid, _, _ in targets}
records = {r['ID']: r for r in csv.DictReader(open('cldf/forms.csv')) if r['ID'] in ids}
assert len(records) == len(ids)
raw = list(csv.reader(open('data/other/forms/20230705-pashai.csv')))
linked = {r['Form_ID'] for r in read_assignments() if r['Status'] == 'accepted'}
linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank'] == '1')
accepted = []
audit = []
for fid, parent, evidence in targets:
    r = records[fid]
    assert fid not in linked and r['Source'] == 'lehr' and r['Language_ID'] == 'Pas' and 'dialect:Pas:dar:' in r['Tags']
    if r['Form'] in {'yan', 'źan'}:
        matches = [x for x in raw if len(x) > 3 and x[0] == 'dar' and x[2] == 'yan, źan' and x[3] == 'mill']
    elif r['Form'] == 'ne':
        matches = [x for x in raw if len(x) > 3 and x[0] == 'dar' and x[2] == 'ne' and x[3] == 'NEG']
        assert r['Gloss'] == '' and 'neg' in r['Tags'].split()
    else:
        matches = [x for x in raw if len(x) > 3 and x[0] == 'dar' and x[2:4] == [r['Form'], r['Gloss']]]
    assert len(matches) == 1, (fid, len(matches))
    accepted.append({
        'record': {k: r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')},
        'parent': parent,
        'kind': 'reflex',
        'citation': f'lehr;CDIAL[{parent}]',
        'evidence': evidence + ' Exact raw Lehr row and persistent compiled identity checked; no import or transcription change.',
    })
    audit.append({'id': fid, 'raw_source_row': matches[0], 'compiled_original': r['Original'], 'compiled_form': r['Form'], 'compiled_gloss': r['Gloss'], 'compiled_tags': r['Tags']})

wanted = {parent for _, parent, _ in targets}
primary = {}
for page_num, html in enumerate(pickle.load(open('data/cdial/cdial.pickle', 'rb')), 1):
    if not any(f'<number>{n}</number>' in html for n in wanted - primary.keys()):
        continue
    soup = BeautifulSoup(html, 'html.parser')
    for paragraph in soup.select('p'):
        number = paragraph.find('number')
        if number and number.get_text(strip=True) in wanted:
            n = number.get_text(strip=True)
            primary[n] = {'page': page_num, 'url': f'https://dsal.uchicago.edu/cgi-bin/app/soas_query.py?page={page_num}', 'html': str(paragraph), 'text': paragraph.get_text(' ', strip=True)}
    if len(primary) == len(wanted):
        break
assert len(primary) == len(wanted)
for n, phrase in {'4286': 'Paš. gō', '10412': 'dar. yan', '7603': 'Paš. ne', '2892': 'Paš.lauṛ. lām'}.items():
    assert phrase in primary[n]['text'], (n, phrase)
(base / 'pashai-primary022.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')
(base / 'pass143-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
held = [
    {'form': 'dār', 'gloss': 'mountain', 'reason': 'The same string also denotes “hold” in Turner; this mountain token needs its own etymon.'},
    {'form': 'kōṭa', 'gloss': 'room', 'reason': 'Turner 3243 has a knee homonym; the room stem needs independent evidence.'},
    {'form': 'sāt', 'gloss': 'watch', 'reason': 'Do not conflate with the numeral seven.'},
]
(base / 'pass143-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': held}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass143-review.md').write_text('\n'.join([
    '# Lehr Darrai Pashai: lizard, mill, not, and work',
    '',
    'Five direct or locally supported Turner matches in Lehr’s Darrai vocabulary. Exact raw rows and persistent compiled identities were checked; the raw “yan, źan” row yields two records, and raw NEG becomes a blank-gloss grammatical form tagged `neg`. Full Turner articles are in `pashai-primary022.json`. The Lehr source gives no page locator, so none is invented.',
    '',
    '| # | Lehr Darrai | Proposed etymology | Evidence |',
    '|---|---|---|---|',
    '| 1 | **gō** ‘lizard’ | **gōdhā́**, CDIAL **4286** | Turner cites Pashai *gō* ‘lizard’ directly. |',
    '| 2–3 | **yan**, **źan** ‘mill’ | **yantrá**, CDIAL **10412** | Turner cites Darrai *yan* and regional Pashai *žantr*; both occur in one Lehr raw variant row. |',
    '| 4 | **ne** grammatical NEG | **naiva**, CDIAL **7603** | Turner cites Pashai *ne* ‘not’; the compiled source tag carries the grammatical gloss. |',
    '| 5 | **Lām** ‘work’ | **kárman**, CDIAL **2892** | Turner cites regional Pashai *lām/λām* ‘work’. |',
    '',
    'Five proposed `reflex` rows affect five Pashai records. **dār** ‘mountain’, **kōṭa** ‘room’, and **sāt** ‘watch’ remain held rather than attaching homophonous unrelated Turner entries. No source ingestion or rebuild.',
    '',
    'The Jambu homepage service was unavailable locally; exact compiled CDIAL comparanda led to full cached primary articles. Validation pending.',
]) + '\n')
print('accepted', len(accepted), 'held', len(held))
