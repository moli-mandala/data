"""Prepare eight further direct Shumashti comparisons in Buddruss."""
import csv
import json
import pickle
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_4nvz6vchpvjwk', '540', 'Turner 540 explicitly lists Shumashti awata adjective/substantive “hungry” under abhukta, while noting a possible alternative ábhakta source. Buddruss avata matches the attested local form; this assignment does not settle Turner’s deeper alternative.'),
    ('f_je7qdxf5ov3ou', '3781', 'Turner 3781 explicitly lists Shumashti xāṭ “bedstead” under kháṭvā. Buddruss xāt “bed” retains its source plain final t notation.'),
    ('f_fb2hdgao6tfoe', '3906', 'Turner 3906 explicitly lists Shumashti xurem “my foot” under khura. Buddruss independent xur “foot” has the same nominal stem without the possessor suffix.'),
    ('f_rdblylhhysclw', '3906', 'Turner 3906 explicitly lists Shumashti xurigʸem “my heel” under khura. Buddruss xurík “heel” shares the same heel extension; the source’s own spelling is retained.'),
    ('f_myqhmih5emyvw', '9299', 'Turner 9299 explicitly lists Shumashti bṓstaṛ “lip” under *būsta. Buddruss bōstar matches the local lip stem with its own final consonant notation.'),
    ('f_cangpdxhosn36', '9560', 'Turner 9560 explicitly lists Shumashti bō̃zil “earthquake” under bhūmicala. Buddruss bõzil has the same nasalized earthquake form.'),
    ('f_q3sz4lzbh3qtm', '13929', 'Turner 13929 explicitly lists Shumashti λau “sleep” under *svāpra. Buddruss λau exactly matches.'),
    ('f_q5qwxxsblaodk', '14152', 'Turner 14152 explicitly lists Shumashti id-əm “my heart” under hŕ̥daya. Buddruss ida “heart” corresponds to its heart stem without the Turner possessor suffix.'),
]
ids = {fid for fid, _, _ in targets}
records = {r['ID']: r for r in csv.DictReader(open('cldf/forms.csv')) if r['ID'] in ids}
assert len(records) == len(ids)
source_rows = list(csv.reader(open('data/other/forms/20260819-buddruss-grangali.csv')))
linked = {r['Form_ID'] for r in read_assignments() if r['Status'] == 'accepted'}
linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank'] == '1')
accepted = []
audit = []
for fid, parent, evidence in targets:
    r = records[fid]
    assert fid not in linked and r['Language_ID'] == 'Shum' and r['Source'].startswith('buddruss-grangali1979[')
    matches = [x for x in source_rows if len(x) > 10 and x[0] == 'Shum' and x[7] == r['Source'] and unicodedata.normalize('NFC', x[2]) == unicodedata.normalize('NFC', r['Original'])]
    assert len(matches) == 1, (fid, len(matches))
    raw = matches[0]
    assert raw[3] == r['Gloss']
    accepted.append({
        'record': {k: r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')},
        'parent': parent,
        'kind': 'reflex',
        'citation': f"{r['Source']};CDIAL[{parent}]",
        'evidence': evidence + ' Exact manually collated source row and original spelling checked; no import or transcription change.',
    })
    audit.append({'id': fid, 'raw_source_row': raw, 'compiled_original': r['Original'], 'compiled_form': r['Form'], 'source': r['Source']})

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
for n, phrase in {'540': 'Shum. awata', '3781': 'Shum. xāṭ', '3906': 'Shum. xurem', '9299': 'Shum. bṓstaṛ', '9560': 'Shum. bō̃zil', '13929': 'Shum. λau', '14152': 'Shum. id-əm'}.items():
    assert unicodedata.normalize('NFD', phrase) in unicodedata.normalize('NFD', primary[n]['text']), (n, phrase)
(base / 'shumashti-primary003.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')
(base / 'pass149-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
held = [
    {'form': 'λe', 'gloss': 'give', 'reason': 'The exact Shumashti give-stem branch has not yet been established in Turner.'},
    {'form': 'lāk', 'gloss': 'grape', 'reason': 'Turner explicitly identifies a loan from another Indo-Aryan lect; immediate donor unresolved.'},
    {'form': 'ãṭhi', 'gloss': 'bone', 'reason': 'Turner explicitly identifies a Dardic loan set; immediate donor unresolved.'},
]
(base / 'pass149-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': held}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass149-review.md').write_text('\n'.join([
    '# Buddruss Shumashti: eight further basic terms',
    '',
    'Eight direct Shumashti citations or stem matches in Turner matched to the manually collated comparison rows of Buddruss’s Grangali survey. Exact original forms and source keys were checked against compiled identities; full Turner articles are in `shumashti-primary003.json`. The Stanford ILL scan was checked during ingestion but is unavailable here for recheck.',
    '',
    '| # | Shumashti | Proposed etymology | Evidence |',
    '|---|---|---|---|',
    '| 1 | **avata** ‘hungry’ | **abhukta**, CDIAL **540** | Turner cites Shumashti *awata* but notes a possible deeper *abhakta* alternative. |',
    '| 2 | **xāt** ‘bed’ | **kháṭvā**, CDIAL **3781** | Turner cites Shumashti *xāṭ* ‘bedstead’. |',
    '| 3–4 | **xur** ‘foot’, **xurík** ‘heel’ | **khura**, CDIAL **3906** | Turner cites Shumashti possessed *xurem* and *xurigʸem*. |',
    '| 5 | **bōstar** ‘lip’ | **\\*būsta**, CDIAL **9299** | Turner cites Shumashti *bṓstaṛ* directly. |',
    '| 6 | **bõzil** ‘earthquake’ | **bhūmicala**, CDIAL **9560** | Turner cites Shumashti *bō̃zil* directly. |',
    '| 7 | **λau** ‘sleep’ | **\\*svāpra**, CDIAL **13929** | Turner cites Shumashti *λau* directly. |',
    '| 8 | **ida** ‘heart’ | **hŕ̥daya**, CDIAL **14152** | Turner cites Shumashti possessed *id-əm*. |',
    '',
    'Eight proposed `reflex` rows affect eight Shumashti records. **λe** ‘give’ lacks a precise Turner branch; **lāk** ‘grape’ and **ãṭhi** ‘bone’ have flagged loan transmission and await immediate donor analysis. No source ingestion or rebuild.',
    '',
    'The Jambu homepage service was unavailable locally; exact compiled Turner comparanda led to full cached primary articles. Validation pending.',
]) + '\n')
print('accepted', len(accepted), 'held', len(held))
