"""Audit four Khetrani LSI survey records against regional Turner branches."""
import csv
import json
import pickle
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_uueiwadrl3vao', '4147-2', 'Turner 4147 §2 assigns regional Punjabi gāu and gā̃ “cow” to feminine *gāvā-. Grierson Khetrani gāũ “Cow” matches this nasalized feminine branch; the first survey slot has its own persistent identity.'),
    ('f_jwggdisr2ph7y', '4147-2', 'Turner 4147 §2 gives Punjabi gāu and gā̃ “cow” from *gāvā-. This separate Khetrani survey slot repeats the same gāũ “Cow” and is linked independently.'),
    ('f_ejr6nxb4yvo3s', '4661', 'Turner 4661 §2 directly cites Khetrani candr “moon” under candrá. Jambu has one unsplit 4661 node; the moon subsection is specified in this evidence.'),
    ('f_eribiqb77tfvq', '6495', 'Turner 6495 gives widespread Indo-Aryan dūr “far”, including neighboring Awan Lahnda dūr. Khetrani dūīr has the same distant/far sense; the additional vowel is retained as printed in the LSI edition.'),
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
wanted = {'4147', '4661', '6495'}
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
for n, phrase in {'4147': 'P. gāu , gā̃ f.', '4661': 'khet. candr m.', '6495': 'awāṇ. dūr adv.'}.items():
    assert unicodedata.normalize('NFD', phrase) in unicodedata.normalize('NFD', primary[n]['text']), (n, phrase)
(base / 'khetrani-primary009.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')
(base / 'pass158-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
(base / 'pass158-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': [
    {'form': 'sunnā', 'gloss': 'Gold', 'reason': 'Turner has a homophonous Khetrani “beautiful” item; a metal etymon needs independent evidence.'},
    {'form': 'kukkur', 'gloss': 'Cock', 'reason': 'Despite a dog-word lookalike, the source sense is “Cock”; source reading and etymology remain unresolved.'},
    {'form': 'kuttē', 'gloss': 'Dogs', 'reason': 'Plural morphology has not yet been verified against the survey’s treatment.'},
]}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass158-review.md').write_text('\n'.join([
    '# Grierson Khetrani: cow, moon and far',
    '',
    'Four LSI digital-edition rows are linked to three Turner entries. Exact original-spelling rows and source locators are in `pass158-source-audit.json`; complete Turner articles are in `khetrani-primary009.json`. Original physical table images have not been independently rechecked.',
    '',
    '| # | Khetrani | Proposed etymology | Evidence |',
    '|---|---|---|---|',
    '| 1–2 | **gāũ** ‘cow’ (two survey slots) | **\\*gāvā-**, CDIAL **4147-2** | Turner’s feminine branch contains Punjabi *gāu/gā̃* ‘cow’; the two LSI keys are distinct. |',
    '| 3 | **candr** ‘moon’ | **candrá**, CDIAL **4661 §2** | Turner cites Khetrani *candr* ‘moon’ directly; the installed parent node is unsplit. |',
    '| 4 | **dūīr** ‘far’ | **dūrá**, CDIAL **6495** | The widespread *dūr* sense includes Awan Lahnda; LSI retains an additional vowel. |',
    '',
    'Four `reflex` rows affect four records. **sunnā** ‘gold’, **kukkur** ‘cock’, and plural **kuttē** remain held. No source ingestion or rebuild.',
    '',
    'Validation pending.',
]) + '\n')
print('accepted', len(accepted))
