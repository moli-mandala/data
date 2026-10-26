"""Prepare three Khetrani numerals/celestial terms from the LSI vocabulary."""
import csv
import json
import pickle
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_pnrageaztqpz2', '5994', 'Turner 5994 lists Khetrani trē under tráyaḥ and Lahnda trāe under the same article’s *trāyaḥ subdivision. Grierson Khetrani trāē agrees with the latter local vowel shape; the broad article is secure, but its precise inherited subdivision is not asserted.'),
    ('f_sbmrwd2al6kec', '12278', 'Turner 12278 directly lists Khetrani sau “100” under śatá. Grierson gives exactly sau “Hundred”.'),
    ('f_qlwkweu6b2djc', '13574', 'Turner 13574 directly lists Lahnda-Khetrani suj “sun” under sū́ra, specifically the sū́rya subdivision. Grierson gives exactly suj “Sun”.'),
]
ids = {fid for fid, _, _ in targets}
records = {r['ID']: r for r in csv.DictReader(open('cldf/forms.csv')) if r['ID'] in ids}
assert len(records) == len(ids)
source_rows = list(csv.reader(open('data/other/forms/20260813-grierson-lsi.csv')))
linked = {r['Form_ID'] for r in read_assignments() if r['Status'] == 'accepted'}
linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank'] == '1')
accepted = []
audit = []
for fid, parent, evidence in targets:
    r = records[fid]
    assert fid not in linked and r['Language_ID'] == 'khet' and 'LSI-KHETRANI' in r['Tags']
    matches = [x for x in source_rows if len(x) > 10 and x[0] == 'LSI-KHETRANI' and x[7] == r['Source']]
    assert len(matches) == 1, (fid, len(matches))
    raw = matches[0]
    assert raw[3] == r['Gloss'] and unicodedata.normalize('NFC', raw[2]) == unicodedata.normalize('NFC', r['Original'])
    accepted.append({
        'record': {k: r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')},
        'parent': parent,
        'kind': 'reflex',
        'citation': f"{r['Source']};CDIAL[{parent}]",
        'evidence': evidence + ' Exact digital-edition key and original spelling checked; no import or transcription change.',
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
for n, phrase in {'5994': 'khet. trē', '12278': 'khet. sau', '13574': 'L.khet. suj'}.items():
    assert unicodedata.normalize('NFD', phrase) in unicodedata.normalize('NFD', primary[n]['text']), (n, phrase)
assert 'L. trāe' in primary['5994']['text']
(base / 'khetrani-primary003.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')
(base / 'pass122-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
held = [
    {'form': 'ḍḍāh', 'gloss': 'Ten', 'reason': 'Turner has Khetrani dā and nearby Lahnda ḍā̆h; the source double retroflex still needs a scan and dialect check.'},
    {'form': 'sunnā', 'gloss': 'Gold', 'reason': 'Turner Khetrani sunnā means “beautiful” under śōbhaná, a clear false friend for this source gloss.'},
    {'form': 'candr', 'gloss': 'Moon', 'reason': 'Turner attests it, but its immediate learned or borrowed status needs assessment.'},
]
(base / 'pass122-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': held}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass122-review.md').write_text('\n'.join([
    '# Grierson Khetrani: three, hundred, and sun',
    '',
    'Three source-backed reflex records in the digital LSI comparative vocabulary. Exact original-spelling rows and source locators were checked; complete Turner articles are in `khetrani-primary003.json`. Original physical table images have not yet been independently checked.',
    '',
    '| # | Khetrani | Proposed etymology | Evidence |',
    '|---|---|---|---|',
    '| 1 | **trāē** ‘three’ | **tráyaḥ**, CDIAL **5994** | Turner lists Khetrani *trē* and Lahnda *trāe* in subdivisions of the same article; the source vowel resembles the latter. No subdivision is asserted. |',
    '| 2 | **sau** ‘hundred’ | **śatá**, CDIAL **12278** | Turner cites Khetrani *sau* directly. |',
    '| 3 | **suj** ‘sun’ | **sū́ra**, CDIAL **13574** | Turner cites Khetrani *suj* directly under the *sū́rya* subdivision. |',
    '',
    'Three proposed `reflex` rows affect three Khetrani records. **ḍḍāh** ‘ten’, **sunnā** ‘gold’, and **candr** ‘moon’ remain held for the reasons in the decisions file. The source **sunnā** must not be conflated with Turner’s Khetrani *sunnā* ‘beautiful’. No source ingestion or rebuild.',
    '',
    'The Jambu homepage service was unavailable locally; exact compiled Turner comparanda led to full cached primary articles. Validation pending.',
]) + '\n')
print('accepted', len(accepted), 'held', len(held))
