"""Audit Grangali basic vocabulary against Western Dardic Turner comparanda."""
import csv
import json
import pickle
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_3luiiysngovcs', '8265', 'Turner 8265 lists neighboring Shumashti puλ and Bashkarik puλ “son” under putrá. Buddruss Grangali puλ matches exact form and sense.'),
    ('f_yljlv7c2ftbbo', '3781', 'Turner 3781 lists Shumashti xāṭ and nearby Pashai xōṭ “bedstead” under kháṭvā. Buddruss Grangali xᵒat “bed” has the same dorsal onset and t-final bed stem; its labial superscript notation is retained.'),
    ('f_35cnsjteur27q', '3906', 'Turner 3906 §1 lists neighboring Shumashti xur-em “my foot” and Pashai xūr “foot” under khura. Buddruss Grangali xur “foot” is the unpossessed matching stem.'),
    ('f_4sdtgwurjqb7w', '3906', 'Turner 3906 §1 lists Shumashti xurigʸ-em “my heel” under khura. Buddruss Grangali xurík is “heel” in Buddruss and “ankle” in the cited Grjunberg comparison; both source senses are preserved.'),
    ('f_5uzpafwpzq7y4', '14152', 'Turner 14152 lists neighboring Shumashti id-əm “my heart” and Ningalami idə́ under hŕ̥daya. Buddruss Grangali yidə́ “heart” matches the local stem with initial glide.'),
    ('f_z5uduiibqiqs2', '2892', 'Turner 2892 lists neighboring Shumashti λăm and Gawarbati λam “work” under kárman. Buddruss Grangali λăm is an exact local cognate.'),
    ('f_hkn4qe52iblus', '9560', 'Turner 9560 lists Shumashti bō̃zil and Gawarbati banzíl “earthquake” under bhūmicala. Buddruss Grangali bũzíl has the diagnostic nasalized stem and same sense.'),
]
ids = {fid for fid, _, _ in targets}
records = {r['ID']: r for r in csv.DictReader(open('cldf/forms.csv')) if r['ID'] in ids}
assert len(records) == len(ids)
source_rows = list(csv.reader(open('data/other/forms/20260819-buddruss-grangali.csv')))
linked = {r['Form_ID'] for r in read_assignments() if r['Status'] == 'accepted'}
linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank'] == '1')
accepted, audit = [], []
for fid, parent, evidence in targets:
    r = records[fid]
    assert fid not in linked and r['Language_ID'] == 'Gng' and r['Source'].startswith('buddruss-grangali1979[')
    matches = [x for x in source_rows if len(x) > 10 and x[0] == 'Gng' and x[7] == r['Source'] and unicodedata.normalize('NFC', x[2]) == unicodedata.normalize('NFC', r['Original'])]
    assert len(matches) == 1, (fid, matches)
    raw = matches[0]
    assert raw[3] == r['Gloss'] and unicodedata.normalize('NFC', raw[2]) == unicodedata.normalize('NFC', r['Original'])
    accepted.append({'record': {k: r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')}, 'parent': parent, 'kind': 'reflex', 'citation': f"{r['Source']};CDIAL[{parent}]", 'evidence': evidence + ' Exact manually collated Buddruss source row and persistent compiled identity checked; no import or transcription change.'})
    audit.append({'id': fid, 'raw_source_row': raw, 'compiled_original': r['Original'], 'compiled_form': r['Form'], 'source': r['Source']})
variant_id = 'f_5larkwqg4yn62'
assert any(r['Child_ID'] == variant_id and r['Parent_ID'] == 'f_hkn4qe52iblus' and r['Kind'] == 'variant' and r['Rank'] == '1' for r in csv.DictReader(open('cldf/edges.csv')))
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
for n, phrase in {'8265': 'Shum. puλ', '3781': 'Shum. xāṭ', '3906': 'Shum. xurem', '14152': 'Niṅg. idə́', '2892': 'Shum. λā̆m', '9560': 'Gaw. banzíl'}.items():
    assert unicodedata.normalize('NFD', phrase) in unicodedata.normalize('NFD', primary[n]['text']), (n, phrase)
(base / 'grangali-primary018.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')
(base / 'pass159-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
(base / 'pass159-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': [
    {'form': 'λõ', 'gloss': 'sleep', 'reason': 'Turner 13929 cites neighboring Shumashti λau, but Grangali nasal-vowel correspondence remains to be checked.'},
    {'form': 'bistṓ', 'gloss': 'lip', 'reason': 'Neighboring lip stems have different vowels and suffixes; precise branch not yet fixed.'},
    {'form': 'dār', 'gloss': 'mountain', 'reason': 'Potential Iranian/Indic homonymy; do not attach an unrelated “hold” verb.'},
]}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass159-review.md').write_text('\n'.join([
    '# Buddruss Grangali: seven basic terms',
    '',
    'Seven unlinked Grangali forms are compared with Shumashti, Ningalami and other western Dardic forms in full Turner articles. Exact manually collated source rows and compiled identities are in `pass159-source-audit.json`; primary articles are in `grangali-primary018.json`. The original Stanford ILL scan, collated during ingestion, is unavailable here for independent image recheck. Buddruss’s second earthquake form *buĩzil* is already linked by a compiled rank-1 variant edge to *bũzíl* and is not duplicated.',
    '',
    '| # | Grangali | Proposed etymology | Evidence |',
    '|---|---|---|---|',
    '| 1 | **puλ** ‘son’ | **putrá**, CDIAL **8265** | Turner lists exact Shumashti and Bashkarik *puλ* ‘son’. |',
    '| 2 | **xᵒat** ‘bed’ | **kháṭvā**, CDIAL **3781** | Neighboring Shumashti *xāṭ* and Pashai *xōṭ* denote a bedstead. |',
    '| 3 | **xur** ‘foot’ | **khura**, CDIAL **3906 §1** | Shumashti *xur-em* ‘my foot’ has this same unpossessed stem. |',
    '| 4 | **xurík** ‘heel; ankle’ | **khura**, CDIAL **3906 §1** | Shumashti *xurigʸ-em* ‘my heel’; Buddruss and Grjunberg senses both retained. |',
    '| 5 | **yidə́** ‘heart’ | **hŕ̥daya**, CDIAL **14152** | Ningalami *idə́* and Shumashti *id-əm* match the local heart stem. |',
    '| 6 | **λăm** ‘work’ | **kárman**, CDIAL **2892** | Turner lists exact Shumashti *λăm* and Gawarbati *λam*. |',
    '| 7 | **bũzíl** ‘earthquake’ | **bhūmicala**, CDIAL **9560** | Shumashti *bō̃zil* and Gawarbati *banzíl* match the nasalized stem and sense. |',
    '',
    'Seven `reflex` rows affect seven records. **λõ** ‘sleep’, **bistṓ** ‘lip’, and **dār** ‘mountain’ remain held. No source ingestion or rebuild.',
    '',
    'Validation pending.',
]) + '\n')
print('accepted', len(accepted))
