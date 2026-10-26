"""Prepare direct Turner comparisons in Buddruss's Shumashti rows."""
import csv
import json
import pickle
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_dmcfarbxr6iys', '3818', 'Turner 3818 explicitly lists Shumashti xareṭá “donkey” as an extension of khara. Buddruss xareṭa has the same animal sense and consonants; source accentuation is retained.'),
    ('f_mrz2elkgrreao', '1921', 'Turner 1921 explicitly lists Shumashti wō “water” under udaká. Buddruss writes vō; the v/w notation difference is recorded, not silently normalized.'),
    ('f_xshmpu2e2ojz2', '3023', 'Turner 3023 explicitly lists Shumashti kō̃ “arrow” under kā́ṇḍa. Buddruss kõ represents the same nasalized arrow stem with its source vowel notation.'),
    ('f_q57cmwbz3be5y', '3440', 'Turner 3440 explicitly lists Shumashti λoā̃lī “mulberry” in the *kr̥mukalī branch. Buddruss gives the same form and tree sense.'),
    ('f_ffhwnhcq4gg76', '43', 'Turner 43 explicitly lists Shumashti aĩċ “eye” under ákṣi. Buddruss aĩʦ uses a different affricate symbol for the same local eye word.'),
    ('f_67zvzuegngrge', '2445', 'Turner 2445 explicitly lists Shumashti ī̃c̣ “bear” under ŕ̥kṣa. Buddruss ĩʦ̣ shares the nasalized vowel and affricate stem; Turner notes taboo remodeling in the wider animal-name family.'),
    ('f_6lsbevvzpk23y', '1872', 'Turner 1872 explicitly lists Shumashti učä́nik “spring of water” under *utsadhānī. Buddruss ucānik has the same consonant pattern and spring/source sense.'),
    ('f_gdxe744uan6qk', '2892', 'Turner 2892 explicitly lists Shumashti λā̆m “work” under kárman. Buddruss λăm has the same lateral onset and work sense.'),
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
    assert raw[3] == r['Gloss'] and unicodedata.normalize('NFC', raw[2]) == unicodedata.normalize('NFC', r['Original'])
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
for n, phrase in {'3818': 'Shum. xareṭá', '1921': 'Shum. wō', '3023': 'Shum. kō̃', '3440': 'Shum. λoā̃lī', '43': 'Shum. aĩċ', '2445': 'Shum. Gaw. ī˜c̣', '1872': 'Shum. učä', '2892': 'Shum. λā̆m'}.items():
    assert unicodedata.normalize('NFD', phrase) in unicodedata.normalize('NFD', primary[n]['text']), (n, phrase)
(base / 'shumashti-primary002.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')
(base / 'pass148-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
held = [
    {'form': 'ãṭhi', 'gloss': 'bone', 'reason': 'Turner 958 explicitly treats this Dardic set as loans from another Indo-Aryan lect; identify the immediate donor before a borrowed edge.'},
    {'form': 'lāk', 'gloss': 'grape', 'reason': 'Turner 6628 likewise marks Shumashti lāk as an Indo-Aryan loan; immediate donor unresolved.'},
    {'form': 'zū', 'gloss': 'girl', 'reason': 'Turner daughter comparison needs a semantic and louse-homophone check.'},
]
(base / 'pass148-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': held}, ensure_ascii=False, indent=2) + '\n')
(base / 'pass148-review.md').write_text('\n'.join([
    '# Buddruss Shumashti: eight direct lexical comparisons',
    '',
    'Eight direct Shumashti citations in Turner matched to the manually collated comparative rows of Buddruss’s Grangali survey. Exact original forms and source keys were checked against compiled identities; complete Turner articles are in `shumashti-primary002.json`. The Stanford ILL scan was checked during ingestion but is unavailable here for recheck.',
    '',
    '| # | Shumashti | Proposed etymology | Evidence |',
    '|---|---|---|---|',
    '| 1 | **xareṭa** ‘donkey’ | **khara¹**, CDIAL **3818** | Turner cites Shumashti *xareṭá* directly. |',
    '| 2 | **vō** ‘water’ | **udaká**, CDIAL **1921** | Turner cites Shumashti *wō*; source v/w notation retained. |',
    '| 3 | **kõ** ‘arrow’ | **kā́ṇḍa**, CDIAL **3023** | Turner cites Shumashti *kō̃* directly. |',
    '| 4 | **λoā̃lī** ‘mulberry’ | **kr̥muká** branch, CDIAL **3440** | Turner cites the Shumashti form directly in its *kr̥mukalī branch. |',
    '| 5 | **aĩʦ** ‘eye’ | **ákṣi**, CDIAL **43** | Turner cites Shumashti *aĩċ*; affricate transcription differs. |',
    '| 6 | **ĩʦ̣** ‘bear’ | **ŕ̥kṣa**, CDIAL **2445** | Turner cites Shumashti *ī̃c̣*; wider bear words show taboo remodeling. |',
    '| 7 | **ucānik** ‘spring/source’ | **\\*utsadhānī**, CDIAL **1872** | Turner cites Shumashti *učä́nik* directly. |',
    '| 8 | **λăm** ‘work’ | **kárman**, CDIAL **2892** | Turner cites Shumashti *λā̆m* directly. |',
    '',
    'Eight proposed `reflex` rows affect eight Shumashti records. **ãṭhi** ‘bone’ and **lāk** ‘grape’ are held because Turner explicitly flags loan transmission, and **zū** ‘girl’ awaits sense/homophone review. No source ingestion or rebuild.',
    '',
    'The Jambu homepage service was unavailable locally; exact compiled Turner comparanda led to full cached primary articles. Validation pending.',
]) + '\n')
print('accepted', len(accepted), 'held', len(held))
