"""Audit Lehr Darrai forms against their raw rows and full Turner articles."""
import csv
import json
import pickle
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

base = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
targets = [
    ('f_u3hansoijefpu', '13551', 'Turner 13551 §2 explicitly cites Darrai and Wegali Pashai sūnčék “needle” under *sūñcī-. Lehr suncek is the corresponding sewing-needle form; no claim of an Iranian borrowing is made for this branch.'),
    ('f_ql3zzdes6jsww', '6283', 'Turner 6283 §1 explicitly cites Darrai Pashai damaṭā́ “rope” as an extension of dā́man. Lehr ḍamatā has the same stem and sense despite the differing initial consonant notation.'),
    ('f_gtf5hol4zvzua', '12415', 'Turner 12415 explicitly cites Darrai Pashai šāl under śāli “growing or unhusked rice”; Lehr śāl is glossed “unharvested rice”.'),
    ('f_a3iglrv4tt6x2', '2655', 'Turner 2655 explicitly cites Darrai Pashai karā́ṛ “wolf” under káḍāra/karāla; the source form and animal sense coincide, although the wider article groups semantically diverse reflexes.'),
    ('f_6u5arb5bopyoa', '9746', 'Turner 9746 §5 explicitly cites Darrai Pashai maṭḗx “frog” under the *maṇṭrakka- or *maṭrakka- subgroup, acknowledging expressive/taboo reshaping of frog names.'),
    ('f_3o4anslcy4ovk', '8233', 'Turner 8233 §1 explicitly cites Darrai Pashai pḗla “yellow” under pītala; Lehr pēlā has the same adjective and sense.'),
    ('f_ykatrts42ejly', '13743', 'Turner 13743 explicitly cites Darrai Pashai tar- “look, see” under *stharati; Lehr tar- is “look at”.'),
    ('f_sqvpvxd5pkfug', '1020b', 'Turner’s printed addendum 14264 explicitly cites Darrai Pashai aṣpā́ṛ “reaped rice” under *ākṣapāṭa-. The installed canonical parent is 1020b (14264 redirects there). Lehr aśpāṛ is broadly glossed “rice”; link rests on the diagnostic form, with the narrower Turner sense retained in this note.'),
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
    assert fid not in linked and r['Source'] == 'lehr' and r['Language_ID'] == 'Pas' and 'dialect:Pas:dar:' in r['Tags']
    matches = [x for x in raw if len(x) > 3 and x[0] == 'dar' and x[2:4] == [r['Form'], r['Gloss']]]
    assert len(matches) == 1, (fid, matches)
    accepted.append({'record': {k: r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')}, 'parent': parent, 'kind': 'reflex', 'citation': f'lehr;CDIAL[{parent}]', 'evidence': evidence + ' Exact raw Lehr row and persistent compiled identity checked; no import change.'})
    audit.append({'id': fid, 'raw_source_row': matches[0], 'compiled_original': r['Original'], 'compiled_form': r['Form'], 'compiled_gloss': r['Gloss'], 'compiled_tags': r['Tags']})
wanted = {parent for _, parent, _ in targets} | {'14264'}
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
for n, phrase in {'13551': 'Paš.dar. weg.', '6283': 'Paš.dar. damaṭā́', '12415': 'Paš.lauṛ. dar. šāl', '2655': 'dar. karā́ṛ', '9746': 'dar. maṭḗx', '8233': 'dar. pḗla', '13743': 'dar. tar', '14264': 'dar. aṣpā́ṛ', '1020b': 'Add. 14264'}.items():
    assert phrase in primary[n]['text'], (n, phrase)
(base / 'pashai-primary023.json').write_text(json.dumps(primary, ensure_ascii=False, indent=2) + '\n')
(base / 'pass153-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
held = [
    {'form': 'peṛek', 'gloss': 'palm-F', 'reason': 'Turner 8358 gives Darrai péṛek but explicitly calls the proposed connection with pr̥thú “wide” doubtful.'},
    {'form': 'uncaṛek', 'gloss': 'wellspring', 'reason': 'Turner 1872 compares Wegali unčaṛék only tentatively, marked with a question; Darrai assignment deferred.'},
]
(base / 'pass153-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': held}, ensure_ascii=False, indent=2) + '\n')
rows = [
    ('suncek', 'sewing needle', '*sūñcī-*, CDIAL 13551 §2', 'Turner lists Darrai and Wegali Pashai *sūnčék* “needle”.'),
    ('ḍamatā', 'rope', '*dā́man*, CDIAL 6283 §1', 'Turner lists Darrai *damaṭā́* “rope” as an extension; initial consonant notation differs.'),
    ('śāl', 'unharvested rice', '*śāli*, CDIAL 12415', 'Turner lists Darrai *šāl* “growing or unhusked rice”.'),
    ('karāṛ', 'wolf', '*káḍāra / karāla*, CDIAL 2655', 'Turner lists Darrai *karā́ṛ* “wolf” directly.'),
    ('maṭēx', 'frog', '*maṇṭrakka- / *maṭrakka-*, CDIAL 9746 §5', 'Turner lists Darrai *maṭḗx* “frog”; the wider group is reshaped by taboo and expression.'),
    ('pēlā', 'yellow', '*pītala*, CDIAL 8233 §1', 'Turner lists Darrai *pḗla* “yellow”.'),
    ('tar-', 'look at', '*stharati*, CDIAL 13743', 'Turner lists Darrai *tar-* “look, see”.'),
    ('aśpāṛ', 'rice', '*ākṣapāṭa-*, CDIAL 1020b (Add. 14264)', 'The printed addendum lists Darrai *aṣpā́ṛ* “reaped rice”; Lehr gives broader “rice”.'),
]
lines = ['# Lehr Darrai Pashai: eight direct Turner comparanda', '', 'Eight previously unlinked Darrai Pashai records have direct comparanda in full Turner articles. Exact Lehr raw rows and compiled identities are audited in `pass153-source-audit.json`; full primary articles are in `pashai-primary023.json`. The source has no page locator.', '', '| # | Lehr Darrai | Proposed etymology | Evidence |', '|---|---|---|---|']
for i, (form, gloss, etym, why) in enumerate(rows, 1):
    lines.append(f'| {i} | **{form}** ‘{gloss}’ | **{etym}** | {why} |')
lines += ['', 'Eight `reflex` rows affect eight records. **peṛek** ‘palm-F’ and **uncaṛek** ‘wellspring’ remain held because Turner qualifies their suggested ancestries. The “rice” gloss for **aśpāṛ** is broader than Turner’s “reaped rice”, and that limitation is preserved in the saved evidence. No source ingestion or rebuild.', '', 'Validation pending.']
(base / 'pass153-review.md').write_text('\n'.join(lines) + '\n')
print('accepted', len(accepted), 'held', len(held))
