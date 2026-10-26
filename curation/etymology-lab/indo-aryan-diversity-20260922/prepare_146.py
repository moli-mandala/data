import csv
import json
from pathlib import Path

from etymology_assignments import read_assignments

P = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
linked = {r['Form_ID'] for r in read_assignments() if r['Status'] == 'accepted' and r['Rank'] == '1'}
linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank'] == '1')
rows = {r['ID']: r for r in csv.DictReader(open('cldf/forms.csv'))}
rec = lambda r: {k: r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')}

groups = [
    ('f_hlqteayk5cvvm', '3855-2', '*khasara-',
     'Dhakal p. 60 prints /kʰəs.rə/ “rough.” Turner3855 section 2 gives Nepali khasro “rough, coarse, thick” under *khasara-. The Darai s-r sequence and sense select section 2; section 1 has shorter khas- forms.'),
    ('f_rtjz7kcjahvnm', '4816', 'cintā́',
     'Dhakal p. 61 prints /cin.ta/ “worry.” Turner4816 gives Old Gujarati cīṁta “worry,” with Pali cintā and Prakrit ciṁtā “thought, care”; form and sense are direct matches.'),
]

accepted, audit = [], []
review = ['# Darai: rough and worry', '',
          'Two source-backed reflex records across two etyma.', '',
          '| # | Darai form | Proposed etymology | Evidence |', '|---|---|---|---|']
source_dir = Path('data/other/forms/raw_data/dhakal_darai_2011/decoded')
for i, (id_, parent, label, evidence) in enumerate(groups, 1):
    r = rows[id_]
    assert id_ not in linked and r['Source'].startswith('dhakal2011darai['), id_
    page = int(r['Source'].split('[p. ')[1].split(']')[0])
    source_text = (source_dir / f'p{page:03}.txt').read_text()
    assert r['Gloss'] in source_text, (id_, page, r['Gloss'])
    audit.append(dict(id=id_, form=r['Form'], gloss=r['Gloss'], source=r['Source'], page=page,
                      sourceText=source_text))
    accepted.append(dict(record=rec(r), parent=parent, kind='reflex',
                         citation=r['Source'] + f';CDIAL[{parent}]', evidence=evidence))
    review.append(f'| {i} | **{r["Form"]}** ‘{r["Gloss"]}’ ({r["Source"]}) | **{label}**, CDIAL **{parent}** | {evidence} |')

holds = []
for form, reason in [
    ('bisvas', 'Dhakal p. 60 gives “trust” and Turner11966 viśvāsa has the same sense, but Darai bisvas appears learned relative to Turner’s vernacular bisās; immediate Nepali/learned transmission is unresolved.'),
    ('nəmda', 'Dhakal p. 60 gives “husband’s younger sister,” corresponding semantically to Turner6946 nánāndr̥ and Nepali nanda, but the Darai m-d cluster has no demonstrated sound development.'),
    ('səmʦar', 'Dhakal p. 61 gives “message,” but searches for samācāra and the Darai shape found no precise Turner article or branch.'),
]:
    for r in rows.values():
        if r['Form'] == form and r['Source'].startswith('dhakal2011darai[') and r['ID'] not in linked:
            holds.append(dict(record=rec(r), reason=reason))

assert len(accepted) == 2 and len(holds) == 3, (len(accepted), len(holds))
(P / 'pass146-decisions.json').write_text(json.dumps(dict(accepted=accepted, held=holds), ensure_ascii=False, indent=2) + '\n')
(P / 'pass146-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(audit), rows=audit), ensure_ascii=False, indent=2) + '\n')
review += ['', 'Two affected records and two reflex rows. Held: *bisvas* “trust,” *nəmda* “husband’s younger sister,” and *səmʦar* “message.” Earlier held Darai forms remain excluded. No lexical-source, identity or build changes.',
           'Primary source: Dhakal 2011 printed pp. 60–61, visually checked in source PDF images and decoded text. Discovery: darai-search057–060; complete CDIAL articles darai-primary053–054.',
           'Validation: pending.']
(P / 'pass146-review.md').write_text('\n'.join(review) + '\n')
print(len(accepted), 'accepted', len(holds), 'held')
