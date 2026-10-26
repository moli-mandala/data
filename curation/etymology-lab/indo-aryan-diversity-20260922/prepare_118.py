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
    ('f_oyume5gt2ryf6', '11920', 'vivāhá',
     'Dhakal p. 57 prints /byaha/ “marriage” as an example of the initial by- cluster. Turner11920 gives Sanskrit vivāhá, Prakrit viāha, Maithili biāh, Bihari biyāh and Hindi byāh “marriage”; the Darai byah- form matches this broad Indo-Aryan continuation.'),
    ('f_mcj2lpxlzvy6i', '11225-2', '*vaḍrakk-',
     'Dhakal p. 66 prints /bəḍkə/ “big.” Turner11225 gives Prakrit vaḍḍa “big” and Old Awadhi baṛkā “big”; the installed Turner-derived 11225-2 is specifically the inferred -kk- extension of vaḍra corresponding to the Darai -ḍk- adjectival stem.'),
    ('f_sarvsok5ntv2k', '8652-2', '*prathilla-',
     'Dhakal p. 69 prints /pəhele/ “first.” Turner8652 section 2 gives Prakrit pahilla-, Kumaoni pailo, Nepali pailo, Oriya pahila and Hindi pahilā “first”; the Darai pəhel- form matches this -ll- branch rather than section 1 *prathila.'),
]

accepted, audit = [], []
review = ['# Darai: marriage, big, and first', '',
          'Three source-backed reflex records across three etyma.', '',
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
    ('kunʣi', 'Two source tokens “key” resemble Nepali kuñji. Turner3225 explicitly marks Nepali kuñji as borrowed from Sindhi or Punjabi and Jambu has two corresponding Nepali donor nodes; Darai immediate donor and exact donor node cannot be selected securely.'),
    ('aŋṭʰi', 'The p. 61 “finger ring” form lacks a specific Turner head in the actual-homepage searches; the superficially similar kaṇṭhikā is a necklace, not a finger ring.'),
]:
    for r in rows.values():
        if r['Form'] == form and r['Source'].startswith('dhakal2011darai[') and r['ID'] not in linked:
            holds.append(dict(record=rec(r), reason=reason))

assert len(accepted) == 3 and len(holds) == 3, (len(accepted), len(holds))
(P / 'pass118-decisions.json').write_text(json.dumps(dict(accepted=accepted, held=holds), ensure_ascii=False, indent=2) + '\n')
(P / 'pass118-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(audit), rows=audit), ensure_ascii=False, indent=2) + '\n')
review += ['', 'Three affected records and three reflex rows. Held: two *kunʣi* “key” records and *aŋṭʰi* “finger ring.” Earlier held Darai forms remain excluded. No lexical-source, identity or build changes.',
           'Primary source: Dhakal 2011 printed pp. 57, 66 and 69, visually checked in source PDF images and decoded text. Discovery: darai-search043–047; complete CDIAL articles darai-primary043–044.',
           'Validation: pending.']
(P / 'pass118-review.md').write_text('\n'.join(review) + '\n')
print(len(accepted), 'accepted', len(holds), 'held')
