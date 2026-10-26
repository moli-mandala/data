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
    ('f_bufwa5fcvmkbg', '9424', 'bhástrā-',
     'Dhakal p. 46 prints /bʰãtʰi/ “bellow.” Turner9424 gives Bihari south-of-Ganges bhā̃thī “bellows” and Nepali bhā̃ṭi “bellows”; the nasal vowel, aspirated initial and specialized sense all agree.'),
    ('f_7sf336t53nlsi', '4804', 'citraka-',
     'Dhakal p. 59 prints /cit.wa/ “leopard.” Turner4804 gives Nepali cituwā “leopard” alongside Maithili cittā and other regional forms; the Darai -tva is especially close to the Nepali reflex.'),
]

accepted, audit = [], []
review = ['# Darai: bellows and leopard', '',
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
    ('kunʣi', 'Turner3225 marks Nepali kuñji as borrowed from Punjabi; Darai has no source statement identifying whether its immediate donor is Nepali, Punjabi or another local form.'),
    ('dam', 'Turner6622 traces dramma through Greek and Middle Persian, while modern Nepali/Bihari dām is exact; Darai source does not settle immediate transmission.'),
    ('lobʰ', 'Turner11147 inherited eastern comparanda have loh-, whereas Darai retains bh as a likely learned or borrowed form; the immediate donor is not identified.'),
]:
    for r in rows.values():
        if r['Form'] == form and r['Source'].startswith('dhakal2011darai[') and r['ID'] not in linked:
            holds.append(dict(record=rec(r), reason=reason))

assert len(accepted) == 2 and len(holds) == 5, (len(accepted), len(holds))
(P / 'pass181-decisions.json').write_text(json.dumps(dict(accepted=accepted, held=holds), ensure_ascii=False, indent=2) + '\n')
(P / 'pass181-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(audit), rows=audit), ensure_ascii=False, indent=2) + '\n')
review += ['', 'Two affected records and two reflex rows. Held: two *kunʣi* “key,” one *dam* “cost,” and two *lobʰ* “greed” records for uncertain immediate transmission. Earlier held Darai forms remain excluded. No lexical-source, identity or build changes.',
           'Primary source: Dhakal 2011 printed pp. 46 and 59, visually checked in source PDF images and decoded text. Discovery: darai-search075–078; complete CDIAL articles darai-primary064–066. Turner3225 explicitly marks Nepali *kuñji* as borrowed from Punjabi; Jambu compiled edges already represent this, so it is not a contamination finding.',
           'Validation: pending shared heavy slot.']
(P / 'pass181-review.md').write_text('\n'.join(review) + '\n')
print(len(accepted), 'accepted', len(holds), 'held')
