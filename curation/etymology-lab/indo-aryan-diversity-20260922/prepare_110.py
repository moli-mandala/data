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
    ('f_hi4guni7dgaau', '8857', 'prastará',
     'Dhakal p. 67 prints /patʰər/ “stone” again in the consonant-onset table. Turner8857 gives Maithili pathar, Bengali pāthar and Oriya pathara “stone,” with Pali/Prakrit patthara. This separate installed source token shares the p. 53 stone form and etymon, whose first record was linked in pass102.'),
    ('f_ph7dyf4phroho', '579', 'amlá',
     'Dhakal p. 61 prints /amlə/ “sour.” Turner579 gives Sanskrit amlá “sour,” Prakrit aṁbila, Nepali amilo, Bengali āmbal and Oriya āmiḷa “sour”; Darai preserves the recognizable aml- adjective stem.'),
    ('f_6da5xzborrbqu', '385', 'andhá',
     'Dhakal p. 61 prints /əndʰa/ “blind (male).” Turner385 gives Sanskrit andhá, Pali andha, Prakrit aṁdha, Oriya andhā “blind man” and Hindi andhā “blind”; the Darai voiced aspirate and meaning select this article, not the -la- extension at 385-2.'),
    ('f_msjf6bmulpksg', '2633-2', '*kaṭaccʰul-',
     'Dhakal p. 62 prints /kərʦʰula/ “ladle.” Turner2633 explicitly separates an -l- extension, installed as CDIAL2633-2, and gives Nepali karchul “ladle,” Bihari karchul and Hindi karchul; the Darai -ul- stem matches that extension rather than the base kaṭacchu.'),
]

accepted, audit = [], []
review = ['# Darai: stone, sour, blind, and ladle', '',
          'Four source-backed reflex records across four etyma.', '',
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
    ('ləŋḍa', 'The p. 61 “lame” form resembles Turner10877 section 4 laṅga, but the retroflex stop and possible Hindi or Nepali contact make the exact pathway uncertain.'),
    ('girgəṭi', 'The p. 62 “lizard” form resembles common modern girgiṭ forms, but actual-homepage searches did not establish a specific Turner ancestral branch.'),
]:
    for r in rows.values():
        if r['Form'] == form and r['Source'].startswith('dhakal2011darai[') and r['ID'] not in linked:
            holds.append(dict(record=rec(r), reason=reason))

assert len(accepted) == 4 and len(holds) == 2, (len(accepted), len(holds))
(P / 'pass110-decisions.json').write_text(json.dumps(dict(accepted=accepted, held=holds), ensure_ascii=False, indent=2) + '\n')
(P / 'pass110-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(audit), rows=audit), ensure_ascii=False, indent=2) + '\n')
review += ['', 'Four affected records and four reflex rows. Held: *ləŋḍa* and *girgəṭi*. Earlier held Darai forms remain excluded. No lexical-source, identity or build changes.',
           'Primary source: Dhakal 2011 printed pp. 61–62 and 67, visually checked in source PDF images and decoded text. Discovery: darai-search039; complete CDIAL articles darai-primary032 and darai-primary039.',
           'Validation: pending.']
(P / 'pass110-review.md').write_text('\n'.join(review) + '\n')
print(len(accepted), 'accepted', len(holds), 'held')
