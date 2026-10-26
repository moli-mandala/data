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
    ('f_qtodvwu34vpu6', '2908', 'kárṣati',
     'Dhakal p. 52 prints /kəsike/ “to tighten,” contrasting aspirated /kʰəsike/ “to fall down.” Turner2908 gives Nepali kasnu, Maithili kasab, Hindi kasnā and Gujarati kasvũ “to tighten”; Darai unaspirated kas- selects this family.'),
    ('f_kl7cu2pqyp2go', '9052-2', '*phara-',
     'Dhakal p. 53 prints /pʰar/ “blade.” Turner9052 section 2 gives Punjabi pharhā “blade, nib,” distinguished from section 1 phal- “blade”; the Darai rhotic stem selects the *phara- branch represented by node 9052-2.'),
    ('f_uusi7wqd25aym', '10893', 'lagna',
     'Dhakal p. 53 prints /ləgi/ “near.” Turner10893 gives Marathi lāgī̃ and Gujarati lāgu “near,” with Nepali lāgi as a postposition “for, to”; the Darai relational adverb matches this regional lag- family.'),
]

accepted, audit = [], []
review = ['# Darai: tighten, blade, and near', '',
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
    ('kəluva', 'Dhakal p. 56 gives “snacks,” but searches for kalēvā/kaluā found no exact Turner article or branch.'),
    ('leʣike', 'Dhakal p. 56 gives “to take” and Turner10948 has le- “take,” but the Darai affricate stem -ʣ- is not explained by the source or article.'),
    ('pəṭva', 'Dhakal p. 59 gives “fiber,” but paṭvā searches found no precise Turner etymon; possible cloth/plant words are not interchangeable.'),
]:
    for r in rows.values():
        if r['Form'] == form and r['Source'].startswith('dhakal2011darai[') and r['ID'] not in linked:
            holds.append(dict(record=rec(r), reason=reason))

assert len(accepted) == 3 and len(holds) == 3, (len(accepted), len(holds))
(P / 'pass161-decisions.json').write_text(json.dumps(dict(accepted=accepted, held=holds), ensure_ascii=False, indent=2) + '\n')
(P / 'pass161-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(audit), rows=audit), ensure_ascii=False, indent=2) + '\n')
review += ['', 'Three affected records and three reflex rows. Held: *kəluva* “snacks,” *leʣike* “to take,” and *pəṭva* “fiber.” Earlier held Darai forms remain excluded. No lexical-source, identity or build changes.',
           'Primary source: Dhakal 2011 printed pp. 52–53, visually checked in source PDF images and decoded text. Discovery: darai-search066; complete CDIAL articles darai-primary059.',
           'Validation: pending.']
(P / 'pass161-review.md').write_text('\n'.join(review) + '\n')
print(len(accepted), 'accepted', len(holds), 'held')
