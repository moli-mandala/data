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
    ('f_dycn2mi6bgzfy', '8181', 'pittá',
     'Dhakal p. 58 prints /pittə/ “bile.” Turner8181 gives Old Indo-Aryan pittá, Pali and Prakrit pitta, Oriya pita and Assamese pit “bile”; the Darai dental geminate and meaning match. Turner notes possible ultimate Munda origin, but this Indo-Aryan word is the relevant ancestor.'),
    ('f_htup3s666kivc', '1341', 'ārdraka',
     'Dhakal p. 59 prints /ədva/ “ginger.” Turner1341 section 1 gives Nepali aduwā, Garhwali ādu, Bengali ādā and Oriya adā “ginger”; Darai ədv- aligns especially with the Nepali aduwā form while the eastern comparanda establish the wider inherited family.'),
    ('f_fvns34j5wqnza', '11239', 'vatsá',
     'Dhakal p. 60 prints /baʦʦʰi/ “a calf.” Turner11239 gives Pali and Prakrit vaccha “calf,” Nepali bācho/°chi, Oriya bāchī “heifer calf,” and Bhojpuri bāchī “calf”; the Darai feminine -i and geminate affricate match this branch.'),
]

accepted, audit = [], []
review = ['# Darai: bile, ginger, and calf', '',
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
    ('hərdarə', 'The p. 62 “green” form resembles Turner13985 hárita/haritara but the -dar- sequence and exact regional branch need explanation.'),
    ('səmmə', 'The p. 60 “flat” form is semantically near Turner13173 samá, but its geminate -mm- and immediate pathway remain unestablished.'),
]:
    for r in rows.values():
        if r['Form'] == form and r['Source'].startswith('dhakal2011darai[') and r['ID'] not in linked:
            holds.append(dict(record=rec(r), reason=reason))

assert len(accepted) == 3 and len(holds) == 2, (len(accepted), len(holds))
(P / 'pass113-decisions.json').write_text(json.dumps(dict(accepted=accepted, held=holds), ensure_ascii=False, indent=2) + '\n')
(P / 'pass113-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(audit), rows=audit), ensure_ascii=False, indent=2) + '\n')
review += ['', 'Three affected records and three reflex rows. Held: *hərdarə* and *səmmə*. Earlier held Darai forms remain excluded. No lexical-source, identity or build changes.',
           'Primary source: Dhakal 2011 printed pp. 58–60, visually checked in source PDF images and decoded text. Discovery: darai-search040–042; complete CDIAL articles darai-primary040–042.',
           'Validation: pending.']
(P / 'pass113-review.md').write_text('\n'.join(review) + '\n')
print(len(accepted), 'accepted', len(holds), 'held')
