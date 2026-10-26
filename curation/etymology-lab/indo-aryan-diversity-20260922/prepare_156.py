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
    ('f_5e4gi6o2lu4ge', '11083', 'lúpyatē / *lukka- replacement',
     'Dhakal p. 56 contrasts /lo.ki.ke/ “to hide” with /ro.ki.ke/ “to stop.” Turner11083 section 3 lists the MIA *lukka- replacement, Nepali luknu and Oriya lukibā “hide,” plus Old Awadhi lokāvaï with the same o-vowel as Darai. The Darai lo- stem belongs to the hide family.'),
    ('f_zzvu5gmq4bzlq', '11418', 'valkala',
     'Dhakal p. 59 prints /bak.lə/ “thick.” Turner11418 lists Nepali and Kumaoni bāklo “thick, coarse” and Western Pahari bāklu, a semantic extension of “bark”; Darai vowel length is noncontrastive, and the bakl- shape is close.'),
    ('f_p6ydtksvheudw', '11418', 'valkala',
     'Dhakal p. 68 independently repeats /bak.lə/ “thick.” Turner11418 directly gives Nepali bāklo and Kumaoni bāklo “thick, coarse,” matching the Darai adjective.'),
]

accepted, audit = [], []
review = ['# Darai: hide and thick', '',
          'Three source-backed reflex records across two etyma.', '',
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
    ('ʦutki', 'Dhakal pp. 58 and 66 give “little,” but chuṭkī/cuṭkī searches found no exact Turner head; the likely clipping or expressive formation needs evidence.'),
    ('nakno', 'Dhakal p. 59 gives “nose,” and Turner6909 has a widespread nak- nose family, but the Darai final -no is not explained by the source or article.'),
]:
    for r in rows.values():
        if r['Form'] == form and r['Source'].startswith('dhakal2011darai[') and r['ID'] not in linked:
            holds.append(dict(record=rec(r), reason=reason))

assert len(accepted) == 3 and len(holds) == 3, (len(accepted), len(holds))
(P / 'pass156-decisions.json').write_text(json.dumps(dict(accepted=accepted, held=holds), ensure_ascii=False, indent=2) + '\n')
(P / 'pass156-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(audit), rows=audit), ensure_ascii=False, indent=2) + '\n')
review += ['', 'Three affected records and three reflex rows. The source’s contrasting *rokike* “stop” was already linked to CDIAL10827 and remains untouched. Held: two *ʦutki* “little” records and *nakno* “nose.” Earlier held Darai forms remain excluded. No lexical-source, identity or build changes.',
           'Primary source: Dhakal 2011 printed pp. 56, 59, 68, visually checked in source PDF images and decoded text. Discovery: darai-search063–065; complete CDIAL articles darai-primary057–058.',
           'Validation: pending.']
(P / 'pass156-review.md').write_text('\n'.join(review) + '\n')
print(len(accepted), 'accepted', len(holds), 'held')
