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
    ('f_yn4wuix6o6v3a', '5055', 'chupáti',
     'Dhakal p. 47 prints /cʰiulə/ “touched” as an /iu/ diphthong example; p. 73 identifies -lə as the past marker. Turner5055 gives Oriya cʰũibā and OAwadhi chuvaï “touches,” beside Nepali chunu; the Darai cʰiu- stem and past inflection fit this family.'),
    ('f_3by3n4wcdyavq', '5055', 'chupáti',
     'Dhakal p. 54 prints /cʰike/ “to touch” among aspirated affricates, distinct from unaspirated /cirike/ “to split.” Turner5055 gives Nepali chunu, Maithili chūyab and Hindi chūnā “touches”; Darai cʰi- is the same touch stem.'),
    ('f_wzhkqdxgbfnlw', '5055', 'chupáti',
     'Dhakal p. 67 repeats /cʰi.ke/ “to touch” in the attested-onset table. Turner5055 lists regional cʰu-/cʰui- touch verbs, including Oriya cʰũibā and Maithili chūyab; the independently cited token is the same Darai stem.'),
]

accepted, audit = [], []
review = ['# Darai: touch verb', '',
          'Three source-backed reflex records in one etymon.', '',
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
    ('ʦʰamike', 'Turner4996 gives Nepali chāmnu “to grope, touch, feel,” but explicitly marks the semantic link to the proposed *chambhayati as very doubtful; hold rather than attaching the doubtful head.'),
    ('uḍʰike', 'Dhakal p. 53 prints an aspirated retroflex in “fly.” Turner1697 §2 has regional unaspirated uḍ-/uṛ- forms, but the Darai aspiration and exact section need independent confirmation.'),
    ('gitʰike', 'Turner4205 has several crossing and contaminated “weave” branches; Dhakal p. 53 /gitʰike/ lacks enough phonological detail for a precise branch.'),
]:
    for r in rows.values():
        if r['Form'] == form and r['Source'].startswith('dhakal2011darai[') and r['ID'] not in linked:
            holds.append(dict(record=rec(r), reason=reason))

assert len(accepted) == 3 and len(holds) == 3, (len(accepted), len(holds))
(P / 'pass134-decisions.json').write_text(json.dumps(dict(accepted=accepted, held=holds), ensure_ascii=False, indent=2) + '\n')
(P / 'pass134-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(audit), rows=audit), ensure_ascii=False, indent=2) + '\n')
review += ['', 'Three affected records and three reflex rows. Held: *ʦʰamike* “feel,” *uḍʰike* “fly,” and *gitʰike* “weave.” Earlier held Darai forms remain excluded. No lexical-source, identity or build changes.',
           'Primary source: Dhakal 2011 printed pp. 47, 54, 67, visually checked in source PDF images and decoded text; p. 73 confirms the past suffix. Discovery: darai-search055; complete CDIAL articles darai-primary051.',
           'Validation: pending.']
(P / 'pass134-review.md').write_text('\n'.join(review) + '\n')
print(len(accepted), 'accepted', len(holds), 'held')
