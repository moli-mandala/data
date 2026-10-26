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
    ('f_zah35egvgsac6', '5827', 'tilá',
     'Dhakal p. 44 prints /til/ “sesame” in the /i/ versus /e/ minimal pair. Turner5827 section 1 gives Pali tila and Prakrit tila, with Nepali and Bengali til “sesamum seed”; this is the common seed noun, not the homonymous mole term at 5828.'),
    ('f_il5gvp52mlesk', '5827', 'tilá',
     'Dhakal p. 44 repeats /til/ “sesame” in Table 3.1 for medial /i/. Turner5827 section 1 gives Nepali, Bengali and Assamese til “sesamum seed,” alongside Pali and Prakrit tila. The distinct installed example record has the same lexical parent.'),
    ('f_hx3twa7mimhgo', '8857', 'prastará',
     'Dhakal p. 53 prints /patʰər/ “stone.” Turner8857 explicitly gives Maithili pathar, Bengali pāthar and Oriya pathara “stone,” with corresponding Pali and Prakrit patthara; the Darai stem and meaning match this branch.'),
    ('f_3igxghpc77dq2', '1200', 'āpayati',
     'Dhakal p. 48 prints /aike/ “to come,” where -ike is the Darai infinitive ending. Turner1200 lists Prakrit āvēi/āvaï “comes,” Nepali āunu, Oriya āibā and Old Maithili āvaï; the Darai ai- stem matches this widespread “come” family.'),
    ('f_dpihbpe5m6p4m', '4571', 'caṭaka',
     'Dhakal p. 54 prints /ʦərəi/ “bird.” Turner4571 section 1 gives Oriya caṛāi and Bengali caṛāi “bird,” with Nepali caro and an eastern -āka extension. The Darai form matches the eastern bird noun rather than section 2 ciṛ- “sparrow.”'),
    ('f_rvfmw2k6j3o3c', '13364', 'sā́rtha',
     'Dhakal p. 54 prints /satʰ/ “together, accompany.” Turner13364 section 3 gives Nepali sātha, Bengali sāthe, Oriya sātha and Maithili sāth in the “with/in company” sense; Darai satʰ continues that adverbial/postpositional branch.'),
]

accepted, audit = [], []
review = ['# Darai: sesame, stone, come, bird, and together', '',
          'Six source-backed reflex records across five etyma.', '',
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
    ('ʦʰike', 'The p. 54 “to touch” form has a short ch-i- stem, whereas Turner5055 eastern comparanda are ch-u-/ch-ũ-; the exact vowel and formation remain uncertain.'),
    ('kumʰar', 'The p. 55 Kumal ethnonym resembles Turner3310 potter terms, including Nepali kum(h)āle, but its immediate Nepali contact path and ethnic-name semantic history require review.'),
]:
    for r in rows.values():
        if r['Form'] == form and r['Source'].startswith('dhakal2011darai[') and r['ID'] not in linked:
            holds.append(dict(record=rec(r), reason=reason))

assert len(accepted) == 6 and len(holds) == 3, (len(accepted), len(holds))
(P / 'pass102-decisions.json').write_text(json.dumps(dict(accepted=accepted, held=holds), ensure_ascii=False, indent=2) + '\n')
(P / 'pass102-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(audit), rows=audit), ensure_ascii=False, indent=2) + '\n')
review += ['', 'Six affected records and six reflex rows. Held: two *ʦʰike* records (pp. 54, 67) and *kumʰar*. Earlier held Darai forms remain excluded. No lexical-source, identity or build changes.',
           'Primary source: Dhakal 2011 printed pp. 44, 48, 53–54, visually checked in source PDF images and decoded text. Discovery: darai-search031–035; complete CDIAL articles darai-primary032–035.',
           'Validation: pending.']
(P / 'pass102-review.md').write_text('\n'.join(review) + '\n')
print(len(accepted), 'accepted', len(holds), 'held')
