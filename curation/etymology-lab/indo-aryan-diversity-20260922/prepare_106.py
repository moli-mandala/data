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
    ('f_gnayp7ng6phqy', '6225', 'davara',
     'Dhakal p. 47 prints /dauri/ “cord for tethering beast.” Turner6225 gives Hindi daurī “rope to which a string of cattle are fastened,” Nepali ḍori “rope, tether” and regional ḍorī/ḍori “rope.” The unnasalised daur-i shape and precise cattle-tether sense favor this article over the nasal dãurī branch under 6283.'),
    ('f_b7vt3ttmdha4a', '6886', '*dhauvati',
     'Dhakal p. 47 prints /dʰoulə/ “washed.” Turner6886 gives Prakrit dhō(v)aï “washes,” Nepali dhunu, Maithili dhoeb and Bhojpuri dhowal; Darai -lə is the source-described past ending, leaving the corresponding dʰou- washing stem. Turner6833 itself places Nepali dhunu probably under 6886.'),
    ('f_b54acfqtvgwaq', '8100', 'pārá',
     'Dhakal p. 51 prints /par/ “across/COND.PART”; the installed record retains the “across” sense. Turner8100 explicitly gives Kumaoni par “at a distance,” Nepali pār “on the other side of” and Maithili/Hindi pār “across.” This assignment addresses the spatial sense only; the source’s conditional-particle use may be a separate grammatical development.'),
    ('f_w5qbchryymzto', '12732', 'ślakṣṇá',
     'Dhakal p. 55 prints /nanʰi/ “small” among breathy-nasal examples. Turner12732 gives Prakrit laṇha “small,” Punjabi nannhā and Hindi nānh/nānhā “small,” plus Maithili nānhiṭā “childhood.” These comparanda support the Darai nanʰ- stem despite the wider ślakṣṇa family’s multiple regional shapes.'),
]

accepted, audit = [], []
review = ['# Darai: tether, wash, across, and small', '',
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
    ('agu', 'The p. 45 “in front of” form may continue Turner73 agratás (Maithili/Hindi āgu “in front”) or Turner94 agrēgū (Bengali āgu “front”); specific ancestry remains ambiguous.'),
    ('əurəti', 'The p. 47 “eyebrow” form is semantically near Turner9688 bhrū but its ərət- material is unexplained.'),
    ('ser', 'The p. 45 measurement noun exactly resembles Turner13106 ser, ultimately Iranian; an immediate borrowing path into Darai is not established.'),
]:
    for r in rows.values():
        if r['Form'] == form and r['Source'].startswith('dhakal2011darai[') and r['ID'] not in linked:
            holds.append(dict(record=rec(r), reason=reason))

assert len(accepted) == 4 and len(holds) == 3, (len(accepted), len(holds))
(P / 'pass106-decisions.json').write_text(json.dumps(dict(accepted=accepted, held=holds), ensure_ascii=False, indent=2) + '\n')
(P / 'pass106-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(audit), rows=audit), ensure_ascii=False, indent=2) + '\n')
review += ['', 'Four affected records and four reflex rows. Held: *agu*, *əurəti* and *ser*. Earlier held Darai forms remain excluded. No lexical-source, identity or build changes.',
           'Primary source: Dhakal 2011 printed pp. 47, 51, 55, plus p. 75 past morphology; visually checked in source PDF images and decoded text. Discovery: darai-search036–038; complete CDIAL articles darai-primary036–038.',
           'Validation: pending.']
(P / 'pass106-review.md').write_text('\n'.join(review) + '\n')
print(len(accepted), 'accepted', len(holds), 'held')
