import csv
import json
from pathlib import Path

from etymology_assignments import read_assignments

P = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
linked = {r['Form_ID'] for r in read_assignments() if r['Status'] == 'accepted' and r['Rank'] == '1'}
linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank'] == '1')
rows = [r for r in csv.DictReader(open('cldf/forms.csv')) if r['Source'].startswith('dhakal2011darai[') and r['ID'] not in linked]
rec = lambda r: {k: r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')}

groups = [
    ('ubʣike', 'to grow', '1814', 'útpadyatē',
     'Dhakal p. 58 prints /ubdzike/ “to grow.” Turner1814 section 1 explicitly gives Nepali ubjanu, Oriya upajibā, Bihari and Maithili upajab “to grow”; the Darai ubʣ- cluster fits the regional ubj-/upaj- stem. -ike is the ordinary infinitive.'),
    ('bʰeṭlə', 'met', '9490-4', '*bhēṭṭ-',
     'Dhakal p. 59 prints /bʰeṭlə/ “met.” Turner9490 section 4 lists Nepali bheṭnu, Maithili bheṭab and Oriya bheṭibā “to meet,” whereas sections 1–3 concern other bhiṭ-/bhēṭ- developments. Darai -lə is the source-described past suffix, so this is an inflected reflex of the section-4 stem.'),
    ('gaḍlarə', 'bury', '3979', '*gaḍḍ-',
     'Dhakal p. 59 prints /gaḍlarə/ “bury-PROS”; the installed gloss shortens it to “bury.” Turner3979 gives Nepali gāṛnu, Bengali gāṛā, Oriya gāṛibā and Maithili gāṛab “bury.” The prospective suffix is source morphology, not a separate etymon.'),
    ('pəglike', 'to melt', '8468', 'pragalati',
     'Dhakal p. 60 prints /pəglike/ “to melt.” Turner8468 section 1 gives Nepali paglanu, Kumaoni pagalṇo and West Pahari pɔgəḷno “to melt”; these match the Darai pagl- stem and intransitive sense, rather than section 3 transitive pagālnu.'),
    ('kʰoʣlə', 'searched', '3929-2', '*khōjjati',
     'Dhakal p. 68 prints /kʰodzlə/ “searched.” Turner3929 section 2 gives Nepali khojnu, Bengali khojā, Oriya khojibā and Maithili khōjab “to search”; section 1 is the noun “footprint.” Darai -lə is the source-described past suffix, leaving the matching khōj- stem.'),
]
accepted, audit = [], []
review = ['# Darai: grow, meet, bury, melt and search', '',
          'Five source-backed reflex records across five etyma.', '',
          '| # | Darai form | Proposed etymology | Evidence |', '|---|---|---|---|']
source_dir = Path('data/other/forms/raw_data/dhakal_darai_2011/decoded')
for i, (form, gloss, parent, label, evidence) in enumerate(groups, 1):
    hits = [r for r in rows if r['Form'] == form and r['Gloss'] == gloss]
    assert len(hits) == 1, (form, gloss, len(hits))
    r = hits[0]
    page = int(r['Source'].split('[p. ')[1].split(']')[0])
    source_text = (source_dir / f'p{page:03}.txt').read_text()
    assert gloss in source_text, (form, page)
    audit.append(dict(id=r['ID'], form=form, gloss=gloss, source=r['Source'], page=page,
                      sourceText=source_text))
    accepted.append(dict(record=rec(r), parent=parent, kind='reflex',
                         citation=r['Source'] + f';CDIAL[{parent}]', evidence=evidence))
    review.append(f'| {i} | **{form}** ‘{gloss}’ ({r["Source"]}) | **{label}**, CDIAL **{parent}** | {evidence} |')

holds = []
for form, reason in [
    ('biʦʰike', 'The p. 54 “pluck” infinitive has no specific Turner branch established by the actual-homepage searches; the bi- prefix and ch- stem require analysis.'),
    ('biʦʰlə', 'The p. 68 “plucked” form is likely the past of biʦʰike, but the parent of their shared stem remains unresolved.'),
    ('pʰəṭkike', 'The p. 59 “jump, leap” form may be expressive; no exact inherited Turner branch has been demonstrated.'),
    ('ʦoḍkalə', 'The p. 59 “bit” form has an internal -k- and lacks a secure relation to the separately printed kapṭike “to bite.”'),
]:
    for r in rows:
        if r['Form'] == form:
            holds.append(dict(record=rec(r), reason=reason))

assert len(accepted) == 5 and len(holds) == 4, (len(accepted), len(holds))
(P / 'pass093-decisions.json').write_text(json.dumps(dict(accepted=accepted, held=holds), ensure_ascii=False, indent=2) + '\n')
(P / 'pass093-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(audit), rows=audit), ensure_ascii=False, indent=2) + '\n')
review += ['', 'Five affected records and five reflex rows. Held: *biʦʰike*, *biʦʰlə*, *pʰəṭkike* and *ʦoḍkalə*. Earlier held Darai forms remain excluded. No lexical-source, identity or build changes.',
           'Primary source: Dhakal 2011 printed pp. 58–60 and 68, plus p. 75 past/infinitive morphology, visually checked in source PDF images and decoded text. Discovery: darai-search027–028; complete CDIAL articles darai-primary027–028.',
           'Validation: pending.']
(P / 'pass093-review.md').write_text('\n'.join(review) + '\n')
print(len(accepted), 'accepted', len(holds), 'held')
