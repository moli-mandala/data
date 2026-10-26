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
    ('bok', 'carry', '9311', '*bōkk-',
     'Dhakal p. 52 prints /bok/ “carry.IMP,” contrasting aspirated /bʰok/ “hunger.” Turner9311 specifically gives Nepali boknu, Kumaoni bokṇo and Garhwali bokṇu “carry”; the Darai imperative is an ordinary form of the same bok- stem.'),
    ('dʰər', 'hold', '6747', 'dhárati',
     'Dhakal p. 52 prints /dʰər/ “hold.IMP,” distinct from /dər/ “rate.” Turner6747 gives Pali dharati, Prakrit dharaï, Nepali dharnu, Maithili dharab and Oriya dharibā with “hold”; this is the dhar- verbal branch rather than dhārayati 6791.'),
    ('gʰali', 'pour', '4457', '*ghalyati',
     'Dhakal p. 52 prints /gʰali/ “pour.ABS,” opposite /gali/ “scolding, abuse.” Turner4457 has Prakrit ghallaï “throws,” Old Awadhi ghālai “pours in,” and Old Marathi ghālaṇeṁ “to pour”; the Darai absolutive ending belongs to the response, not the ancestral stem.'),
    ('tʰapike', 'to set up/spread', '13759', 'sthāpyatē',
     'Dhakal p. 52 prints /tʰapike/ “to set up/spread,” contrasting unaspirated /tapike/ “to warm.” Turner13759 gives Prakrit thappaṇa, Nepali thāpnu, Oriya thāpibā and Hindi thāpnā “set up, establish”; the first source sense and aspirate select this family.'),
    ('paṭʰi', 'female kid', '8015', '*paṣṭha-',
     'Dhakal p. 52 prints /paṭʰi/ “female kid,” contrasting /paṭi/ “rest-house.” Turner8015 section 1 gives feminine Nepali pāṭhi, Maithili pāṭhī and Oriya peṇṭhī “kid,” along with Assamese pāṭhī “she-goat”; the feminine and aspirated stop support this branch rather than the distinct section-2 meanings.'),
]

accepted, audit = [], []
review = ['# Darai: five source-page contrasts', '',
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
    ('kəsike', 'The p. 52 “to tighten” form is semantically like some distant CDIAL2908 reflexes, but an East Indo-Aryan comparison and precise vowel development are unestablished.'),
    ('gali', 'The p. 52 “scolding, abuse” form may match CDIAL4145 through a regional borrowing; donor path versus inherited continuation needs review.'),
]:
    for r in rows:
        if r['Form'] == form:
            holds.append(dict(record=rec(r), reason=reason))

assert len(accepted) == 5 and len(holds) == 2, (len(accepted), len(holds))
(P / 'pass098-decisions.json').write_text(json.dumps(dict(accepted=accepted, held=holds), ensure_ascii=False, indent=2) + '\n')
(P / 'pass098-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(audit), rows=audit), ensure_ascii=False, indent=2) + '\n')
review += ['', 'Five affected records and five reflex rows. Held: *kəsike* and *gali*. Earlier held Darai forms remain excluded. No lexical-source, identity or build changes.',
           'Primary source: Dhakal 2011 printed p. 52, visually checked in source PDF image and decoded text. Discovery: darai-search029–030; complete CDIAL articles darai-primary029–031.',
           'Validation: pending.']
(P / 'pass098-review.md').write_text('\n'.join(review) + '\n')
print(len(accepted), 'accepted', len(holds), 'held')
