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
    ('f_b3dv3ubbvxkic', '3594', 'krīṇā́ti',
     'Dhakal p. 61 prints /kin.la.rə/ “buy.PROS”; p. 56 identifies -larʰə as the prospective marker. The root kin- matches Turner3594, which gives Nepali kinnu “to buy,” Maithili kinab and Bhojpuri kinal. The prospective suffix is Darai morphology, not part of the etymon.'),
    ('f_fdj6b2ugrqzxk', '2735', 'kanyasa-',
     'Dhakal p. 61 prints /kan.cʰə/ “the youngest one.” Turner2735 gives Nepali kāncho “younger, the youngest brother,” and its addenda Jaunsari kaṇchā “younger, youngest.” This is the affricate branch of kanyasa-, not the separate kaniṣṭhá article 2718.'),
    ('f_v6nboxvql6dyw', '1371', 'ā́lasya',
     'Dhakal p. 62 prints /əl.cʰi/ “lazy.” Turner1371 lists adjective descendants of ālasya, including Nepali alchi “lazy,” alongside Punjabi ālsī and Oriya āḷasi. The Darai affricate adjective matches the Nepali form closely.'),
]

accepted, audit = [], []
review = ['# Darai: buy, youngest, and lazy', '',
          'Three source-backed reflex records across three etyma.', '',
          '| # | Darai form | Proposed etymology | Evidence |', '|---|---|---|---|']
source_dir = Path('data/other/forms/raw_data/dhakal_darai_2011/decoded')
for i, (id_, parent, label, evidence) in enumerate(groups, 1):
    r = rows[id_]
    assert id_ not in linked and r['Source'].startswith('dhakal2011darai['), id_
    page = int(r['Source'].split('[p. ')[1].split(']')[0])
    source_text = (source_dir / f'p{page:03}.txt').read_text()
    assert r['Gloss'] in source_text or (id_ == 'f_b3dv3ubbvxkic' and 'buy. PROS' in source_text), (id_, page, r['Gloss'])
    audit.append(dict(id=id_, form=r['Form'], gloss=r['Gloss'], source=r['Source'], page=page,
                      sourceText=source_text))
    accepted.append(dict(record=rec(r), parent=parent, kind='reflex',
                         citation=r['Source'] + f';CDIAL[{parent}]', evidence=evidence))
    review.append(f'| {i} | **{r["Form"]}** ‘{r["Gloss"]}’ ({r["Source"]}) | **{label}**, CDIAL **{parent}** | {evidence} |')

holds = []
for form, reason in [
    ('unḍike', 'The p. 49 and p. 61 fall verb has no secure matching Turner article from the homepage search; the retrieved luṇḍi form at 4497 is semantically similar but phonologically insufficient.'),
    ('kʰonʦalə', 'The p. 61 crippled verb has no exact Turner match in homepage searches; defer the root segmentation and parent.'),
    ('kʰalṭa', 'The p. 62 ditch noun has no secure Turner match from homepage searches; a broad hollow/ditch resemblance is inadequate.'),
]:
    for r in rows.values():
        if r['Form'] == form and r['Source'].startswith('dhakal2011darai[') and r['ID'] not in linked:
            holds.append(dict(record=rec(r), reason=reason))

assert len(accepted) == 3 and len(holds) == 4, (len(accepted), len(holds))
(P / 'pass126-decisions.json').write_text(json.dumps(dict(accepted=accepted, held=holds), ensure_ascii=False, indent=2) + '\n')
(P / 'pass126-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(audit), rows=audit), ensure_ascii=False, indent=2) + '\n')
review += ['', 'Three affected records and three reflex rows. Held: two *unḍike* “fall” records, *kʰonʦalə* “crippled,” and *kʰalṭa* “ditch.” Earlier held Darai forms remain excluded. No lexical-source, identity or build changes.',
           'Primary source: Dhakal 2011 printed pp. 61–62, visually checked in source PDF images and decoded text; p. 56 confirms the prospective suffix. Discovery: darai-search051–052; complete CDIAL articles darai-primary048–049.',
           'Validation: pending.']
(P / 'pass126-review.md').write_text('\n'.join(review) + '\n')
print(len(accepted), 'accepted', len(holds), 'held')
