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
    ('f_ipgwtfwbfsy6m', '6265', 'dā́na',
     'Dhakal p. 52 prints /dan/ “gift” against /dʰan/ “paddy.” Turner6265 gives Nepali and Bhojpuri dān “gift,” with the same unaspirated stop; this is dā́na¹, not the contrasting paddy word.'),
    ('f_hun22nbgxfr2u', '6265', 'dā́na',
     'Dhakal p. 55 independently prints /dan/ “gift” against /dam/ “cost.” Turner6265 gives Nepali and Bhojpuri dān “gift”; the second source token agrees exactly in form and sense.'),
    ('f_ymyciqqn446ee', '5684', 'tápyati',
     'Dhakal p. 52 prints /tapike/ “to warm” opposite aspirated /tʰapike/ “to set up/spread.” Turner5684 gives Nepali tāpnu, Maithili tāpab and Hindi tāpnā “to warm oneself,” with the Hindi transitive “to heat” derived from that verb. Darai tap- selects this heat family.'),
    ('f_uqg2enjp2iocg', '5684', 'tápyati',
     'Dhakal p. 58 prints /tap.lə/ “warmed”; p. 73 identifies -lə as the past marker. Turner5684 gives Nepali tāpnu and Maithili tāpab “to warm oneself,” plus Hindi transitive “to heat”; the Darai tap- root is the reflex and -lə is source-language inflection.'),
]

accepted, audit = [], []
review = ['# Darai: gift and warming', '',
          'Four source-backed reflex records across two etyma.', '',
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
    ('bʰar', 'Dhakal p. 52 gives only “load.” Turner9393 bhára and 9459 bhārá both include NIA bʰar “load”; Darai vowel length is noncontrastive, so the two heads cannot be separated by this token.'),
    ('mədʰu', 'Dhakal p. 55 gives “bee,” while Turner has several compound bee heads (9788 madʰukara, 9789 madʰukāra, 9792 madʰumakṣikā). The source does not establish which compound, if any, was shortened.'),
    ('dani', 'Dhakal p. 55 gives “generous,” but the searched Turner article for dā́na merely cross-references derivatives; exact immediate head and formation need independent support.'),
]:
    for r in rows.values():
        if r['Form'] == form and r['Source'].startswith('dhakal2011darai[') and r['ID'] not in linked:
            holds.append(dict(record=rec(r), reason=reason))

assert len(accepted) == 4 and len(holds) == 4, (len(accepted), len(holds))
(P / 'pass130-decisions.json').write_text(json.dumps(dict(accepted=accepted, held=holds), ensure_ascii=False, indent=2) + '\n')
(P / 'pass130-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(audit), rows=audit), ensure_ascii=False, indent=2) + '\n')
review += ['', 'Four affected records and four reflex rows. Held: *bʰar* “load,” two *mədʰu* “bee” records, and *dani* “generous.” Earlier held Darai forms remain excluded. No lexical-source, identity or build changes.',
           'Primary source: Dhakal 2011 printed pp. 52, 55, 58, visually checked in source PDF images and decoded text; p. 73 confirms the past suffix. Discovery: darai-search053–054; complete CDIAL articles darai-primary050.',
           'Validation: pending.']
(P / 'pass130-review.md').write_text('\n'.join(review) + '\n')
print(len(accepted), 'accepted', len(holds), 'held')
