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
    ('f_a2euxbsvpxdp6',
     'Dhakal p. 53 prints /dʰijaputa/ “children.” Turner8266 putraká explicitly cites Bihari dhiyāputā “sons and daughters,” an exact regional compound parallel. The first member is the daughter stem treated in Turner6481; this row follows Turner’s article for the attested compound, not a claim that the whole word descends from a simplex putraká.'),
    ('f_jhozxq3iuuify',
     'Dhakal p. 67 repeats /dʰi.ja.pu.ta/ “children.” Turner8266 explicitly lists Bihari dhiyāputā “sons and daughters” under putraká; the matching source compound supports the same historical family, with daughter first member noted in Turner6481.'),
    ('f_klkufo444jm6c',
     'Dhakal p. 69 prints /dʰi.ja.pu.ta/ “children/heir.” Turner8266 gives Bihari dhiyāputā “sons and daughters.” The heir extension is compatible with the source’s kinship compound; the daughter first member is traced separately by Turner6481.'),
]

accepted, audit = [], []
review = ['# Darai: children compound', '',
          'Three source-backed records of a compound explicitly attested in Turner8266.', '',
          '| # | Darai form | Proposed etymology | Evidence |', '|---|---|---|---|']
source_dir = Path('data/other/forms/raw_data/dhakal_darai_2011/decoded')
for i, (id_, evidence) in enumerate(groups, 1):
    r = rows[id_]
    assert id_ not in linked and r['Source'].startswith('dhakal2011darai['), id_
    page = int(r['Source'].split('[p. ')[1].split(']')[0])
    source_text = (source_dir / f'p{page:03}.txt').read_text()
    assert r['Gloss'] in source_text, (id_, page, r['Gloss'])
    audit.append(dict(id=id_, form=r['Form'], gloss=r['Gloss'], source=r['Source'], page=page,
                      sourceText=source_text))
    accepted.append(dict(record=rec(r), parent='8266', kind='reflex',
                         citation=r['Source'] + ';CDIAL[8266]', evidence=evidence))
    review.append(f'| {i} | **{r["Form"]}** ‘{r["Gloss"]}’ ({r["Source"]}) | Bihari **dhiyāputā** in CDIAL **8266** | {evidence} |')

holds = []
for form, reason in [
    ('bʰəŋna', 'Dhakal pp. 53 and 67 give “nephew,” but homepage searches for the likely bhāgina forms returned no exact Turner head; do not choose a kinship parent by resemblance alone.'),
    ('ʦʰegri', 'Dhakal p. 68 gives “goat,” but searched chʰegrī/chʰāgrī spellings returned no exact Turner comparator; branch remains uncertain.'),
    ('pʰokso', 'Dhakal p. 59 gives “lung”; the exact homepage hit is already under a Dravidian etymon d4452. That does not establish the immediate Darai transmission path.'),
]:
    for r in rows.values():
        if r['Form'] == form and r['Source'].startswith('dhakal2011darai[') and r['ID'] not in linked:
            holds.append(dict(record=rec(r), reason=reason))

assert len(accepted) == 3 and len(holds) == 4, (len(accepted), len(holds))
(P / 'pass140-decisions.json').write_text(json.dumps(dict(accepted=accepted, held=holds), ensure_ascii=False, indent=2) + '\n')
(P / 'pass140-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(audit), rows=audit), ensure_ascii=False, indent=2) + '\n')
review += ['', 'Three affected records and three reflex rows. The parent article itself records the regional compound; Turner6481 explains the daughter member. Held: two *bʰəŋna* “nephew,” *ʦʰegri* “goat,” and *pʰokso* “lung.” Earlier held Darai forms remain excluded. No lexical-source, identity or build changes.',
           'Primary source: Dhakal 2011 printed pp. 53, 67, 69, visually checked in source PDF images and decoded text. Discovery: darai-search056; complete CDIAL articles darai-primary052.',
           'Validation: pending.']
(P / 'pass140-review.md').write_text('\n'.join(review) + '\n')
print(len(accepted), 'accepted', len(holds), 'held')
