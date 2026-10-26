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
    ('f_btfhupzqkexqs', '10636-2', 'ramyà with -ḍ- extension',
     'Dhakal p. 60 prints /ram.rə/ “good.” Turner10636 describes the -ḍ- extension and explicitly gives Nepali rāmro “beautiful, good” and Western Pahari rāmṛā “good”; Darai ramrə has the same regional shape and sense, selecting the established extension node 10636-2.'),
    ('f_yvdw4d255bv3o', '7753', '*padadaṇḍa',
     'Dhakal p. 53 prints /pəĩḍa/ “road.” Turner7753 gives Punjabi paĩḍā “road, distance,” Maithili paĩṛā “road” and Hindi paĩḍā “track, path”; the nasalized Darai road noun closely matches this family.'),
    ('f_kwhdwva5neyda', '7700-2', 'paṭṭa with -kk- extension',
     'Dhakal p. 68 prints /pəṭ.ka/ “a piece of cloth worn round a waist by woman.” Turner7700 gives Hindi paṭkā “loincloth” and Pali paṭṭaka “girdle”; the established -kk- extension node 7700-2 captures the -ṭk- cloth form, while the source specifies the wearer.'),
]

accepted, audit = [], []
review = ['# Darai: good, road, and waist cloth', '',
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
    ('kvari', 'Dhakal p. 66 gives “lamb,” but actual homepage searches for kvarī/kvari found no exact Turner branch.'),
    ('bʰərkʰe', 'Dhakal p. 62 gives “just now,” but bharkhar searches found no precise Turner article; immediate donor and morphological clipping need evidence.'),
    ('koʦra', 'Dhakal p. 68 gives “plate,” but koṭora/katora searches did not yield a precise article matching the Darai affricate shape.'),
]:
    for r in rows.values():
        if r['Form'] == form and r['Source'].startswith('dhakal2011darai[') and r['ID'] not in linked:
            holds.append(dict(record=rec(r), reason=reason))

assert len(accepted) == 3 and len(holds) == 3, (len(accepted), len(holds))
(P / 'pass151-decisions.json').write_text(json.dumps(dict(accepted=accepted, held=holds), ensure_ascii=False, indent=2) + '\n')
(P / 'pass151-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(audit), rows=audit), ensure_ascii=False, indent=2) + '\n')
review += ['', 'Three affected records and three reflex rows. Held: *kvari* “lamb,” *bʰərkʰe* “just now,” and *koʦra* “plate.” Earlier held Darai forms remain excluded. No lexical-source, identity or build changes.',
           'Primary source: Dhakal 2011 printed pp. 53, 60, 68, visually checked in source PDF images and decoded text. Discovery: darai-search061–062; complete CDIAL articles darai-primary055–056.',
           'Validation: pending.']
(P / 'pass151-review.md').write_text('\n'.join(review) + '\n')
print(len(accepted), 'accepted', len(holds), 'held')
