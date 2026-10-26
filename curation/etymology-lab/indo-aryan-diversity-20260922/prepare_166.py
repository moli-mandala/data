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
    ('f_ozpfzh3rhihcc', '4145', 'gāli',
     'Dhakal p. 52 prints /gali/ “scolding, abuse,” contrasting aspirated /gʰali/ “pour.ABS.” Turner4145 gives Nepali gāli “abuse,” Oriya gāḷī “scolding,” and Maithili gārī; the unaspirated Darai word matches this family.'),
    ('f_vn6obkfyz247a', '9225', 'bāṣpāyatē',
     'Dhakal p. 53 prints /bapʰaike/ “to cook by boiling.” Turner9225 gives Sindhi bāphaṇu “to cook by steam” and Gujarati bāphvũ “to boil”; Darai bapʰ- shares the steam-cooking verb, with -aike as local verbal morphology.'),
]

accepted, audit = [], []
review = ['# Darai: scolding and steam cooking', '',
          'Two source-backed reflex records across two etyma.', '',
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
    ('baʦaike', 'Dhakal p. 54 gives “to save,” but Turner11208 has competing §1 and §2 causative paths for bacā-; defer the precise branch.'),
    ('ʣʰanike', 'Dhakal p. 54 gives “to fry”; Turner5363 cites Nepali jhānnu “fry” but marks that comparison with a question mark.'),
    ('ṭʰusike', 'Dhakal pp. 53 and 67 give “thrust,” but ṭʰūs/ṭʰus homepage searches did not find a precise Turner article.'),
]:
    for r in rows.values():
        if r['Form'] == form and r['Source'].startswith('dhakal2011darai[') and r['ID'] not in linked:
            holds.append(dict(record=rec(r), reason=reason))

assert len(accepted) == 2 and len(holds) == 4, (len(accepted), len(holds))
(P / 'pass166-decisions.json').write_text(json.dumps(dict(accepted=accepted, held=holds), ensure_ascii=False, indent=2) + '\n')
(P / 'pass166-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(audit), rows=audit), ensure_ascii=False, indent=2) + '\n')
review += ['', 'Two affected records and two reflex rows. Held: *baʦaike* “save,” *ʣʰanike* “fry,” and two *ṭʰusike* “thrust” records. Earlier held Darai forms remain excluded. No lexical-source, identity or build changes.',
           'Primary source: Dhakal 2011 printed pp. 52–53, visually checked in source PDF images and decoded text. Discovery: darai-search067; complete CDIAL articles darai-primary060.',
           'Validation: pending.']
(P / 'pass166-review.md').write_text('\n'.join(review) + '\n')
print(len(accepted), 'accepted', len(holds), 'held')
