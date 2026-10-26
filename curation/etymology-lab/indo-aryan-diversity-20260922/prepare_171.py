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
    ('f_mki6zuibjjozc', '9052-2', '*phara-',
     'Dhakal p. 67 prints /pʰar/ “blade,” the second source token of the form linked from p. 53 in pass161. Turner9052 section 2 gives Punjabi pharhā “blade, nib,” selecting the rhotic *phara- branch over section 1 phal-.'),
    ('f_vijaljk6wdkcu', '6079', 'trōṭáyati',
     'Dhakal p. 67 prints /ṭu.ri.ke/ “break.” Turner6079 gives Bengali tuṛā “to break,” Kumaoni ṭoṛṇo and regional toṛ-/tuṛ- verbs; the Darai rhotic distinguishes this branch from Turner6065 ṭuṭ- “be broken.” The source does not specify transitivity, so the rhotic form is the decisive match.'),
]

accepted, audit = [], []
review = ['# Darai: blade and break', '',
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
    ('rəktarə', 'Dhakal pp. 56 and 67 give “red,” but Turner10539 inherited red comparanda mostly have rat-/rāt-; the Darai -ktar- suggests later learned morphology not directly established by the article.'),
    ('ḍʰĩḍ', 'Dhakal pp. 53 and 67 give “abdomen,” but ḍʰĩḍ/ḍʰiṇḍ searches found no precise Turner article.'),
]:
    for r in rows.values():
        if r['Form'] == form and r['Source'].startswith('dhakal2011darai[') and r['ID'] not in linked:
            holds.append(dict(record=rec(r), reason=reason))

assert len(accepted) == 2 and len(holds) == 4, (len(accepted), len(holds))
(P / 'pass171-decisions.json').write_text(json.dumps(dict(accepted=accepted, held=holds), ensure_ascii=False, indent=2) + '\n')
(P / 'pass171-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(audit), rows=audit), ensure_ascii=False, indent=2) + '\n')
review += ['', 'Two affected records and two reflex rows. Held: two *rəktarə* “red” and two *ḍʰĩḍ* “abdomen” records. Earlier held Darai forms remain excluded. No lexical-source, identity or build changes.',
           'Primary source: Dhakal 2011 printed p. 67, visually checked in source PDF image and decoded text. Discovery: darai-search068–074; complete CDIAL articles darai-primary061–063. The sand token investigated during this search was already linked and is excluded.',
           'Validation: pending.']
(P / 'pass171-review.md').write_text('\n'.join(review) + '\n')
print(len(accepted), 'accepted', len(holds), 'held')
