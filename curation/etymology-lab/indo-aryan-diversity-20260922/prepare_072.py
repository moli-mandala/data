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
    ('ənnə', {'grain (generic)'}, '395', 'anna', 'Dhakal p. 61 prints /ən.nə/ with the generic grain gloss. Turner395 section 1 gives Pk aṇṇa food/grain and NiDoc aṁna corn, with later L/P ann grain. The source geminate is meaningful, and section 2 *anda- meat is excluded.'),
    ('ʣʰor', {'liquid/soup'}, '5416', '*jhōla²', 'Dhakal pp. 54 and 67 print /dzʰor/ liquid/soup. Turner5416 gives Maithili jhor soup, Nepali/Bengali jhol juice/soup and Oriya jhoḷa. The source voiced aspirated affricate and final r fit the eastern comparison, which is more specific than a generic liquid match.'),
    ('ḍeli', {'basket'}, '5536-4', '*ḍēlla-', 'Dhakal pp. 53 and 67 print /ḍeli/ basket. Turner5536 section 4 explicitly gives Nepali ḍeli small round basket without a handle; other numbered sections concern lumps, fruitstones or other shapes. The source has the broader basket gloss, which is retained.'),
]
accepted = []
audit = []
review = ['# Darai: grain, soup and baskets', '', 'Five reviewed reflex records across three source-backed families.', '', '| # | Darai form | Proposed etymology | Evidence |', '|---|---|---|---|']
source_dir = Path('data/other/forms/raw_data/dhakal_darai_2011/decoded')
for i, (form, glosses, parent, label, evidence) in enumerate(groups, 1):
    hits = [r for r in rows if r['Form'] == form and r['Gloss'] in glosses]
    assert len(hits) == (1 if form == 'ənnə' else 2), (form, len(hits))
    for r in hits:
        page = int(r['Source'].split('[p. ')[1].split(']')[0])
        source_text = (source_dir / f'p{page:03}.txt').read_text()
        assert source_text, page
        if form == 'ənnə': assert 'grain (generic)' in source_text
        if form == 'ʣʰor': assert 'liquid/soup' in source_text
        if form == 'ḍeli': assert 'basket' in source_text
        audit.append(dict(id=r['ID'], form=form, gloss=r['Gloss'], source=r['Source'], page=page, primaryText=source_text))
        accepted.append(dict(record=rec(r), parent=parent, kind='reflex', citation=r['Source'] + f';CDIAL[{parent}]', evidence=evidence))
    review.append(f'| {i} | **{form}** ‘{" / ".join(sorted(glosses))}’ ({len(hits)} records) | **{label}**, CDIAL **{parent}** | {evidence} |')

holds = []
for form, reason in [
    ('ʦaur', 'Dhakal p. 47 glosses caur only uncooked; Turner4749 means husked rice and has eastern cāur. Source does not explicitly supply rice, so this semantic identification is held.'),
    ('dan', 'Gift matches CDIAL6265 dāna, but source p. 55 minimal-pair context does not distinguish inherited vernacular dān from a learned/modern regional loan.'),
    ('mədʰu', 'The source means bee, while Turner9784 madhu means honey; a bee/honey metonymy or shortening of a compound has not been established.'),
]:
    for r in rows:
        if r['Form'] == form: holds.append(dict(record=rec(r), reason=reason))

assert len(accepted) == 5 and len(holds) == 5
(P / 'pass072-decisions.json').write_text(json.dumps(dict(accepted=accepted, held=holds), ensure_ascii=False, indent=2) + '\n')
(P / 'pass072-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(audit), rows=audit), ensure_ascii=False, indent=2) + '\n')
review += ['', 'Five affected records and five reflex rows. Held: *ʦaur* (1), *dan* (2), *mədʰu* (2), for the semantic or transmission questions above.', 'Primary source: Dhakal 2011 printed pp. 53–54, 61 and 67, checked in the source PDF and decoded text. Discovery: homepage darai-search015–016; complete CDIAL articles and subsections darai-primary010–011. No source, identity or build changes.', 'Validation: pending.']
(P / 'pass072-review.md').write_text('\n'.join(review) + '\n')
print(len(accepted), 'accepted', len(holds), 'held')
