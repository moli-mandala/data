import csv
import json
from pathlib import Path

from etymology_assignments import read_assignments

P = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
selected = [
    ('vēl', 'night', '11625', 'vikāla', 'Turner gives Pashai Lauṛi vyāl, Arang viyōl and Kuṛangal vel for night, explicitly indicating Shumashti wyel is a loan from Pashai. Darrai vēl belongs to this Pashai regional family; its unmarked source form is preserved.'),
    ('ṣoṛ', 'sixteen', '12812', 'ṣōḍaśa', 'Turner section 1 explicitly gives Pashai Darrai ṣōṛ for sixteen. Section 2 has other Pashai c̣hui/xūi forms and section 3 has Arang/Kuṛangal ṣⁱēṛ/ṣē̃; the Darrai form selects the main section.'),
    ('mās', 'forehead', '9926', 'masta', 'The source itself records stem *mask-*. Turner section 1 gives Pashai Wegal maskē forehead beside Lauṛi māsk brain under mastaka; section 2 is a distinct mastra formation. Darrai mās is a shortened member of the *mask- set, with the truncation noted rather than normalized.'),
    ('bēn', 'honey', '9614', '*bhēna', 'Turner explicitly lists Pashai Lauṛi and other dialects bēn honey, and Arang bēᵉn, under *bhēna. The same form in the Darrai source supports a family reflex; Turner calls the deeper Indo-European comparison uncertain.'),
    ('tārā', 'star', '5798-4', 'tāraka', 'Turner section 4 explicitly gives Pashai Darrai tārá star. The section contrasts the tārā head, tā́rakā and *tārikā branches, so the exact Darrai witness determines the narrower node.'),
    ('kaṛek', 'tree', '3228', 'kuṭa', 'Turner explicitly gives Pashai Darrai kaṛék tree, oak beside Lauṛi kuṛā́ tree, under kuṭa. The source gloss only says tree, so the source record is not expanded to oak.'),
    ('ṣe', 'six', '12803', 'ṣaṣ', 'Turner section 1 gives Pashai Darrai ṣε῀ six and related Laghman x̌e and Shutul ṣä. The Darrai source writes ṣe without vowel nasalization; this is retained, and section 1 is chosen over the distinct c̣hā section 3.'),
]

linked = {r['Form_ID'] for r in read_assignments() if r['Status'] == 'accepted' and r['Rank'] == '1'}
linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank'] == '1')
rows = [r for r in csv.DictReader(open('cldf/forms.csv')) if r['Source'] == 'lehr' and r['Language_ID'] == 'Pas']
accepted = []
audit = []
review = ['# Pashai Darrai: seven source-matched families', '', 'Seven reflex records in the Lehr survey; standing autonomous authorization.', '', '| # | Pashai form | Proposed etymology | Evidence |', '|---|---|---|---|']
raw = list(csv.reader(open('data/other/forms/20230705-pashai.csv')))
for i, (form, gloss, parent, label, evidence) in enumerate(selected, 1):
    matches = [r for r in rows if r['Form'] == form and r['Gloss'] == gloss and r['ID'] not in linked]
    assert len(matches) == 1, (form, gloss, len(matches))
    source_matches = [n for n, r in enumerate(raw, 1) if r[0] == 'dar' and r[2] == form and r[3] == gloss and r[7] == 'lehr']
    assert len(source_matches) == 1, (form, gloss, source_matches)
    r = matches[0]
    accepted.append(dict(record={k: r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')}, parent=parent, kind='reflex', citation=f'lehr;CDIAL[{parent}]', evidence=evidence))
    audit.append(dict(id=r['ID'], sourceRow=source_matches[0], form=form, gloss=gloss, parent=parent))
    review.append(f'| {i} | **{form}** ‘{gloss}’ | **{label}**, CDIAL **{parent}** | {evidence} |')

assert len(accepted) == 7
(P / 'pass066-decisions.json').write_text(json.dumps(dict(accepted=accepted, held=[]), ensure_ascii=False, indent=2) + '\n')
(P / 'pass066-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(audit), rows=audit), ensure_ascii=False, indent=2) + '\n')
review += ['', 'Seven affected records, seven reflex rows. Source rows were checked against the installed Darrai `lehr` records, including the source’s shortened *mās* and unmarked nasalization in *ṣe*.', 'Discovery: homepage pashai-search013. Full articles: pashai-primary010, including all numbered subsections. No source/identity edit or rebuild.', 'Validation: pending.']
(P / 'pass066-review.md').write_text('\n'.join(review) + '\n')
print(len(accepted), 'accepted')
