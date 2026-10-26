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
    ('f_tootta6hpmo7g', '14084-5', '*hōccʰa-',
     'Dhakal p. 46 prints /hõʦo/ “short.” Turner14084 section 5 explicitly gives Nepali hõco and hoco “low, short,” plus Jaunsari hochā “short.” The nasal vowel and affricate distinguish this branch from the article’s other defective-word sections.'),
    ('f_fvpyvcsr7snyi', '5427-2', 'ṭaṅga-',
     'Dhakal p. 53 prints /ṭaŋgi/ “axe.” Turner5427 section 2 gives Bengali ṭāṅgi “spade, axe,” Oriya ṭāṅgi “battle-axe” and Bhojpuri ṭāṅī “axe”; Darai -gi belongs to the ṭaṅga section rather than section 1 ṭaṅka “chisel.”'),
    ('f_wpw7634mp5u6c', '13774', 'sthū́ṇā',
     'Dhakal p. 55 prints /tʰunʰi/ “pillar of wood” and notes breathy /nʰ/ as a predictable intervocalic feature. Turner13774 gives Old Awadhi thūnī “small pillar,” Bhojpuri thūnhī “prop,” and Pali/Prakrit thūṇā “post, pillar”; these support the Darai wooden-post noun.'),
    ('f_igjmydck6gpyi', '8222', 'pīṭha',
     'Dhakal p. 56 prints /pirʰa/ “low wooden stool” and explicitly treats breathy /rʰ/ as nonphonemic. Turner8222 section 1 gives Nepali pirā, Oriya piṛhā, Maithili pīṛhā and Hindi pīṛhā “stool”; the Darai sense and stem match this section.'),
    ('f_h6j46lcbubzoq', '12130', 'vēṣṭá',
     'Dhakal p. 56 prints /berʰa/ “wall” in the same nonphonemic breathy-/rʰ/ set. Turner12130 gives Assamese ber “wall of house,” Oriya beṛā “wall of house,” and Maithili beṛ “hedge, wall”; the Darai noun corresponds to this enclosure/wall branch.'),
]

accepted, audit = [], []
review = ['# Darai: short, axe, pillar, stool, and wall', '',
          'Five source-backed reflex records across five etyma.', '',
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
    ('kʰabʰo', 'The p. 53 and p. 67 “pillar” tokens resemble Turner13639 skambhá, but the Darai loss of the nasal and medial breathy stop need a firmer correspondence before direct attachment.'),
    ('gʰinṭi', 'The p. 53 and p. 67 “neck” tokens lack an exact Turner article in actual-homepage searches; similarity to “bell” words alone does not establish the etymology.'),
]:
    for r in rows.values():
        if r['Form'] == form and r['Source'].startswith('dhakal2011darai[') and r['ID'] not in linked:
            holds.append(dict(record=rec(r), reason=reason))

assert len(accepted) == 5 and len(holds) == 4, (len(accepted), len(holds))
(P / 'pass121-decisions.json').write_text(json.dumps(dict(accepted=accepted, held=holds), ensure_ascii=False, indent=2) + '\n')
(P / 'pass121-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(audit), rows=audit), ensure_ascii=False, indent=2) + '\n')
review += ['', 'Five affected records and five reflex rows. Held: two *kʰabʰo* “pillar” and two *gʰinṭi* “neck” records. Earlier held Darai forms remain excluded. No lexical-source, identity or build changes.',
           'Primary source: Dhakal 2011 printed pp. 46, 53, 55–56, visually checked in source PDF images and decoded text. Discovery: darai-search048–050; complete CDIAL articles darai-primary045–047.',
           'Validation: pending.']
(P / 'pass121-review.md').write_text('\n'.join(review) + '\n')
print(len(accepted), 'accepted', len(holds), 'held')
