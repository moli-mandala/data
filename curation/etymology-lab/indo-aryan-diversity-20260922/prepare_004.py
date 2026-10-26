import json,csv
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922')
records=json.loads((P/'varli-records.json').read_text())
assigned={r['Form_ID'] for r in read_assignments()}
assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
(['fire'],['āg'],'55','agni','Turner gives Gujarati āgⁱ and Marathi/Konkani āg fire via the Middle Indo-Aryan aggi family, matching simplification of gn and loss of final i.'),
(['air','wind'],['vārā'],'11497','vātara','Turner section 1 explicitly gives Marathi vārā wind, air, gust and Konkani vāro. Intervocalic t loss accounts for the form; this is not the separate *vātāra section 3.'),
(['moon'],['candɔ','cā̃nd'],'4661','candra','The nominal moon sense is section 2 within the existing unsplit node 4661 (there is no 4661-2 node), with Pk caṃda/caṃdaa, Gujarati cā̃d/cā̃do and Marathi cā̃d. Nasal notation and the Davar final vowel are preserved; the adjectival shining section is not selected.'),
(['water','rain'],['pāni'],'8082','pānīya','Turner gives Gujarati/Marathi pāṇī and broader pāni forms. The addendum explicitly gives water and rain under the same family; the source rain responses therefore retain their own meaning instead of being rewritten as water.'),
(['sand'],['reti'],'10816','retra','Turner lists Gujarati and Marathi retī sand. This is the regional retr-/ret- family; the addendum notes that evidence for historical tr is limited to a few extended forms, so the assignment does not resolve the deeper reconstruction.'),
(['road','way'],['vāṭ'],'11366','vartman','Turner explicitly lists Gujarati/Marathi/Konkani vāṭ path under vartman. As in the earlier adjudicated path family, vartis 11363 remains an explicit alternative for feminine forms; the survey does not settle that historical ambiguity.'),
(['star'],['tāra'],'5798-4','tāraka','Turner assigns the Gujarati tārɔ and Marathi tārā star family to section 4 tāraka, alongside Hindi tārā; this provides the regional analysis for Davar tāra. The older feminine tārā head is kept distinct.')]
a=[]
for glosses,forms,parent,label,ev in groups:
 for r in records:
  if r['Form'] in forms and r['Gloss'] in glosses and r['ID'] not in assigned:a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}];'+('CDIAL[11363];' if parent=='11366' else '')+r['Source'],evidence=ev))
(P/'pass004-decisions.json').write_text(json.dumps(dict(accepted=a,held=[]),ensure_ascii=False,indent=2)+'\n')
lines=['# Davar and Dungar Varli: first diversity batch','',f'Autonomously reviewed under the standing authorization. {len(a)} proposed records and reflex rows; seven etyma. No rebuild.','','| # | Language forms and meanings | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(glosses,forms,parent,label,ev) in enumerate(groups,1):
 rs=[x['record'] for x in a if x['parent']==parent];display=' / '.join(forms);n=len(rs)
 lines.append(f'| {i} | **{display}** ‘{" / ".join(glosses)}’ ({n} records) | **{label}**, CDIAL **{parent}** | {ev} |')
lines+=['','Davar retains Language_ID Bhili, Dungar retains Language_ID Varli and their source dialect tags. No uncertain-tagged record or multiword response is included. Source ownership remains pattanaik-koul2003varli.','', 'Validation and save status: pending.']
(P/'pass004-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted')
(P/'varli-search003-input.json').write_text(json.dumps([dict(gloss='stone',lang='Marathi',source='CDIAL')]))
