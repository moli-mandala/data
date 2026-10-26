import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922');rs=json.loads((P/'zargari-records.json').read_text())
assigned={r['Form_ID'] for r in read_assignments()};assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
('duy','two','6648','dva','Turner explicitly gives European Romani dui two. Zargari final y writes the same glide outcome of the duv-/dui series.'),
('terin','three','5994-3','trīṇi','Turner section 3 gives European Romani trin three, preserving tr and final n. Zargari terin has a vowel within that initial cluster; neither the trayaḥ nor the *trāyaḥ subsection supplies its n.'),
('iśtār','four','4655','catvāraḥ','Turner section 1 explicitly gives European Romani štar four. Zargari iśtār has a supporting initial vowel before śt; this matches the štar series, distinct from the cār and cōr sections.'),
('pānj','five','7655','pañca','Turner gives European Romani panǰ/panč five; Zargari retains the nasal and voiced affricate of the panǰ variant.'),
('śov','six','12803-2','*ṣuvaṭ','Turner specifically gives European Romani šov in section 2 *ṣuvaṭ. The rounded vowel and retained v distinguish this from the ṣaṣ and *kṣaṭ sections.'),
('deś','ten','6227','daśa','Turner gives European Romani deš ten, directly matching the Zargari sibilant and fronted vowel.'),
('yag','fire','55','agni','The source explicitly compares Sanskrit agni and Hindi āg. Turner gives European Romani yag fire and explains y from the feminine article; this lexicalized initial is not split off as a modern component.'),
('berś','year','11392','varṣa','Turner sense 2 explicitly gives European Romani berš year; the existing node covers both rain and year. The source year sense and retained rś are preserved.')]
a=[]
for form,gloss,parent,label,ev in groups:
 for r in rs:
  if r['ID'] not in assigned and (r['Form'],r['Gloss'])==(form,gloss):a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}];'+r['Source'],evidence=ev))
(P/'pass007-decisions.json').write_text(json.dumps(dict(accepted=a,held=[]),ensure_ascii=False,indent=2)+'\n')
lines=['# Zargari: inherited numerals, fire and year','',f'Autonomously reviewed under the standing authorization. {len(a)} proposed records and reflex rows; eight etyma. No rebuild.','','| # | Zargari | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(form,gloss,parent,label,ev) in enumerate(groups,1):
 label=label.replace('*',r'\*');lines.append(f'| {i} | **{form}** ‘{gloss}’ | **{label}**, CDIAL **{parent}** | {ev} |')
lines+=['','Only atomic numerals are included; compound numerals and loan numerals require their own full analyses.','', 'Validation and save status: pending.']
(P/'pass007-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted')
