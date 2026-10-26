import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922');rs=json.loads((P/'darai-records-current.json').read_text());assigned={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'};assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
(['sor'],['pig'],'13544','sūkara','Turner section1 gives sōr/soru beside MIA sūara and eastern sūar pig. Darai sor belongs to the non-nasal branch, distinct from Nepali sũgur under *sūṅkara section2.'),
(['siŋ'],['horn'],'12583','śṛṅga','Turner explicitly gives Nepali siṅ horn beside MIA siṅga. The source velar nasal and s onset match this family.'),
(['hatʰ'],['hand'],'14024','hasta','Turner gives Bihari/Maithili hāth hand/forearm and Nepali hāt under hasta. Source hath preserves aspiration and does not mark vowel length.'),
(['hor'],['plow','plough'],'14000','hala','Turner gives Bihari/Maithili/Bhojpuri har plough under hala, beside western hɔḷ in the addendum. The source r and o are retained in this regional plough-family assignment.'),
(['badri'],['cloud'],'11567','vārdala','Turner gives Bihari/Maithili bādar cloud, Hindi badrā cloudiness and feminine badlī under vārdala, with feminine derivatives also in the heading. The source badri belongs to this cloud series.'),
(['giddə'],['vulture'],'4233','gṛdhra','Turner section2 gives Prakrit giddha and Nepali giddha vulture. Source giddə is assigned to the animal sense of existing unsplit4233, with deaspiration and vowel notation retained; this is not a claim about greedy gṛddha.'),
(['əsi'],['eighty'],'911','aśīti','Turner section1 gives Nepali asi and Bhojpuri asī eighty. Darai əsi matches the non-epenthetic series; Turner notes irregular phonetic development in these numerals.'),
(['ʦulʰa'],['oven'],'4879','cullī','Turner gives Bihari/Maithili cūlhā and Nepali culo under cullī, with oven explicit in the addendum. Darai ts represents its source affricate; the older Dravidian-origin discussion does not establish a recent donor for this regional form.'),
(['roike','roulə'],['to cry','cried'],'10840','rodati','Turner section1 gives MIA royaï/roaï/rovaï and Nepali runu, past royo, under rodati. The Darai grammar identifies -ike as infinitive and -lə as past; the two cited ro- forms retain those ordinary inflections, rather than being assigned to the noun roda10839.'),
(['dekʰike'],['to see'],'6507-2','*dekṣati','Turner section2 gives Nepali dekhnu and Maithili dekhab under *dekṣati. Darai dekh-ike selects the e-vowel subsection, with the source infinitive ending retained.'),
(['uṭʰlə'],['got up'],'1900','*ut-sthāti','Turner section1 gives Nepali uṭhnu and Bhojpuri uṭhal rise/get up. Darai uṭh-lə has the grammar’s past ending and retroflex ṭh, selecting1900 rather than the reformed dental uth branch1900-2.')]
a=[]
for fs,gs,pa,la,ev in groups:
 for r in rs:
  if r['ID'] not in assigned and r['Form'] in fs and r['Gloss'] in gs:a.append(dict(record=r,parent=pa,kind='reflex',citation=f'CDIAL[{pa}];'+r['Source'],evidence=ev+' Source spelling, phonetic-example citation and Chitwan dialect label are preserved.'))
(P/'pass022-decisions.json').write_text(json.dumps(dict(accepted=a,held=[]),ensure_ascii=False,indent=2)+'\n')
lines=['# Darai: further inherited vocabulary and inflected verbs','',f'{len(a)} records / {len(a)} reflex rows, eleven etyma. Earlier Darai links remain unchanged; already-linked records were excluded. No rebuild.','','| # | Darai | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(fs,gs,pa,la,ev) in enumerate(groups,1):
 la=la.replace('*',r'\*');n=sum(x['parent']==pa for x in a);lines.append(f'| {i} | **{" / ".join(fs)}** ‘{" / ".join(gs)}’ ({n}) | **{la}**, CDIAL **{pa}** | {ev} |')
lines+=['','Grammar support: decoded primary p.75 explicitly glosses -ike INF and -lə PST. Source pp.47–48 give roulə cried and roike to cry; p.53 gives dekhike to see. No separate derivation is inferred for ordinary inflection. Previous sesame/load/stone holds remain unresolved.','', 'Validation and save status: pending.'];(P/'pass022-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted')
