import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922');rs=json.loads((P/'kasargod-records.json').read_text());assigned={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'};assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
('āgi','fire, flame.','55','agni',137,'Turner gives Old Marathi āgi and Marathi/Konkani āg fire, through MIA aggi. The source ā:gi on p.137 supports this assignment.'),
('āti','hand.','14024','hasta',137,'Turner gives Marathi hāt and Konkani hātu hand, through hattha. The source ā:ti retains the long vowel and t with initial h absent; this is the regional hand family, with the source ending preserved.'),
('kɛ̄̃sɨ','hair.','3471','keśa',143,'Turner gives Marathi kes/kẽs hair under keśa. The nasalised front vowel and s match this family; p.143 and the existing image-backed correction confirm the vowel notation.'),
('gā̃u','village.','4368','grāma',144,'Turner gives Marathi gā̃v/gāv and Konkani gā̃vu village under grāma. The scan confirms gã:u, preserving the nasal vowel and vocalic ending.'),
('jībi','tongue.','5228','jihvā',147,'Turner gives Konkani jīb(h)a and Marathi jībh beside Prakrit jibbhā. Source ǰi:bi on p.147 supports the b-series tongue family; the existing transcription profile renders the affricate as j.'),
('dānti','tooth.','6152','danta',151,'Turner gives Konkani dāntu tooth under danta. The source da:nti retains nt and a final vowel; this is not the dantī plant entry or an extended dental stem.'),
('nāki','nose.','6909','*nakka',152,'Turner gives Marathi nāk nose under *nakka. The scan confirms na:ki, supporting the long-vowel k series with a retained final vowel.')]
a=[]
for form,gloss,parent,label,pg,ev in groups:
 for r in rs:
  if r['Form']==form and r['Gloss']==gloss and r['ID'] not in assigned:a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}];'+r['Source'],evidence=ev+' Source page image inspected. Existing spelling, dialect label and OCR-review tag remain unchanged.'))
reasons={'kāṇni':'The p.141 scan appears to read kāṇi (ka:ṇi); installed kāṇni has an extra n. Hold pending source transcription repair, outside this overlay-only pass.','pāṇi':'Merged water and leaf records require different etymologies; do not assign this mixed node to pānīya or pattra.','duy ṭi':'The p.151 scan shows a nasal consonant in the smoke reading; installed spacing and consonants require transcription review.','dū du':'The p.151 scan has a single lexical word du:du. Hold the installed split form pending transcription review.','gīri':'The p.144 scan shows central-vowel notation; do not infer the ghara relationship from the installed gīri without resolving the reading.'}
held=[dict(record=r,reason=reasons[r['Form']]) for r in rs if r['Form'] in reasons and r['ID'] not in assigned]
(P/'pass015-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=2)+'\n')
lines=['# Marati of Kasargod: image-checked etymologies','',f'{len(a)} records / {len(a)} reflex links. Source is catalogued as Marathi (M), dialect Marati of Kasargod. No rebuild or transcription repair.','','| # | Marati of Kasargod | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(f,g,pa,la,pg,ev) in enumerate(groups,1):
 la=la.replace('*',r'\*');status=' (already linked)' if not any(x['record']['Form']==f for x in a) else '';lines.append(f'| {i} | **{f}** ‘{g.rstrip(".")}’{status} | **{la}**, CDIAL **{pa}** | {ev} |')
lines+=['','Primary scan: ../../../../output/pdf/ghatage-1970-marati-of-kasargod.pdf, printed pp.137, 141, 143–144, 147, 151–152 (PDF page = printed + 8). Images inspected at 1600-pixel scale. Existing image-backed corrections corroborate hair, village and tongue.','',f'{len(held)} records held:']+[f'- **{x["record"]["Form"]}**: {x["reason"]}' for x in held]+['','Validation and save status: pending.']
(P/'pass015-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(held),'held')
