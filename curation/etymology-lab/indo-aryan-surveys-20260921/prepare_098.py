import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())};a=[];held=[]
for r in rs:
 if r['ID'] in done:continue
 x=None
 if r['Language_ID']=='Hajong' and r['Gloss']=='foot' and r['Form']=='tʰeŋ':x=dict(parent='5500-2',kind='reflex',citation='CDIAL[5500-2]',evidence='Turner’s *ṭheṅga second branch explicitly gives Bengali ṭheṅg and Oriya ṭheṅga leg. Hajong theŋ foot matches this eastern leg/foot family, with loss of the final stop in the nasal cluster. The separate defective adjective *ṭheṅga under 5501 is excluded.')
 if r['Language_ID']=='Hajong' and r['Gloss']=='tiger' and r['Form']=='bak':x=dict(parent='12193',kind='reflex',citation='CDIAL[12193]',evidence='Turner vyāghra gives Bengali and Assamese bāgh, Bengali also bāg, tiger. Hajong bak is compared with this eastern bāg family with final devoicing; no separate origin is inferred from the survey k spelling.')
 if r['Language_ID'] in ['Hajong','Bishnupriya'] and r['Gloss']=='light (not heavy)' and r['Form'] in ['patala','patla','patol']:x=dict(parent='7736',kind='reflex',citation='CDIAL[7736]',evidence='Turner pattrala, sense 2 thin, explicitly gives Assamese/Bengali pātal and Bengali pātlā. The Hajong and Bishnupriya survey uses shift thin/slender to light in weight; the local a/o and syncope variants fit this eastern family. The adjective is not assigned to an unrelated patra leaf node.')
 if x:a.append(dict(record=r,**x))
(P/'pass098-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=1)+'\n');print(len(a))
