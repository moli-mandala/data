import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())};a=[];held=[]
for r in rs:
 if r['ID'] in done:continue
 x=None
 if r['Gloss']=='small river' and r['Form'] in ['kʰolā','kʰolo']:x=dict(parent='3943',kind='reflex',citation='CDIAL[3943]',evidence='Turner *khola cavity/hollow explicitly gives Nepali kholo small river/valley and Churahi khoḷ stream in the first branch. The Nepal survey khola/kholo forms match this river-valley family, not the geminate *kholla hollow adjective or *khola cap homonym.')
 if r['Gloss']=='firewood' and r['Form'] in ['dārū','daru','darū','daṛu']:x=dict(parent='6298',kind='reflex',citation='CDIAL[6298]',evidence='Turner dāru piece of wood includes Khowar dar timber/firewood and the wider dār-/daru firewood family. The survey’s Khowar and southeastern daru/daṛu wood terms are compared with that first-branch wood noun, retaining their vowel and r/ṛ transcription. The homophonous medicine/liquor loan and separately listed dāruka extension are not substituted.')
 if r['Gloss']=='cold (weather)' and r['Form'] in ['jār','jā̃r','jāḍi','jāḍ','jāḍo','jar','jaḍ']:x=dict(parent='5180',kind='reflex',citation='CDIAL[5180];CDIAL[5090]',evidence='Turner jāḍya explicitly gives Nepali jāṛo and Bengali/Bhojpuri jāṛ as coldness/winter, with Hindi jāṛā. These weather-cold responses are compared with that coldness noun family; the related jaḍa/*jaḍḍa adjectives overlap formally, so this is a sense-based choice rather than an exclusive phonological proof.')
 if x:a.append(dict(record=r,**x))
 if r['Gloss']=='cold' and r['Form'] in ['jār','jāḍ']:held.append(dict(record=r,reason='Unqualified cold leaves the overlapping jaḍa/*jaḍḍa adjective and jāḍya coldness noun unresolved.'))
(P/'pass099-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=1)+'\n');print(len(a),len(held))
