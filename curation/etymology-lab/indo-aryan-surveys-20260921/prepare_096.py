import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())};a=[]
for r in rs:
 if r['ID'] in done:continue
 x=None
 if r['Language_ID']=='Kal' and r['Gloss']=='hungry' and r['Form'] in ['ānorā','ānārā']:x=dict(parent='299',kind='reflex',citation='CDIAL[299]',evidence='Turner anāhāra explicitly gives Kalasha Rumbur anora and Urtsun anara hungry. These are direct dialect matches; the source forms are retained.')
 if r['Language_ID']=='Kal' and r['Gloss']=='sand' and r['Form'] in ['śīgov','śīgo','śīgūr']:x=dict(parent='f_p6va5kjekybd2',kind='borrowed',citation='CDIAL[13386];trail-cooper1999[printed pp. 283–284, PDF pp. 308–309]',evidence='Turner explicitly marks Kalasha Rumbur śīgou and Urtsun śigər as borrowed from Khowar śuɣur (earlier *śiguδ). The existing Khowar sand entry is the immediate donor; Trail and Cooper confirm śigaw with Urtsun śigur. No direct inherited link to Sanskrit sikatā is substituted.')
 if r['Language_ID']=='Kho' and r['Gloss']=='sand' and r['Form'] in ['śūxūr','śūxur','śuxūr']:x=dict(parent='13386',kind='reflex',citation='CDIAL[13386]',evidence='Turner explicitly places Khowar śuɣur sand, from earlier *śiguδ, in the *śikatā branch of sikatā. The survey x spellings are retained as local transcription variants. The entry discusses Iranian interaction and uncertainty in the broader family, so this is a source-based family assignment rather than a claim of wholly regular development.')
 if x:a.append(dict(record=r,**x))
(P/'pass096-decisions.json').write_text(json.dumps(dict(accepted=a,held=[]),ensure_ascii=False,indent=1)+'\n');print(len(a))
