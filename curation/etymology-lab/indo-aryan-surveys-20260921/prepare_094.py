import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
a=[];held=[]
for r in rs:
 if r['Language_ID']!='Kal' or r['ID'] in done:continue
 g=r['Gloss'];f=r['Form'];x=None
 if g=='woman' and f in ['istrīźā','istīźā']:
  x=dict(components=['13734','5205'],citation='CDIAL[13734];CDIAL[5205];trail-cooper1999[printed p. 132, PDF p. 157]',evidence='Turner explicitly analyzes Kalasha istrīžā woman as a modern compound of strī and jāyā. Trail and Cooper confirm the local woman term. Both components are represented, including the contracted survey variant istīźā.')
 if g=='girl' and f in ['istrīźāgūāk','ṣtrīźāgụ̄ɛk','istrīźgụ̄k']:
  x=dict(components=['13734','5205','3245'],citation='CDIAL[13734];CDIAL[5205];CDIAL[3245];trail-cooper1999[printed pp. 132, 175; PDF pp. 157, 200]',evidence='Trail and Cooper explicitly segment girl as istriža kuạk, with kuạk from *kuḍa and compound allomorph guạk. Turner separately analyzes istriža as strī plus jāyā. The three historical components are recorded in surface order, retaining local contraction and vowel variation.')
 if g=='boy' and f in ['pūrūṣ gūāk','pūrūṣgụ̄ɛk','pūrūṣgụ̄k']:
  x=dict(components=['8289','3245'],citation='CDIAL[8289];CDIAL[3245];trail-cooper1999[printed pp. 175, 255; PDF pp. 200, 280]',evidence='Trail and Cooper explicitly segment puruṣguạk boy as puruṣ kuạk, deriving its members from puruṣa and *kuḍa. Both components are represented in order for the spaced and fused survey forms.')
 if g=='good' and f=='prūṣṭ':
  x=dict(parent='f_twehftzejookk',kind='borrowed',citation='CDIAL[8820];trail-cooper1999[printed p. 252, PDF p. 277]',evidence='Turner explicitly marks Kalasha pruṣṭ good as a borrowing from Khowar pruṣṭ/pruṣṭi/pruṣṭa in front. The existing Khowar head form pruṣṭ is used as the immediate donor, not the remote Indo-Aryan praṣṭha ancestor. Trail and Cooper confirm the recipient form and meaning.')
 if x:a.append(dict(record=r,**x))
 if g=='speak':held.append(dict(record=r,reason='Trail and Cooper explicitly connect contracted ma-/ama- to matrik, but Turner marks the Kalasha verb as borrowed from Kafir; the immediate donor remains unresolved.'))
(P/'pass094-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=1)+'\n');print(len(a),len(held))
