import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())};a=[];held=[]
for r in rs:
 if r['ID'] in done:continue
 if r['Gloss']=='mother' and r['Form'] in ['māī yā','mɔi̯yā','ma u','mayā','māye']:
  a.append(dict(record=r,parent='10016',kind='reflex',citation='CDIAL[10016]',evidence='Turner mātṛ explicitly gives Hindi maiyā, Assamese māu and the māya/māi family. The surveyed mother forms match these ordinary regional continuations; the source-internal spacing of māī yā and ma u is preserved and read as the attested kin term, not two independently inherited lexical members.'))
 if r['Gloss']=='hammer' and r['Form'] in ['muṭla','mutla','mūṭlā']:
  held.append(dict(record=r,reason='DEDR 4932 gives exact Pengo, Manda and Kuwi muṭla hammer in the muṭ- family, but the immediate transmission route into these Indo-Aryan survey varieties is unresolved. Do not force an inherited mudgara link or invent a Gondi donor.'))
 if r['Gloss']=='mother' and r['Form'] in ['aya','aia','āyā','yā','aīyā','aīā','ayyā','āīyā','āyyā']:
  held.append(dict(record=r,reason='Compare CDIAL 997 *āī nursery mother/aunt, but āryikā and regional Dravidian aya kin terms supply competing histories. A specific inherited or borrowed route is not established for this response.'))
(P/'pass101-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=1)+'\n');print(len(a),len(held))
