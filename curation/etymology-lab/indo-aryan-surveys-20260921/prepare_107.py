import json
from pathlib import Path
P=Path(__file__).resolve().parent;rs=json.loads((P/'remaining-initial.json').read_text());d={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())};a=[];h=[]
for r in rs:
 if r['ID'] in d:continue
 f,g=r['Form'],r['Gloss'];x=None
 if g.startswith('we '):
  if f in ['hamsab','ami sʌbu','ame sɔbū']:
   x=dict(components=['986','13276'],citation='CDIAL[986];CDIAL[13276]',evidence='This we expression consists of the regional ham/ami/ame pronoun plus sab/sabu all. Turner explicitly supplies both families under asmad and sarva. Both components are represented in surface order; the survey’s inclusive label is retained.')
  if f in ['ham sāre','ham sārā']:
   x=dict(components=['986','13355'],citation='CDIAL[986];CDIAL[13355]',evidence='The whole response is ham we plus sārā/sāre all, compared with Turner asmad and sāra respectively. The latter is the documented whole/all adjective, not the distinct sarva > sab family.')
  if f=='ham sagaḷe':h.append(dict(record=r,reason='The sakala article 13066 does not itself attest the retained-g sagaḷe family; verify its exact historical formation before adding the whole we-all expression.'))
  if f=='amū badha':
   x=dict(components=['986','9126'],citation='CDIAL[986];CDIAL[9126]',evidence='The response is amū we plus badha all. Turner asmad supplies the western pronoun family; baddha explicitly gives Gujarati bādhũ whole/entire. Both members are represented rather than reducing the phrase to the pronoun alone.')
  if f in ['amɛ duy','ame duyai','am duyi','ham duy','hem ḍuy','as dɔuɛ','hem ḍɔnɔ','hem ḍono','hem ḍunɔ']:
   x=dict(components=['986','6648'],citation='CDIAL[986];CDIAL[6648]',evidence='The whole response combines a we pronoun from the asmad family with two/both from dvau, whose article supplies regional dui/do and don formations. Source retroflex spelling and emphatic/case endings are retained; both ordered lexical components are represented. More elaborate expressions containing people or a third word are excluded.')
  if f in ['apūn','āpen']:
   x=dict(parent='1135',kind='reflex',citation='CDIAL[1135]',evidence='Turner ātman first branch explicitly records Gujarati āpaṇ and Old Marwari āpa as inclusive we, beside related reflexive forms. These western survey pronouns are compared with that reflexive-to-first-person family; they are not assigned to asmad by English gloss alone.')
 if x:a.append(dict(record=r,**x))
(P/'pass107-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n');print(len(a))
