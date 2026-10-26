import json
from pathlib import Path
P=Path(__file__).resolve().parent;rs=json.loads((P/'remaining-initial.json').read_text());d={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())};a=[];h=[]
langs={'H','Kannauji-Central','Marw','bagri_fatehabad','bundeli_atarra','dhundari_badagaon','dhundari_bamore','kaithal','had','hadothi_kelwada','P','mewati_akera','marwari_bagra','Brj','Rana','Dang','Kathoriya','Sunha','awan','bagheli_lakshman','Bhilali','Bhili','Nimadi','dog','Goj','mewari_basad','Noiri','Vasavi'}
forms={'bā','bo','bɔ','vo','vɔ','vā','ve','bɛ','be','o','ū','u','ũ','vī','ō','ɔ'}
for r in rs:
 if r['ID'] in d:continue
 f,g=r['Form'],r['Gloss']
 if r['Language_ID'] in langs and f in forms and g!='that u' and all(v.strip().split(' ')[0] in ['he','she','they','that','those'] for v in g.split(';')):
  a.append(dict(record=r,parent='972',kind='reflex',citation='CDIAL[972]',evidence='Turner asau explicitly supplies Old Marwari vo, Hindi wah, Punjabi/Lahnda o and Nepali u as remote demonstratives and third-person forms. These western and central survey paradigms are compared with that family, preserving ordinary b/v alternation and local gender/number inflection. Only third-person or distal-labelled responses are included; this/these responses and extra lexical material are excluded.'))
 if f in ['bāste','bāstī','bāsti'] and g=='fire':
  h.append(dict(record=r,reason='The cached Bagri original explicitly lists bast̪e/bast̪i under item 55 fire (printed p. 34), so this is not silently treated as the for postposition. A historical parent for the fire word remains unestablished; do not force the homonymous vāste loan or a bellows etymology.'))
(P/'pass108-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n');print(len(a),len(h))
