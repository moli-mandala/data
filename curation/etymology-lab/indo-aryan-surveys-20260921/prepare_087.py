import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
a=[];h=[]
for r in rs:
 if r['ID'] in done or r['Gloss']!='broken':continue
 f=r['Form'];l=r['Language_ID']
 if l in ['Mai','Chil','Gowro'] and f in ['śɛrɛlo','śarɛl','śārɛl','śarīl','śɛrɛl']:
  a.append(dict(record=r,parent='12328',kind='reflex',citation='CDIAL[12328]',evidence='Turner’s first *śarati branch explicitly gives Chiliss/Gowro šar- break. These Chiliss, Gowro and neighbouring Maiya broken-state responses continue that stem with ordinary l-participial endings and local vowel variation. They are distinguished from the second *śalati rot branch.',family=0))
 elif f in ['tūtgāīyyā','tutʰgel','ṭuṭīga','ṭuṭago']:
  a.append(dict(record=r,components=['6065','4008'],citation='CDIAL[6065];CDIAL[4008]',evidence='The complete broken-state response is tūṭ-/ṭuṭ- break plus gay-/gel-/go gone. Turner truṭyati lists the Hindi, Bhojpuri, Awadhi and western Pahari break verb and its broken participles; gata supplies the second member. Both members are represented despite fusion in the transcription.',family=1))
 elif l=='awan' and f=='trotī':
  a.append(dict(record=r,parent='6065',kind='reflex',citation='CDIAL[6065]',evidence='Turner truṭyati explicitly gives Lahnda truṭaṇ and ṭuṭṭaṇ break, Punjabi taruṭṭṇā and broken-state participles. The Awan broken response trotī preserves the diagnostic tr- of that western verb family with an ordinary feminine past ending.',family=2))
 elif f in ['śɛrga','cʰīnā','cīrdū','cirdū','cʰīrdū','cīsān','puṭ / ṭuṭ','ṭūṭo / payo','ṭūṭ / pāy','tuṭyā oḍ','tuṭioḍ']:
  h.append(dict(record=r,reason='Broken: retain for review of the full response. Dardic śɛrga has an unresolved ending; chīn-/cīr-/cīs- require distinguishing chindati/chinna, chidyate and cīrayati with local morphology/contact evidence; slash responses contain different roots; oḍ in the Kaithal response remains unanalyzed.'))
(P/'pass087-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held')
