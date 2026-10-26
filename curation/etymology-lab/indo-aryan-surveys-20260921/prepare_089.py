import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
a=[];h=[]
for r in rs:
 if r['ID'] in done:continue
 if r['Gloss']=='all' and r['Form'] in ['baddū','badha','badhā','bɛddā','baddā','bhaddɛ','badɛ']:
  a.append(dict(record=r,parent='9126',kind='reflex',citation='CDIAL[9126]',evidence='Turner baddha bound explicitly gives western Pahari bado/bāddhe/baddhɔ all and Gujarati bādhũ whole/entire. The Jaunsari and Bhil/Bareli all responses fit this bound-together > whole/all family, with local vowels, gemination and aspiration retained. This is a comparative family assignment; neighbouring contact may also have contributed to its distribution.',family=0))
 elif r['Gloss']=='all' and r['Form'] in ['bāḍõ','bāḍẽ','bāḍḍe','bāḍe']:
  h.append(dict(record=r,reason='All: the retroflex bāḍ- forms may be related to the baddha all/whole family (9126), but retroflexion and possible overlap with baḍ- big require local evidence. Dental badd-/badh- forms are distinguished for now.'))
 elif r['Gloss']=='cloth' and r['Form'] in ['cīttarā','cītrā','cittro','cītra','sītrā','cīttaro','citrɔ']:
  a.append(dict(record=r,parent='4802',kind='reflex',citation='CDIAL[4802]',evidence='Turner *citth tear records both extended cīthṛā/cithṛũ rag and the r-extension Gujarati cithrũ/cithrī rag. These Malvi/Nimadi cloth responses are compared with that specific rag/cloth family, preserving local deaspiration and palatal/dental sibilant spellings. Nasal-containing cīntr-/chindr- forms are excluded because their history may involve chinna or contamination.',family=1))
 elif r['Gloss']=='cloth' and r['Form']=='odʰene':
  a.append(dict(record=r,parent='2547',kind='reflex',citation='CDIAL[2547]',evidence='Turner *ōḍḍh wear/cover explicitly includes Awadhi/Bhojpuri oṛhanā mantle and the Braj uṛhnā̃ clothes addendum. The Bagheli odhene cloth response fits that garment formation, retaining its source transcription. Turner treats the wider family as an old Dravidian loan; the comparison is to the established Indo-Aryan family, not a newly asserted immediate Dravidian donor.',family=2))
(P/'pass089-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(i,sum(x['family']==i for x in a)) for i in range(3)])
