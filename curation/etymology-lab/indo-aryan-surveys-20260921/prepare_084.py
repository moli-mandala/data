import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
('2561',['ūśak','ūṣak'],'Turner oṣati explicitly records Khowar uṣák cold beside Dameli oṣé- to be frostbitten. The survey uṣak forms match that cited Khowar word. The apparently unusual burn/cold semantic history is Turner’s explicit comparison, not an inferred equation with warm uṣṇa.'),
('855',['oṣ','oś'],'Turner avaśyā/avaśyāya explicitly records Kalasha oš ice/cold, alongside the wider dew/frost family. These Kalasha cold responses match that exact comparandum; they are kept separate from Khowar uṣák under oṣati.'),
('2819',['kakor','kakər','kakʌr','kʌkəṛ','kakəṛ','kakʌṛ','kakɔr'],'Turner karkara explicitly gives Oriya kākara dew and Punjabi/Lahnda kakkar frost. These eastern kākər/kakor cold responses are compared with that documented dew/frost family, preserving the survey’s adjectival cold use and local vowels. The semantic extension is regional and does not imply a claim about every individual consonant change.'),
('12487',['hīro','hīḷo','hīlo'],'Turner śītala explicitly gives Gujarati śīḷũ cold/stale and Marathi śiḷā cold/stale. These Mewari hīl-/hīr- cold forms are qualified regional comparisons with that ī-vowel family, including s/h and l/r variation; the competing e-vowel śaitala family is not merged into this group.')]
a=[]
for i,(parent,forms,evidence) in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss']=='cold' and r['Form'] in forms:a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}]',evidence=evidence,family=i))
h=[]
forms=['hevo','seḷo','ceḷḷo','śevāīlo','śello','xeḷāḷū','ceḷḷū̃','seḷlo','selḷo','seḷḷɔ','heḷḷū','seḷū','seḷlū','hello','hevavo','helɔ','helo','helaino','hellɔ','helõ','ṣelo','çeyḷo','śeṛlo']
for r in rs:
 if r['ID'] not in done and r['Gloss']=='cold' and r['Form'] in forms:h.append(dict(record=r,reason='Compare śītala 12487 (Gujarati śīḷũ, Marathi śiḷā) and *śaitala 12613 (Marathi śeḷ, Konkani śeḷo, Himalayan śeḷo). The modern e-vowel alone does not safely distinguish inherited vṛddhi from regional vowel lowering; hold pending source-specific phonological evidence.'))
assert len(a)==len({x['record']['ID'] for x in a})
(P/'pass084-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass084-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g[0],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
