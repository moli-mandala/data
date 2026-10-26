import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
('13845',['ɸūṭīyɔ','ɸuṭelu','ɸuṭalo','pūtlata','pūtɨla','pʰūtlata','pūṭlata','pʰatla','foṭāl','pʰoṭo','poṭāl','poṭālā','pʰoṭɛl','pʰoṭālo','poṭ','fut / put','put / pʰut'],'Turner *sphuṭyati explicitly records the regional phuṭ-/phūṭ- break/burst verb, including Oriya phuṭibā, Gujarati phuṭvũ and Marathi phuṭṇẽ. These broken-state responses are its ordinary participial/past forms, retaining regional deaspiration and l-past or -ta endings. Both alternatives in the selected paired responses express the same verb family.'),
('9353',['baṇgila','bʌṇla','baŋgla','baŋglaṭa','bʰaŋglaṭa','bʰaŋglata','baŋglī','bʰaŋlata','bʰŋgla ta','baŋglata','bʰaŋgla','baŋgise','baganī','bagɛsɛ','baŋgī'],'Turner bhaṅga explicitly gives Bengali bhāṅā to be broken, Oriya bhāṅgibā to break and Gujarati bhā̃gũ broken to pieces. These eastern and western broken-state forms are compared with that nasal bhaṅg- family, with ordinary past/stative suffixes and local reductions preserved. They are kept distinct from the unnasalized bhagna branch of bhajyate.'),
('9363',['bātsla','bā̃tsla','bʰāceko','bʰā̃cila','bʰā̃cili','bā̃cen','bʰā̃cela','bʰācelo','bacal'],'Turner *bhañjati explicitly records Nepali bhā̃cnu break and neighbouring Kumauni bhā̃cṇo. These Nepal/Tharu broken-state responses fit that affricate family, including ordinary -ko/-l- participles and the survey’s ts notation; they are not merged with the distinct nasal-velar bhaṅga family.')]
compgroups=[
(['13845','4008'],['phuṭīgeyo','pʰūtīgɔla','pūṭī gala','pʰūṭgaya','phūṭigayo'],'The whole completed-action phrase is break/burst + gone, phuṭ-/phūṭ- plus gay-/gal-. Turner *sphuṭyati and gata provide both lexical members; regional inflection and fusion are retained.'),
(['9353','4008'],['bɔŋgīgala'],'The full phrase is the eastern bhaṅg- break verb plus gal-/gone. Turner bhaṅga and gata supply the members, with the ordinary eastern participial construction represented in full.'),
(['9361-2','4008'],['bhāgī gyū','bāgīgyo','bagīgayo'],'The complete western broken-state phrase is bhāg-/broken + gone. Turner’s bhagna branch explicitly includes Gujarati bhāgũ broken and bhāgvũ break; gata supplies gay-/gy-. This is distinguished from the nasal bhaṅg- stem and from the related runaway sense.'),
(['9353','1031'],['bəṇgla ace'],'The full eastern phrase is bhaṅg- broken, with its l-past form, plus ach-/ace present copula. Both the bhaṅga verb family and the ākṣeti copula family are linked.')]
a=[]
for i,(parent,forms,evidence) in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss']=='broken' and r['Form'] in forms:a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}]',evidence=evidence,family=i))
for i,(components,forms,evidence) in enumerate(compgroups,start=len(groups)):
 for r in rs:
  if r['ID'] not in done and r['Gloss']=='broken' and r['Form'] in forms:a.append(dict(record=r,components=components,citation=';'.join(f'CDIAL[{p}]' for p in components),evidence=evidence,family=i))
assert len(a)==len({x['record']['ID'] for x in a})
(P/'pass085-decisions.json').write_text(json.dumps(dict(accepted=a,held=[]),ensure_ascii=False,indent=1)+'\n')
(P/'pass085-rules.json').write_text(json.dumps(groups+compgroups,ensure_ascii=False,indent=1)+'\n')
print(len(a),[(i,sum(x['family']==i for x in a)) for i in range(len(groups)+len(compgroups))])
