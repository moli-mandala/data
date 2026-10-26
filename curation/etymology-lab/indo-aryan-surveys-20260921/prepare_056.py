import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[{'parent': '6914-2', 'gloss': ['fingernail'], 'forms': ['nakh', 'nuk', 'nok', 'nɔk', 'nɔkh', 'nak', 'nɔkhe', 'nokhaḍo', 'nākhaṛī', 'nakhḍā', 'nɔkhaṛɔ', 'nɔkhɔḍɔ', 'nakhḍiyā', 'naūkhaḍayā', 'nakhaḍyo', 'nakhḍīyū', 'nakhaḍyū', 'nokh', 'nākh', 'nāk', 'nɔg', 'nag'], 'citation': 'CDIAL[6914,2]', 'evidence': 'Turner distinguishes the retained-velar *nakkha branch from simple nakha with lost intervocalic k: explicit comparanda include Bashkarik nakh, Torwali nōkh and western Pahari nekhər/nègər. These survey nakh/nokh forms and western diminutives retain that diagnostic velar; the eastern forms are qualified members of the same retained-velar family, without excluding learned reinforcement. Nasal nāṅg-type compounds and Persian nākhun are excluded.'}, {'parent': '9758-2', 'gloss': ['fish'], 'forms': ['macʰī', 'maccī', 'macʰːī', 'maci', 'macī'], 'languages': ['poth', 'srk', 'kul'], 'citation': 'CDIAL[9758,2 and addenda]', 'evidence': 'Turner explicitly places Lahnda Awankari macchī, Khetrani machī and Kotgarhi/Kiunthal macchi fish in the second *matsiya branch. The Pothwari/Siraiki and Kullui survey forms match those specific Lahnda and western Pahari comparanda. Gujari and other macchī forms are excluded here because Turner also lists feminine macchī under the main matsya branch.'}, {'parent': '2574', 'gloss': ['what?'], 'forms': ['ka', 'kã'], 'languages': ['jaun'], 'citation': 'CDIAL[2574,1 addenda]', 'evidence': 'Turner’s addendum explicitly lists Jaunsari kā and Kiunthal ka what/why under the interrogative ka stem. These Jaunsari ka/kã responses match that direct regional comparison; other short ka/ke forms are not automatically equated across languages.'}, {'parent': '3164', 'gloss': ['what'], 'forms': ['gī'], 'languages': ['Chil', 'Gowro'], 'citation': 'CDIAL[3164]', 'evidence': 'Turner explicitly gives neighbouring Maiya gī what under kim. The Chilisso and Gowro gī responses match this established Kohistani interrogative development; it is distinct from the neighbouring kā̃ who series under ka.'}]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms'] and ('languages' not in g or r['Language_ID'] in g['languages']):
   x=dict(record=r,citation=g['citation'],evidence=g['evidence'],family=i)
   if 'components' in g:x['components']=g['components']
   else:x.update(parent=g['parent'],kind=g.get('kind','reflex'))
   a.append(x)
assert len(a)==len({x['record']['ID'] for x in a})
h=[]
(P/'pass056-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass056-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g.get('parent',g.get('components')),sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
