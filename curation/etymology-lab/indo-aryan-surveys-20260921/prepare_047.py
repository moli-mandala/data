import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[{'parent': 'f_7tuydfvqt7ici', 'kind': 'borrowed', 'gloss': ['body'], 'forms': ['badn'], 'languages': ['poth'], 'citation': 'platts1884[p. 141, badan body]', 'evidence': 'Platts distinguishes Arabic-origin badan body (p. 141) from the separate Sanskrit-origin badan face (p. 140). Pothwari badn is the syncope of this regional Hindustani body word. The existing Hindi/Urdu badan entry represents the likely contact donor provisionally; Punjabi mediation cannot be excluded, and no direct Arabic borrowing is asserted.'}, {'parent': 'f_oa25zzj3txsdu', 'kind': 'borrowed', 'gloss': ['body'], 'forms': ['vayūd'], 'languages': ['awan', 'Goj'], 'citation': 'platts1884[s.v. wujūd]', 'evidence': 'Platts gives Hindustani wujūd body/person, an Arabic-origin loan. The SSNP Hindko/Gujari transcription uses y for its affricate j, as independently checked against the source, so survey vayūd corresponds to vajūd. The existing Hindi/Urdu donor is a provisional regional contact-language proxy; Punjabi mediation is possible and the edge does not skip to Arabic.'}, {'parent': 'f_22utxbhjyd5lu', 'kind': 'borrowed', 'gloss': ['onion'], 'forms': ['pɛad̯ ʒ', 'pīajī', 'pyāj', 'pyadz', 'pyāz', 'piyaji', 'pʸæj', 'pʸej'], 'languages': ['B', 'AdivasiOriya', 'kaithal', 'jaun', 'kul', 'Dang', 'Kathoriya'], 'citation': 'platts1884[s.v. piyāz];kannauji[p. 73]', 'evidence': 'Platts identifies Hindustani piyāz onion as Persian, and Jambu already attests the regional Hindi pyāj donor. Survey pyāj/pyāz and vocalic or affricated variants fit this widespread loan family. Hindi/Urdu is used as a provisional contact-language proxy rather than claiming direct Persian transmission; Bengali/Oriya or neighbouring Pahari mediation may intervene, and final i is retained as regional adaptation.'}]
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
(P/'pass047-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass047-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g.get('parent',g.get('components')),sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
