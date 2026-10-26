import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[{'parent': '65', 'gloss': ['fire'], 'forms': ['ākṭhɔ', 'akṭhī', 'āgṭī', 'ākṭhī', 'ākṭī', 'ākhṭī', 'ākṭi', 'akṭho', 'ākṭhu', 'ākhṭū', 'āgṭu', 'agṭho', 'aṭhī'], 'citation': 'CDIAL[65]', 'evidence': 'Turner explicitly gives Marathi āgṭhī, āgṭī and ākṭī heap of kindled sticks under agniṣṭha, beside Gujarati ãgīṭhī fireplace and Konkani āgṭī brazier. These adjoining Bhili/Bareli and Nimadi fire words preserve the diagnostic ṭ(h) extension; the broader fire sense follows the attested burning-sticks meaning, rather than simple agni.'}, {'parent': '9480', 'gloss': ['fire'], 'forms': ['bʰā'], 'languages': ['srk'], 'citation': 'CDIAL[9480]', 'evidence': 'Turner explicitly lists Lahnda bhā fire/flame under bhāsa light, with Punjabi bhāh. The Siraiki bhā survey response directly matches that Lahnda fire word.'}]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms'] and ('languages' not in g or r['Language_ID'] in g['languages']):
   x=dict(record=r,citation=g['citation'],evidence=g['evidence'],family=i)
   if 'components' in g:x['components']=g['components']
   else:x.update(parent=g['parent'],kind=g.get('kind','reflex'))
   a.append(x)
assert len(a)==len({x['record']['ID'] for x in a})
h=[{'record':r,'reason':'Hajong/Bishnupriya jui/jī fire resembles both CDIAL 5300 jyotis (Assamese zui, Bengali jũi) and 6606 dyuti (Assamese jui in addendum). The competing eastern citations do not resolve the immediate inherited family; held for more specific evidence.'} for r in rs if r['ID'] not in done and r['Gloss']=='fire' and r['Language_ID'] in ['Hajong','Bishnupriya'] and r['Form'] in ['jui','jī']]
(P/'pass059-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass059-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g.get('parent',g.get('components')),sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
