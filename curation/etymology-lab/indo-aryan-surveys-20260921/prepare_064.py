import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[{'parent': '988', 'gloss': ['our', 'my', 'we', 'my; I'], 'forms': ['hāmrā', 'hāmrɔ', 'hamaro', 'hāmro', 'hāmre', 'hamār', 'hammar', 'hāmbro', 'hāmra', 'hāmar', 'hamar', 'hamre', 'hamare'], 'citation': 'CDIAL[988]', 'evidence': 'Turner specifically places Nepali hāmro, Awadhi hamar, Hindi hamārā and Bhojpuri hamār my in the extended asmāka possessive family. These Nepal/Terai hāmr-/hamār- forms match it, including nominative or singular usages in the survey; the Bhojpuri comparison already documents the shift to my.'}, {'parent': '9691', 'gloss': ['my'], 'forms': ['mero', 'morā', 'morɔ', 'moro', 'mora', 'mara', 'mvār', 'mor'], 'citation': 'CDIAL[9691]', 'evidence': 'Turner lists Nepali/Kumauni mero, Assamese mora/mohora, Oriya mohara and Old Awadhi moraü in the possessive series built on ma. These Nepal/Terai mer-/mor- survey forms match that documented adjectival possessive family, including ordinary gender-vowel variants.'}, {'parent': '10511', 'gloss': ['your'], 'forms': ['tamaro', 'taᵘ̃ro', 'tohora', 'tohorā', 'ṭohorā', 'tʰohar', 'tʰɔhar'], 'citation': 'CDIAL[10511 and addenda]', 'evidence': 'Turner specifically gives Kumauni tumaro, Nepali timro, Bhojpuri tohār, Hindi tumhārā and Pahari tuãro in the possessive yuṣmad series. The Dotyali and Terai survey forms match these full tumar-/tohar- comparanda; contracted tor- forms are separated because they also match the singular possessive family.'}, {'parent': '5889', 'gloss': ['your', 'your (low form/singular)'], 'forms': ['ṭorā', 'torā', 'toro', 'torɔ', 'tor', 'tʰor'], 'citation': 'CDIAL[5889,5]', 'evidence': 'Turner explicitly lists Apabhramsha tuhara and Bengali/Oriya tora among possessive derivatives of tuvam, beside Hindi terā. These short tor- singular/low-form Nepal/Terai possessives are assigned to that series; historical overlap with the plural tohar family is acknowledged rather than inferred solely from the generic your gloss.'}]
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
(P/'pass064-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass064-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g.get('parent',g.get('components')),sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
