import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[{'parent': '4435', 'gloss': ['husband', 'wife'], 'forms': ['gʰarbalā', 'ghɔrvāḷo', 'gʰarvālā', 'gharbālā', 'gʰarbālā', 'ghɛrvaḷo', 'gharvāḷū', 'ghrvalo', 'gʰarbāli', 'ghɔrvāḷī', 'gʰaravālī', 'gharbāli', 'ghɛvaḷī', 'gʰarvāli', 'ghrvali', 'kɛrɛvālī', 'kɛ̤rvālī', 'kɛrāvālī'], 'citation': 'CDIAL[4435]', 'evidence': 'Turner explicitly lists Sindhi gharavāro husband and Hindi gharwālā householder/husband, gharwālī wife under *gharapāla house-guardian. These complete survey formations preserve both the house base and the vāl-/vāḷ- member, with gender and local gh/k/b/w correspondences retained. The link is to the already analysed historical compound, not a partial house-only etymology; heavily contracted gharālā/kɛrɛlā and mixed responses are excluded.'}, {'parent': '13930', 'gloss': ['husband'], 'forms': ['sā̤ĩ', 'sā̃hi', 'sāhĩ', 'sāĩek'], 'citation': 'CDIAL[13930]', 'evidence': 'Turner gives Punjabi sāī̃, Hindi sā̃ī and Maithili saĩẽ husband under svāmin, via Middle Indo-Aryan sāmi/sāhi. These nasalized survey sãi/sāhi forms fit that documented husband term; final -ek is retained as ordinary survey-context morphology. Less diagnostic sāhek and learned-looking swāmi forms are excluded.'}, {'parent': '7013', 'gloss': ['husband'], 'forms': ['naūrā', 'navora'], 'citation': 'CDIAL[7013]', 'evidence': 'Turner explicitly records Marathi navrā/naurā bridegroom or marriageable young man from *navavaraka newly wed. These western survey husband words extend the same spouse term beyond the wedding, preserving their vocalic forms.'}, {'parent': '9402', 'gloss': ['husband'], 'forms': ['bʰatāras'], 'citation': 'CDIAL[9402]', 'evidence': 'Turner gives Hindi/Bihari/Awadhi bhatār husband from bhartṛ through Middle Indo-Aryan bhattāra. The survey bhatāras has this complete stem plus its recorded final inflection; it is not treated as a learned Sanskrit citation form.'}]
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
for r in rs:
 if r['ID'] not in done and r['Gloss']=='husband' and r['Form'] in ['dhaṇī','dhehṇī','dhɛṇī','deṇī','dhāṇī']:
  h.append(dict(record=r,reason='Turner 6722 groups the owner/master dhani family under dhanin but explicitly offers dhanika under dhanikā for husband. The survey spelling does not distinguish those competing formations, so no unique root is saved.'))
(P/'pass055-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass055-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g.get('parent',g.get('components')),sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
