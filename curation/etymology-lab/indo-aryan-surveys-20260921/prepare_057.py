import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[{'parent': '12349', 'gloss': ['firewood'], 'forms': ['śūlā', 'śalā', 'śelā', 'śɛla', 'śɛlla', 'śɛlā', 'śɛllɔ'], 'languages': ['Kal', 'Mai', 'Tor', 'Bshk'], 'citation': 'CDIAL[12349]', 'evidence': 'Turner explicitly compares Kalasha šula, Bashkarik šala, Torwali šalā and Maiya shalā̃ wood under śalākā stick. These matching regional forms denote fuel wood in the surveys; the vowel variants fit the same directly attested family.'}, {'parent': '12379', 'gloss': ['firewood'], 'forms': ['śākā'], 'languages': ['Phal'], 'citation': 'CDIAL[12379]', 'evidence': 'Turner specifically assigns Palula šāk and šākhā wood/tree to śākhya, belonging to a branch. The survey śākā firewood matches this regional comparison and is kept distinct from the neighbouring śalākā family.'}, {'parent': '6311', 'gloss': ['firewood'], 'forms': ['dāurā', 'dourā'], 'languages': ['Bote', 'Dotyali'], 'citation': 'CDIAL[6311]', 'evidence': 'Turner lists Nepali dāuro firewood as the extended reflex of dāva. Bote dāurā and Dotyali dourā match this Nepali-area fuel term; this is distinct from retained-r dāru wood.'}, {'parent': '6670', 'gloss': ['firewood'], 'forms': ['bāluṇ', 'bālūn', 'bālan', 'bhālan'], 'citation': 'CDIAL[6670]', 'evidence': 'Turner lists Lahnda bāluṇ and Punjabi bālaṇ firewood under *dvālana kindling. The Awankari forms directly match Lahnda; Kaithal bālan and Gujari bhālan are regional members of this same fuel-word family, with the latter aspirated variant qualified.'}, {'parent': '12461', 'gloss': ['spice mortar', 'spices mortar'], 'forms': ['silāuṭā', 'silāuṭi', 'silaṭʰ', 'silat', 'silaṭ', 'silaot', 'silāuṭo', 'silāuto', 'silāᵘṭʰ', 'siloṭʰ', 'siloṭā', 'silaᵘṭo', 'siloṭ', 'silaᵘṭ', 'siloṭa', 'silaᵘṭʰ'], 'citation': 'CDIAL[12461]', 'evidence': 'Turner explicitly gives Kumauni/Nepali silauṭo and Bihari/Hindi silauṭ, siloṭ stone for grinding spices, from śilāpaṭṭa. These Terai and Nepal survey grinding slabs preserve both historical components, including the reduced silaṭ variants; they are not assigned to simple śilā.'}, {'parent': '12459', 'gloss': ['spices mortar'], 'forms': ['silo', 'śilo'], 'languages': ['Dotyali'], 'citation': 'CDIAL[12459,1]', 'evidence': 'Turner explicitly gives Kumauni silo stone for grinding spices under simple śilā. Dotyali silo/śilo matches that nearby comparison and lacks the second element of silauṭo < śilāpaṭṭa.'}]
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
(P/'pass057-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass057-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g.get('parent',g.get('components')),sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
