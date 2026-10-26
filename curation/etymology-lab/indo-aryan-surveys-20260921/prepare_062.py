import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[{'parent': '10048', 'gloss': ['man', 'person'], 'forms': ['man'], 'languages': ['Hajong'], 'citation': 'CDIAL[10048]', 'evidence': 'Turner explicitly cites Haijong māna and Mymensingh Bengali mān man under mānava. The Hajong man survey forms directly match that unusually specific regional evidence.'}, {'parent': '10048', 'gloss': ['man', 'person', 'husband'], 'forms': ['manay', 'manæ', 'manaī', 'manæyā', 'manāĩya', 'manaⁱ', 'manāⁱ', 'mānā̃ⁱyā'], 'languages': ['H', 'bundeli_atarra', 'Kathoriya', 'Dang'], 'citation': 'CDIAL[10048]', 'evidence': 'Turner explicitly derives Bhojpuri/Awadhi manaī man and Hindi manaī man/husband/bridegroom from the *mānavika extension of mānava. These central and Terai forms match that documented manaī family, including nasalized and inflected variants; they are kept distinct from mānuṣa/mānis.'}, {'parent': '10049', 'gloss': ['man', 'person'], 'forms': ['māṇas', 'māṇos', 'manūs', 'maṇuś', 'mūṇəś', 'mõṇəś', 'manəś', 'māntźʰe', 'mānse', 'māⁱśa', 'māis', 'maṇu', 'məṇu', 'mũṇu', 'mõṇə', 'monc'], 'citation': 'CDIAL[10049 and addenda]', 'evidence': 'Turner gives Hindi/Gujarati mānas/māṇas, Nepali mānis/mānche, Sirmauri māṇaś and Kotgarhi maṇu/maṇch under mānuṣa. The survey forms match these central and Pahari regional comparanda, including the explicitly documented nś > ñch development and shortened Kotgarhi maṇu type.'}, {'parent': '13735a', 'gloss': ['woman'], 'forms': ['trɛm', 'trīmat', 'trīmatʰ'], 'languages': ['Goj', 'awan'], 'citation': 'CDIAL[13735a]', 'evidence': 'Turner explicitly lists Lahnda trīmat, Multani trīmit/trēm- and Punjabi tarīmat woman/wife under *strīmātrī. Awankari trīmat directly matches; Gujari trɛm matches the documented Multani reduced trēm- stem. Mixed elicitation responses are excluded.'}]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms'] and ('languages' not in g or r['Language_ID'] in g['languages']):
   x=dict(record=r,citation=g['citation'],evidence=g['evidence'],family=i)
   if 'components' in g:x['components']=g['components']
   else:x.update(parent=g['parent'],kind=g.get('kind','reflex'))
   a.append(x)
assert len(a)==len({x['record']['ID'] for x in a})
h=[{'record':r,'reason':'Majhi māncek/māncək husband likely contains the Nepali mānche person stem (CDIAL 10049), but the final k and exact derivation have not been verified; held rather than treating the whole form as ordinary inflection.'} for r in rs if r['ID'] not in done and r['Language_ID']=='Majhi' and r['Form'] in ['māncek','māncək'] and r['Gloss']=='husband']
(P/'pass062-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass062-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g.get('parent',g.get('components')),sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
