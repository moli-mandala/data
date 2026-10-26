import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[{'parent': '4822', 'gloss': ['ant'], 'forms': ['cīmṭā', 'cimṭā', 'cīmaṭ', 'cimta', 'cimti'], 'citation': 'CDIAL[4822];Platts[s.v. cimṭā, cimṭī]', 'evidence': 'Turner connects cimṭā pincers and the pinch/cling verbs with *cimb. Platts explicitly gives cimṭī both pincers and ant and cross-refers the larger-ant cīṅṭā to cimṭā; this supplies the otherwise missing insect sense. The Nepal survey cimṭ-/cimt- forms are compared within this lexical family, without extending the analysis to the separate ciũṭ-/cuṭ- series.'}, {'parent': '3193', 'gloss': ['ant'], 'forms': ['kīṛo'], 'languages': ['Goj'], 'citation': 'CDIAL[3193]', 'evidence': 'Turner’s kīṭa insect family includes Punjabi/Hindi/Gujarati kīṛī ant, beside kīṛā/kīṛo insect. Gujari kīṛo ant matches the same stem with a masculine ending and the documented narrower insect sense; this does not force a morphological analysis of the longer Bhili kiḍāvi forms.'}]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms'] and ('languages' not in g or r['Language_ID'] in g['languages']):
   a.append(dict(record=r,parent=g['parent'],kind='reflex',citation=g['citation'],evidence=g['evidence'],family=i))
assert len(a)==len({x['record']['ID'] for x in a})
h=[]
(P/'pass040-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass040-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g['parent'],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
