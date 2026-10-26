import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
('10896',['loʦ','lōʦ'],'Turner laghu explicitly includes Khowar lʰoċ light, alongside Kalasha lihoċ. The survey loʦ/lōʦ forms fit this comparandum. Turner marks the historical extension with +?, so the identification is with this documented family rather than a claim that every sound or suffix has been explained.'),
('10896',['xɔḷkɔ','haḷeklū','haḷkū','halkū','olko','olkõ','ālkõ'],'Turner laghu documents the metathesized halk-/haḷk- light family, including Kotgarhi hɔḷkɔ, Marathi haḷkā and regional haluko/halukā. These survey forms are compared with the same family, retaining loss of initial h and local extensions where present. Turner also notes Hindi/Punjabi diffusion into western languages; this family-level link leaves the precise regional route open.'),
('10896',['heru','hoṛo','hōī','haṛo','olo','lo','ola','olā','horo'],'Turner laghu documents the metathesized light adjectives Lahnda hôla/haulā, Punjabi haulā, Hindi harū/haruā and Gujarati haḷvũ. These are qualified comparisons with that regional family, including h-loss and r/ṛ/l variants. The evidence does not establish a separate prehistoric stem for each modern phonetic variant.'),
('2540',['ūcīk'],'Turner *ōccha explicitly gives Kalasha učīk light in weight. The survey’s ūcīk matches that lexical item, with the vowel length as recorded in the survey; it is kept separate from the laghu family.')]
a=[]
for i,(parent,forms,evidence) in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss']=='light' and r['Form'] in forms:
   a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}]',evidence=evidence,family=i))
h=[]
for r in rs:
 if r['ID'] not in done and r['Gloss']=='light' and r['Form'] in ['usaso','usas','ūsas']:
  h.append(dict(record=r,reason='Oriya usāsa relief/ease under CDIAL 1868 competes with the causative adjective uśāsiā lighter under 1866-2. The short survey adjective does not safely distinguish these nominal and causative histories; defer rather than selecting a branch by meaning alone.'))
assert len(a)==len({x['record']['ID'] for x in a})
(P/'pass077-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass077-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g[0],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
