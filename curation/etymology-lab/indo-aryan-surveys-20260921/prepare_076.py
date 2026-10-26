import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
('4209',['gūrākā','garvʰo','garho','garuvā','garum','gʰaruŋga','goru','goruo','goreko','garhu'],'Turner guru explicitly gives Kalasha gūrak heavy, Jaunsari garūo, Nepali garũo/garuṅo/garaũ/garũ and the Kotgarhi gɔrkɔ extension. The unprefixed survey forms match this family, with local vowel reduction, inflection and nasal extensions preserved. The Danuwar/Bote variants are a qualified comparison with the adjacent Nepali family, without claiming that their precise contact history is resolved.'),
('1962',['āŋgūrākā','aŋgūrūk','uŋgūro','ogū','ūxūr','ūgʰūrū','hūgūrū','hāgūrū','hāgūra','ūgʰūr'],'Turner *udguru explicitly gives Kalasha agūrak, Bashkarik ugūr, Torwali ūgū, Palula úguro/úṅguro and Shina Kohistani hăgúro heavy. These prefixed northwestern forms fit that family rather than unprefixed guru. The local aspiration/frication and initial h are retained; Turner himself marks the explanation of emphatic h as uncertain.'),
('11465',['boc','bɔj','bʰɔj','boj','bojʰ','bajʰ','bʌjʰ','boz','bʰoj'],'Turner vahya gives Prakrit vojjha burden and Oriya bojha, Bengali bojhā and Hindi bojh load, with crossing from voḍhum. The survey uses this burden/weight word adjectivally for heavy. These are qualified regional family comparisons; Turner explicitly identifies Hindi diffusion of the noun into several western languages, so uninterrupted local inheritance is not asserted.')]
a=[]
for i,(parent,forms,evidence) in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss']=='heavy' and r['Form'] in forms:
   a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}]',evidence=evidence,family=i))
assert len(a)==len({x['record']['ID'] for x in a})
(P/'pass076-decisions.json').write_text(json.dumps(dict(accepted=a,held=[]),ensure_ascii=False,indent=1)+'\n')
(P/'pass076-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),[(g[0],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
