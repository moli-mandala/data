import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text())
done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
 dict(parent='6065',gloss=['broken'],forms=['tutla','tutala','tutyæ','tuteko','tutʰlā','tutāl','tuṭulo','ṭūṭlū̃','ṭūṭelā','tuṭeḍo','tuṭʰā','tutiyɛ','ṭuṭiyɛ','ṭʰūṭnā','trūṭā','ṭūṭeā','ṭūṭnɔ'],citation='CDIAL[6065]',evidence='Full truṭyati includes Prakrit tuṭṭai/ṭuṭṭai and the associated tuṭṭa “broken”, with Lahnda/Punjabi ṭuṭṭ-, Nepali ṭuṭnu, Bihari ṭūṭal and Hindi ṭūṭā. These whole survey responses retain that break stem with regional infinitive or participial morphology (-eko, -la/-lo, -yā/-eā); the dental/retroflex and aspiration notation is preserved. Responses containing go/be auxiliaries, a negative particle, or two stems are excluded.'),
 dict(parent='9361',gloss=['broken'],languages=['poth'],forms=['pajā','pajīā','pajnā'],citation='CDIAL[9361,1]',evidence='The bhajyate article explicitly gives Lahnda bhajjaṇ and Punjabi bhajjṇā “be broken/flee”, contrasting the separate g-retaining bhagna branch. These Pothwari paj- broken/infinitive responses match the former stem with the regional voiceless outcome of initial voiced aspiration; the j is retained. This is a bhajj- family analysis, leaving the original loss of aspiration/tone realization at the survey’s recorded level, and it does not cover the SSNP y-spelled forms awaiting font checks.'),
 dict(components=['13902','10452'],gloss=['sleep!, he slept'],forms=['so jā','sūijā','soija','soijā','hūija','hūvijā','hūvijū','hūijā','hūvijo'],citation='CDIAL[13902];CDIAL[10452]',evidence='Turner’s svapati article explicitly unites the coalescing so-/su-/sov- sleep stems and lists Hindi sonā and Gujarati sūvũ; yāti supplies the jā-/jo go auxiliary. The whole affirmative response is analysed synchronically as sleep + go in that order, with ordinary imperative vowels and western s-to-h outcomes preserved. The compound segmentation is an analysis of these survey expressions, not a claim that Turner prints each expression; no additional tense auxiliary or negative is omitted.'),
 dict(components=['13902','4008'],gloss=['sleep!, he slept'],forms=['sūgyɔ','sɔgayɔ','hūigyo','sūigīo','sogayo','soygyo','soigayo','sogayā','sogāyo','hūigya','sūvigayo','hūvigoi-nū'],citation='CDIAL[13902];CDIAL[4008]',evidence='The complete expression consists of the so-/su(v)- sleep stem (Turner svapati) followed by the gayo/gyo/gai regional perfect auxiliary (Turner gata). Both lexical components are represented in their surface order, retaining h for s and ordinary finite ending variation. This is a modern compound-verb analysis from the whole response; forms with ambiguous standalone -ga, an extra particle, or a second elicited answer are excluded.'),
 dict(components=['9361-2','10452'],gloss=['run!, he ran'],forms=['bʰāgjā','bʰagjā'],citation='CDIAL[9361,2];CDIAL[10452]',evidence='Turner’s bhagna branch explicitly gives Hindi bhāgnā and neighboring g-retaining run verbs. These whole imperatives are bhāg + jā “run away”, with yāti as the go auxiliary; the ordered components distinguish this g-stem from the bhajj- branch. The segmentation is inferred from the complete modern expression, without adding a claim about its regional borrowing route.'),
 dict(components=['9361-2','10948'],gloss=['run!, he ran'],forms=['bʰāglɛ'],citation='CDIAL[9361,2];CDIAL[10948]',evidence='The whole imperative is the g-retaining bhāg run stem from Turner’s bhagna branch plus le “take” from the lāti family, used as a light verb. Both components are included in order; the source ɛ vowel is retained and the link does not treat the full predicate as one inherited stem.'),
 dict(components=['9361-2','4008'],gloss=['run!, he ran'],forms=['bʰagīgāyo'],citation='CDIAL[9361,2];CDIAL[4008]',evidence='The full past expression has bhāg run (Turner bhagna branch) followed by gāyo gone (gata), with the linking vowel retained. Both lexical pieces are represented as ordered components; the analysis does not confuse the go auxiliary with part of the inherited run stem.'),
]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms'] and ('languages' not in g or r['Language_ID'] in g['languages']):
   it=dict(record=r,citation=g['citation'],evidence=g['evidence'],family=i)
   if 'components' in g:it['components']=g['components']
   else:it.update(parent=g['parent'],kind='reflex')
   a.append(it)
held=[]
for r in rs:
 if r['Gloss']=='fly!, it flew' and r['Form'].startswith(('ūḍ','uḍ','ūṛ','ūd')):
  held.append(dict(record=r,candidates=['1697-2','1697-5'],reason='Full CDIAL 1697 Addenda 5 explicitly proposes *uddayati as an alternative to uḍḍayate for potentially all section-2 forms representing ḍ or ḍḍ. These survey fly responses do not distinguish the historical sources. Auxiliaries can often be segmented, but the lexical-parent ambiguity remains; do not assign a section merely from the main-entry parsed comparator.'))
assert len(a)==len({x['record']['ID'] for x in a})
(P/'pass018-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
(P/'pass018-decisions.json').write_text(json.dumps({'accepted':a,'held':held},ensure_ascii=False,indent=1)+'\n')
print(len(a),sum(len(x.get('components',[])) or 1 for x in a),[(g.get('parent',g.get('components')),sum(x['family']==i for x in a)) for i,g in enumerate(groups)],'held',len(held))
