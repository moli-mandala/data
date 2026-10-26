import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text())
done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
dict(parent='2245',forms=['beīṭʰɔ','beyṭʰɔ','betʰɔ','beṭo','baīṭʰɔ','bɛyṭʰ','baīṭo','baīṭho','baīṭṭho','baīṭhalo','bæṭʰo','bæṭʰā','bæṭʰal','bæṭʰ','bɛṭo','bɛīṭh','bɛṭ','bɛt','baṭh','beṭa','baṭno','baⁱṭʰ','bāⁱṭ','baⁱṭ','bāit'],citation='CDIAL[2245,1]',evidence='Full upaviśati includes both the sitting verb and the upaviṣṭa past stem, including Apabhramsha baiṭṭha, Hindi baiṭhā/baiṭhnā, Old Marwari baithā and Gujarati bɛṭho. The selected standalone ṭ(h) forms fit that past-stem-based paradigm with ordinary imperative, participial or infinitive endings. The source’s aspiration and vowel notation is retained. The bas/bes forms, for which Turner also allows contamination or derivation from vasati, and merged son/sit records are excluded.'),
dict(components=['2245','10452'],forms=['beṭ jā','beṭ jāo','beṭjā','beṭ ja','beṭja','beṭʰjā','baīṭjāo','beṭhjā','beṭhjāo','baīṭhjao','beṭhjav','baīṭhjā','baīṭhījao','bæṭjā','bæṭʰjā','beyt ja','beyt jao','baīṭja','baṭhi ja','bɛṭhijao','beṭījā','beṭṭīja','baiṭjā','beṭījāo','baṭijā'],citation='CDIAL[2245,1];CDIAL[10452]',evidence='The complete response is baiṭh/beṭ “sit” plus jā “go” as an auxiliary, in source order. CDIAL 2245 explicitly includes the baiṭh/beṭ stem rebuilt on upaviṣṭa; 10452 gives the regional jā imperative and go verb. The segmentation is a compositional analysis of the survey response, not a claim that Turner lists the full modern construction. Ordinary imperative endings and the conjunctive i of the first member are accounted for. Responses containing another finite clause or unclear additional words are excluded.'),
dict(components=['2245','4008'],forms=['bāīṭgaye','beṭgɔ','bæṭʰagao','baṭṭīgyo','baṭīgo','bayṭgyo','beṭigyo','beṭgāyo','beṭgayo','baṭigayo'],citation='CDIAL[2245,1];CDIAL[4008,1]',evidence='The response contains the baiṭh/beṭ sitting stem plus the gayo/go go-perfect auxiliary. CDIAL 2245 accounts for the first member’s past-stem-based paradigm and 4008,1 for the second member, including regional contraction. Both ordered components are retained; this is a modern auxiliary construction, not a single inherited Sanskrit word. Mixed slash responses and additional unanalysed tense material are excluded.'),
]
glosses=['sit down!','sit down!, he sat down','sit down','sit down; he sat do','sit down/he sat down']
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in glosses and r['Form'] in g['forms']:
   a.append(dict(record=r,**({'parent':g['parent'],'kind':'reflex'} if 'parent' in g else {'components':g['components'],'kind':'component'}),citation=g['citation'],evidence=g['evidence'],family=i))
assert len(a)==len({x['record']['ID'] for x in a})
(P/'pass011-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
(P/'pass011-decisions.json').write_text(json.dumps({'accepted':a,'held':[]},ensure_ascii=False,indent=1)+'\n')
print(len(a),[(i,sum(x['family']==i for x in a)) for i in range(3)])
