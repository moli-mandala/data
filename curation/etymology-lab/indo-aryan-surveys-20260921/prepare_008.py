import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text())
done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
dict(components=['9871','4008'],gloss=["don't die!, he died",'he died','he dies; he died','(he) died','He died.','Die! / (He) died','die'],forms=['margyɔ','margɔ','mari go','marigayo','mar-gīyo','mar-gīyū','marīgiyo','margayo','margo','marīgyo','margiyo','marigayā','marīgo','margaū','maragɔ','margayā','mar gya','maragu','maragao','marigāo','marge','margīā','margīa','marīga','marīgā','marga','mar gyā','margyā','margyo','mar gayā','mar gaye','mar gel','mar-giyo','mar-giyu'],citation='CDIAL[9871];CDIAL[4008,1];platts1884[s.v. marnā, mar-jānā]',evidence='The complete response contains mar(i) “die” plus the perfective go auxiliary, in that order. CDIAL 9871 supplies the regional mar stem (its deeper Vedic history remains debated); 4008,1 gives gaya/gaa, Hindi gayā, Gujarati gyau/giu and the extended gel forms. Platts explicitly records mar-jānā “die”. The links identify both members of the regional compound, not an inherited Sanskrit compound. Paired elicitation glosses are retained; selected responses contain the affirmative past form, with no unrepresented negative word. Mixed responses and unclear additional material are excluded.'),
dict(components=['10066','6140-3'],gloss=["don't kill!, he killed",'he killed'],forms=['mārdīyɔ','mār dīyɔ','mārdyo','mār diyɔ','mārdyɔ','mārdīyā','mardīyo','mardiyā'],citation='CDIAL[10066];CDIAL[6140,3];platts1884[s.v. mārnā, mār-denā]',evidence='This is mār “kill” followed by diyā/diyo “gave” as a perfective auxiliary. CDIAL 10066 supplies the causative kill stem, while specifically 6140,3 *dita supplies Hindi diyā and Kumaoni/Nepali diyo; the contracted dyo retains the same auxiliary. Platts explicitly lists mār-denā as an intensive of mārnā. Both ordered component families are represented; the source paired gloss does not imply that a negative particle occurs in these affirmative responses.'),
dict(components=['3865','10948'],gloss=['eat!, he ate','eat!','Eat! / he ate'],forms=['kʰālɛ','kʰāle','kʰālɔ','kʰālo','kʰāllɔ','kʰāleɔ','kʰālyɔ','kʰālīɔ'],citation='CDIAL[3865,1];CDIAL[10948];platts1884[s.v. khānā, khā-lenā]',evidence='The response is khā “eat” plus le/lo or liyo “take” as auxiliary. Platts explicitly records khā-lenā; CDIAL 3865 supplies the bare eating stem, and 10948 gives le with Old Marwari perfect liyo and explains the analogical interaction of take, give and lead. Both ordered components are retained, including the imperative/past inflection of the auxiliary. Mixed slash responses and longer unanalysed auxiliaries are excluded.'),
dict(components=['3865','4008'],gloss=['bite!, he bit'],forms=['kʰāgyo','kʰāgyɔ','kʰāggɔ'],citation='CDIAL[3865,1];CDIAL[4008,1];platts1884[s.v. khānā, khā-jānā]',evidence='The bite response contains khā “eat/bite” plus gyo/go “went” as auxiliary. CDIAL 3865 explicitly includes biting as well as eating, and 4008,1 gives the contracted perfect go forms; Platts documents the khā-jānā construction. The compound receives both ordered component links. Forms with intervening y that could represent an independently inflected first member are held for separate analysis.'),
]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms']:
   a.append(dict(record=r,components=g['components'],kind='component',citation=g['citation'],evidence=g['evidence'],family=i))
assert len(a)==len({x['record']['ID'] for x in a})
(P/'pass008-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
(P/'pass008-decisions.json').write_text(json.dumps({'accepted':a,'held':[]},ensure_ascii=False,indent=1)+'\n')
print(len(a),[(g['components'],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
