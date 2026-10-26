import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text())
done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
 dict(parent='8996',gloss=['rainbow'],languages=['awan','Goj','srk','poth'],forms=['pīŋg','piŋg','pīŋ','piŋ'],citation='CDIAL[8996]',evidence='Turner gives Lahnda/Awankari pīgh and Punjabi pī̃gh/pī̃g under prēṅkha “swing”. Punjabi University, Punjabi Vocabulary of Common Terms (2007), p. 177, Weather, explicitly uses satraṅgī pīṅgh for rainbow (https://pnarang.weebly.com/uploads/1/4/5/6/14563640/vocab_f.pdf#page=177). The simple regional survey responses retain that swing noun with a rainbow extension; the comparative-family link leaves regional borrowing open. Responses ending in unexplained m/k or containing alternatives are held.'),
 dict(parent='13937',gloss=['duck'],languages=['Hajong','Bishnupriya','B'],forms=['has','haś','aś','hãʃ'],citation='CDIAL[13937]',evidence='The full haṁsa entry gives Bengali hā̃s and Nepali hā̃s “duck”, alongside Assamese hā̃h and other regional reflexes. These short eastern forms match the hās/ās family, with source palatal-s and loss or absence of written nasalization preserved. Regional contact is not resolved by this family link. The longer haŋos forms and aśkorakī are excluded because the additional material needs separate explanation.'),
 dict(parent='9459',gloss=['heavy'],languages=['RathwiBareli','Bhilali','PauriBareli','mewari_basad'],forms=['bhāro','bʰāro','bāro','bharo','baro','bʰārā'],citation='CDIAL[9459]',evidence='The full bhāra article explicitly has Lahnda/Punjabi bhārā “heavy” and other a/o-stem heavy adjectives, distinct from the separate bhārika family. These western survey a/o forms retain the complete bhār/bar stem with ordinary masculine vowel endings and variation in aspiration. The link identifies the bhāra adjective family; ī-ending and internally aspirated bhahr- responses are not included.'),
 dict(parent='f_tfolnhjem4qz6',kind='borrowed',gloss=['white'],languages=['H'],forms=['safed','safet','saphed'],citation='platts1884[s.v. safed]',evidence='Platts explicitly labels safed “white” Persian and records vernacular vowel variants. The existing Persian safēd node supplies the immediate lexical donor for these Hindi attestations; final t and ph for f are retained as survey adaptations. This is a loan analysis, not inheritance from cognate Sanskrit śveta.'),
]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms'] and r['Language_ID'] in g['languages']:
   a.append(dict(record=r,parent=g['parent'],kind=g.get('kind','reflex'),citation=g['citation'],evidence=g['evidence'],family=i))
assert len(a)==len({x['record']['ID'] for x in a})
(P/'pass013-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
(P/'pass013-decisions.json').write_text(json.dumps({'accepted':a,'held':[]},ensure_ascii=False,indent=1)+'\n')
f=P/'lexical-reference-links.json'; links=json.loads(f.read_text());links['s.v. safed']='https://www.rekhta.org/urdudictionary?keyword=safed';f.write_text(json.dumps(links,ensure_ascii=False,indent=1)+'\n')
print(len(a),[(g['parent'],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
