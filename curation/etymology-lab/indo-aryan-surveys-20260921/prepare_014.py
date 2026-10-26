import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text())
done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
 dict(parent='10286',gloss=['soil/ground'],languages=['Hajong','Bishnupriya','B'],forms=['mati','matī','maʈi'],citation='CDIAL[10286]',evidence='The full mr̥ttikā entry gives Assamese, Bengali and Oriya māṭi “earth, soil, clay”, alongside Prakrit mattiyā/maṭṭiā. These eastern soil responses match the complete regional māṭi family, with survey dental t versus retroflex ṭ notation preserved. This family link leaves interlanguage transmission open; Turner’s separate uncertainty over Shina māṭi is not generalized to these eastern forms.'),
 dict(parent='10203',gloss=['ring'],languages=['Vasavi','PauriBareli','RathwiBareli','Noiri','Or','Bhili','Dang'],forms=['munḍi','muḍi','mūndhī','mūḍhi','munḍhi','mũṇḍri','munḍari'],citation='CDIAL[10203]',evidence='Full mudrā lists Prakrit muddiā, Punjabi mundī “ring”, Gujarati mū̃drī, Marathi mudī and Oriya mudi. The selected western/eastern survey mundī/muḍī/mundrī forms preserve that ring family, including source aspiration and optional nasal or r. The ancient head is itself probably an Iranian loan according to Turner; the modern link identifies its later Indo-Aryan continuations and does not erase that qualification. Extra diminutive material and potentially maṇḍala-derived forms are excluded.'),
 dict(parent='8214',gloss=['boy'],languages=['Bhatri','Or'],forms=['pila'],citation='CDIAL[8214]',evidence='Turner explicitly includes Oriya pilā “small; child” under *pilla, with Prakrit pella/pīlua and Bengali pil/children expressions. These adjacent eastern boy forms fit that full child word. The entry derives the older family from Dravidian; this reflex link concerns its later Indo-Aryan continuation, with Oriya-mediated regional transmission left open.'),
 dict(parent='6250',gloss=['chin'],languages=['Hajong'],forms=['dadi'],citation='CDIAL[6250]',evidence='The full *dāṁṣṭra article includes its dāṭhikā/dāḍhiā beard branch and explicitly Bengali/Oriya dāṛ(h)i “beard, chin”. Hajong dadi matches this eastern chin/beard family with the survey d for the medial rhotic/retroflex segment. The link preserves Turner’s qualified expressive/contact discussion and the addendum’s alternative reconstruction; it is not an assertion that the head’s remote history is settled.'),
 dict(parent='6671',gloss=['burn (the wood)'],languages=['poth'],forms=['bāl'],citation='CDIAL[6671]',evidence='The full *dvālayati “kindles” entry gives Lahnda bālaṇ and Punjabi bālaṇā, and the addendum explicitly marks Sindhi bārṇū transitive “to burn”. Pothwari bāl is a transitive imperative with wood as its object, selecting this long-vowel causative family rather than intransitive *dvalati (6654).'),
 dict(parent='6010',gloss=['thread'],languages=['poth'],forms=['tāɣā'],citation='CDIAL[6010]',evidence='Turner’s *trāgga/*tārga/*tāgra article explicitly has Punjabi tāgā/taggā “thread”, also Hindi, Bengali and Nepali tāgā, continuing Prakrit tagga. Pothwari tāɣā preserves the whole word with intervocalic g lenited to ɣ. The reconstruction alternatives remain alternatives within the chosen dictionary family; aspirated thāgā forms are excluded pending comparison with *dhāgga.'),
]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms'] and r['Language_ID'] in g['languages']:
   a.append(dict(record=r,parent=g['parent'],kind='reflex',citation=g['citation'],evidence=g['evidence'],family=i))
assert len(a)==len({x['record']['ID'] for x in a})
(P/'pass014-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
(P/'pass014-decisions.json').write_text(json.dumps({'accepted':a,'held':[]},ensure_ascii=False,indent=1)+'\n')
print(len(a),[(g['parent'],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
