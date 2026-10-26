import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
 dict(parent='10930',gloss=['cloth'],forms=['laṭā','laṭṭā'],citation='CDIAL[10930,1]',evidence='Turner’s first latta branch explicitly gives Awankari lattā clothes, Punjabi lattā rag, Kumaoni lātā/lattā/latā clothes and Hindi lattā tattered cloth. The Kathoriya and Haryanvi laṭ(t)ā responses match this cloth/rag family with regional retroflexion. The r-expanded lattara/latthara branches are not selected.'),
 dict(parent='5452',gloss=['cloth'],forms=['talle','tala'],languages=['dog','Goj'],citation='CDIAL[5452 and addenda]',evidence='Turner gives Doda ṭalla clothes and Western Pahari ṭalo cloth, alongside Nepali ṭālo piece of cloth. The adjacent Dogri/Gujari tal(l)- survey responses fit this cloth family; the source dental spelling is preserved, so this is explicitly a regional lexical comparison rather than a claimed exact phonetic match.'),
 dict(parent='3808',gloss=['cloth'],forms=['kʰɔdī'],citation='CDIAL[3808,1]',evidence='Turner gives Bengali khādi small piece of cloth, Nepali khādi rough cloth and Hindi/Gujarati/Marathi khādī coarse cotton cloth in the first khadda branch. The Adivasi Oriya khɔdi cloth answer fits the eastern short form; the r-bearing khaddara branch is excluded. This comparative placement does not decide possible later diffusion of the cloth term.'),
 dict(parent='8857',gloss=['stone'],forms=['pətʰəṛə'],languages=['Or'],citation='CDIAL[8857]',evidence='Turner explicitly lists Oriya pathara stone under prastara. The survey pəthəṛə is the same regional form with the source’s central vowels and rhotic. The entry separately marks several Punjabi-mediated Hindi/Bihari forms as loans; that instruction is not extended to this explicitly listed Oriya reflex.'),
 dict(parent='8857',gloss=['stone'],forms=['pʰatar'],languages=['srk'],citation='CDIAL[8857]',evidence='Turner gives Lahnda pathar and Sindhi patharu stone under prastara. Saraiki phatar matches that northwestern series with aspiration on the initial stop in this survey. This is a qualified regional comparison; it is not automatically treated as one of the Punjabi loans Turner specifically identifies elsewhere.'),
 dict(parent='8857',gloss=['stone'],forms=['pətəṛ'],languages=['kul'],citation='CDIAL[8857 and addenda]',evidence='Turner explicitly records Western Pahari Kotgarhi pátthər and Jaunsari pāthar stone. Kullui pətəṛ matches this neighbouring prastara family, with reduced vowels and no aspiration marked in the survey; its exact phonetic development remains a comparative qualification.'),
 dict(parent='3965',gloss=['stone'],forms=['gaṭā','gaṭī','gatʰī'],languages=['awan','poth','Goj'],citation='CDIAL[3965,1 and addenda]',evidence='Turner’s gaṭṭa piece family includes Punjabi gaṭṭ/gaṭṭā and, explicitly in the addenda, Western Pahari gɔṭṭɔ and gaṭi small stone/pebble. The Awankari, Pothwari and Gujari stone responses match that northwestern lexical family. i-vowel giṭṭ- forms and responses combining different words are excluded.'),
]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms'] and ('languages' not in g or r['Language_ID'] in g['languages']):
   a.append(dict(record=r,parent=g['parent'],kind='reflex',citation=g['citation'],evidence=g['evidence'],family=i))
assert len(a)==len({x['record']['ID'] for x in a})
h=[]
for r in rs:
 if r['ID'] not in done and r['Gloss']=='cloth' and r['Form'] in ['gābā','gābbā','gāba','gābɔ','ghābā','gaba','gabbā','gabba','gabbɔ','gʰabbā','gʰābā']:
  h.append(dict(record=r,reason='Cloth gābā/gabbā requires separating gabba filling (CDIAL 4023) from garbha interior (4055), whose Marathi gābhā also denotes cloth under a turban in Molesworth and Aryabhushan. The meanings overlap but do not establish which history explains the unaspirated Rajasthan survey family; hold pending regional lexical evidence.'))
(P/'pass038-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass038-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g['parent'],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
