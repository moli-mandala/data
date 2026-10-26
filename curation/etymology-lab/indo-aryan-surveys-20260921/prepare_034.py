import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
 dict(parent='7136',gloss=['near'],forms=['nīhāṛā','nīõ','ṇiḍe','niṛe / nɛḍe','nayre','niyere','ṇiṛe','nɛṛo','neṛā','neṛa','niḍā','nɛṛu','nēḷə','nēḷe','nēḷ','neḍə','nuṛ','nēḍ','neṛə','neṛ','nuḷ'],citation='CDIAL[7136]',evidence='Turner explicitly derives Phalura nihāṛa, Torwali niō, Punjabi/Lahnda neṛe, Hindi nīre and eastern niyar from nikaṭam. The addenda give Western Pahari neḍi and nɛṛu. These survey forms match the regional intervocalic-retroflex/flap series; Kullui lateral variants are included as comparative members of that local series, without proposing a separate sound law. Both Gujari slash variants belong to the same family.'),
 dict(parent='7540',gloss=['below'],forms=['nīsū','nītsū','netsa','netso','nise','niso','nisɔ','niṣe'],citation='CDIAL[7540]',evidence='Turner includes Gujarati nīcε, Oriya nice and Old Marwari nīcaï below in the nīca family, explaining the strengthened cc by influence of opposite ucca. The western Bhili survey s/ts variants and inflected final vowels match this below/low series. Multiword and -l/-ḍ-expanded responses are excluded here.'),
 dict(parent='5731',gloss=['below'],forms=['tolot','tolo'],citation='CDIAL[5731,1]',evidence='Turner explicitly gives Assamese/Bengali tale and Oriya taḷe as below postpositions/adverbs, beside tal/talā bottom. The Hajong tol- locative responses and Oriya tolo belong to that simple tala family; no extra directional noun is being ignored.'),
 dict(parent='6495',gloss=['far'],forms=['dūrāī','dʰū','durut','durāhũ','duras','ḍuṛo','durīhan','dureī̯t','dūrīke','dūrīa','dʰūrke','dūrīk','dʰūrīa','dūrke','duḷ','dəṛ'],citation='CDIAL[6495]',evidence='Turner gives Torwali dū, Phalura dhūro, Western Pahari dūr, Oriya dūra and eastern dur under dūra, including adverbial and adjectival case forms. These single-stem survey responses continue that regional family with locative/adverbial endings; Kullui lateral/vowel and aspirated variants remain explicitly comparative. Responses containing an additional adjective or adverb are excluded.'),
]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms']:
   a.append(dict(record=r,parent=g['parent'],kind='reflex',citation=g['citation'],evidence=g['evidence'],family=i))
assert len(a)==len({x['record']['ID'] for x in a})
h=[]
for r in rs:
 if r['ID'] not in done and r['Gloss']=='whole' and r['Form'] in ['akhlo','ākhlo']:
  h.append(dict(record=r,reason='The whole sense suggests akṣata (28: Gujarati ākhũ), whose ākhlɔ is specifically an uncastrated bull, but akhila (48b) is also whole. The l-bearing Bareli adjective needs local formation evidence before choosing between these histories.'))
(P/'pass034-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass034-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g['parent'],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
