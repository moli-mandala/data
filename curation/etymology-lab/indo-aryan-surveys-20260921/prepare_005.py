import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text())
done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
dict(parent='9312',gloss=['goat'],forms=['bokaḍī','bokaḍi','bokḍi','bokṛa','bokṛe','bokṛə','bokəṛə','bokəṛu','bukaṛī'],citation='CDIAL[9312]',evidence='Full *bōkka gives Prakrit bokkaḍa, Old Gujarati bokaḍaü, Gujarati bokṛũ, Marathi bokaḍ and Hindi bokrā “goat”. The selected survey forms retain this specific extended stem, with ordinary local gender/number endings and medial contraction or ḍ/ṛ variation; feminine i is analysed, not discarded as unexplained material. The CDIAL node includes these extensions. Turner’s tentative deeper origin and regional transmission remain unresolved.'),
dict(parent='5161-3',gloss=['leech'],forms=['jukā','jõk','jʰõk','jugā','jɔ̃k'],citation='CDIAL[5161,3]',evidence='The full jalūkā article separates l-retaining jalūkā/jalaukā from section 3 *jōkkā/*jōṅkā. The selected forms have the contracted k/g stem of section 3: compare Nepali juko, Oriya juka and Bengali/Hindi jõk. Final a/o are ordinary nominal endings; voicing, aspiration and nasalization variants are retained. Responses containing separate words or unexplained -kuvā material are excluded.'),
dict(parent='4847',gloss=['cold (water)'],languages=['DewasDoneDanuwar','Majhi','Bote'],forms=['ciso','ciśo'],citation='CDIAL[4847]',evidence='Turner’s *cīssa article explicitly gives Nepali ciso “cold”, despite its reconstructed head meaning “sharp pain”. These neighbouring Nepal survey adjectives agree in form and exact sense; ciśo preserves the source sibilant notation. The link identifies the regional family and does not claim to resolve inheritance versus Nepali-mediated diffusion.'),
dict(parent='9229',gloss=['arm','arm/hand'],forms=['bā̃','bā̤̃','bā̤','bā̃v','bahe','bāhe','bav','bãi'],citation='CDIAL[9229]',evidence='Full bāhu gives Lahnda/Punjabi bā̃h, explicit Awankari bā̀ with plural bā̀vā̃, Kumaoni bā̃(h), and Western Pahari bā́/bái. The selected nasal, breathy and v/h forms fit this arm family, including contraction of intervocalic h and ordinary oblique/plural endings. Source breathiness/nasalization is retained. Forms with the z/j of bāzū and expanded -ṭ/-ṛ stems are excluded.'),
dict(parent='7012-2',gloss=['new'],languages=['Bhili','Bhilali','PalyaBareli','PauriBareli','RathwiBareli'],forms=['navlo','nɔvlɔ','ṇāolo','naūlū','naūla','nāvlɔ','nāvlo','nɔvlū̃','navlā','nɔvḷā','navulu','naolo','naoḷa','ṇāoḷa','ṇāulu','nāula'],citation='CDIAL[7012,2]',evidence='Turner’s section 2 *navalla gives Prakrit ṇavalla/ṇavullaḍaya and specifically Gujarati navlũ “new”. The neighbouring Bhil forms preserve the same n-v-l stem or its regular vowel-contracted counterpart and gender endings. This follows the Gujarati comparator in section 2; Kumaoni/Nepali naulo, for which Turner also allows *navala, is deliberately not included.'),
]
acc=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms'] and ('languages' not in g or r['Language_ID'] in g['languages']):
   acc.append(dict(record=r,parent=g['parent'],kind='reflex',citation=g['citation'],evidence=g['evidence'],family=i))
held=[]
for r in rs:
 if r['Gloss']=='fire' and r['Form']=='jui' and r['Language_ID']=='Hajong':
  held.append(dict(record=r,candidates=['5300','6606'],reason='CDIAL lists Assamese zūi/fire under jyotis 5300 and Assamese jui (phonetic z-) under dyuti 6606 in the addendum. Hajong jui fits both after contraction, so the regional match does not resolve which ancient formation it continues.'))
 if r['Gloss'] in ['cold','cold (weather)'] and r['Form'] in ['jār','jar','jāṛ']:
  held.append(dict(record=r,candidates=['5090-2','5180'],reason='Full jaḍa section 2 *jaḍḍa and jāḍya 5180 both give jāṛ-type cold forms. The selected bare forms lack a diagnostic feature to choose the parent; retain this explicit competing analysis.'))
assert len(acc)==len({x['record']['ID'] for x in acc})
(P/'pass005-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
(P/'pass005-decisions.json').write_text(json.dumps({'accepted':acc,'held':held},ensure_ascii=False,indent=1)+'\n')
print(len(acc),[(g['parent'],sum(x['family']==i for x in acc)) for i,g in enumerate(groups)],'held',len(held))
