import json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='expansion-pass77';assert not (P/(stem+'-decisions.json')).exists()
spec={
'9964':'CDIAL 9964 mahiṣa gives regional mhes/mɔĩś buffalo, Hindi mhaĩs and neighboring maĩś forms. Goj mes/mēs, Kaithal mẽs and Hajong moiś are compared with this buffalo family; vowel, sibilant, nasal and aspiration details remain explicit and intra-Indo-Aryan transmission is open.',
'13161':'CDIAL 13161 saptāha period of seven days supports the pt-retaining Bengali/Hajong śopta week forms as learned Sanskrit vocabulary. Vowel rounding and the possible Bengali intermediary into Hajong are retained without requiring a settled intra-Indo-Aryan route.',
'11242':'CDIAL 11242 vatsara gives Assamese basar and Bengali bachar year in the addenda. Hajong bośor fits this eastern year family; the sibilant and vowels are retained, and Turner himself leaves learned transmission possible.',
'9917':'CDIAL 9917 maśaka gives Bengali maśā, Oriya masā and Maithili mos mosquito. Hajong and Adivasi Oriya mosa match this eastern mosquito family with the source vowels and sibilants retained.',
'1268':'CDIAL 1268 āmra explicitly gives Hindi and Punjabi ambī mango. Kaithal ambi is a direct regional match, with possible intra-Indo-Aryan transmission retained.',
'9051':'CDIAL 9051 phala gives regional phal fruit and Oriya phara, with p-initial reflexes elsewhere. Bhatri pəl/pʌl is compared with this fruit family; loss of aspiration is retained as a regional comparative qualification.',
'4147-3':'CDIAL 4147.3 gāvī gives Hindi and neighboring gāi/gāy cow. These Dang/Kathoriya gaiya-like forms match the already reviewed feminine cow expansion, retaining the final -ya and nasal or raised glide details as regional qualifications.',
'4272':'CDIAL 4272 *goḍḍa gives Nepali goṛo and Hindi goṛā leg. Dang gora repeats the regional leg comparison with plain rhotic articulation retained.',
'10951':'CDIAL 10951 lamba gives Maithili namā long beside regional lambā. Magahi nambā is compared with this eastern n-initial long adjective with mb retained as a regional detail.',
'11813':'CDIAL 11813 *vibhāna gives bihān morning and regional dawn forms ending in i. Bote bihani matches the regional morning family already reviewed for Danuwar; its ending and the deeper vibhāyana alternative noted by Turner remain qualified.',
'7727':'CDIAL 7727 pati explicitly means husband; its vernacular northern forms normally lose t (pai/poi). Kullui pətiː husband retains dental t and is linked as a learned Sanskrit loan, leaving any intra-Indo-Aryan intermediary open.',
'6663':'CDIAL 6663 dvāra gives regional dār door. Pothohari der is compared with this contracted door family with vowel fronting retained; intra-Indo-Aryan transmission remains open.'}
rules=[dict(parent=p,citation='CDIAL['+p.replace('-','.')+']',evidence=e) for p,e in spec.items()];ix={r['parent']:i for i,r in enumerate(rules)};acc=[];held=[]
for x in json.loads((P/(stem+'-candidates.json')).read_text()):
 if len(x['parents'])!=1:continue
 p=x['parents'][0];r=x['record']
 if p in ix:
  q=rules[ix[p]];acc.append(dict(record=r,parent=p,family=ix[p],kind=('borrowed' if p in ['13161','7727'] else 'reflex'),citation=q['citation'],evidence=q['evidence']+' Exact survey form '+r['Form']+' is preserved.'))
 # Unreviewed glyph-sensitive families remain in the research queue, not blanket holds.
 elif p in ['11396','11398','7089-2']:
  reason={'11396':'The retained rś rain noun needs distinguishing varṣa from varṣā and the learned borrowing route; the short-vowel survey spelling does not settle the head.', '11398':'Kathoriya barśat rain fits the barsāt family, but CDIAL 11398 explicitly allows varṣartu instead of or crossed with varṣārātri. Check the preferred branch before extending the prior assignment.', '7089-2':'Adivasi Oriya nasi nose is not sufficient evidence for the same suffix history as Gawri nāsī; verify local nāsā versus nāsikā derivation.'}[p]
  held.append(dict(record=r,families=[],reason=reason,passNumber=77))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'expansion_pass77_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
print({'accepted':len(acc),'held':len(held)})
