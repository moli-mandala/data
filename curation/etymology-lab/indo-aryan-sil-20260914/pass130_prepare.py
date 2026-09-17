import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass130';assert not (P/(stem+'-decisions.json')).exists()
rules=[
 dict(parent='6141',citation='CDIAL[6141]',evidence='CDIAL 6141 dadāti describes the development of the de- present stem and explicitly supplies Punjabi deṇā, Bengali deoyā, Oriya debā, Maithili deb, Hindi denā and Gujarati devũ. Selected simple de- giving responses preserve their elicited infinitive, finite and regional l-endings; no compound response is reduced to its first verb. The local Indo-Aryan transmission route is unresolved. The unrelated cutting verb at 6257 is not used.'),
 dict(parent='6140-3',citation='CDIAL[6140.3]',evidence='CDIAL 6140 subsection 3 *dita explicitly gives Kumaoni/Nepali diyo and Oriya diā given. The Dotyali diyoᵘ and Adivasi Oriya dia responses are assigned to that specific participial branch, retaining the final offglide in the Dotyali transcription. The survey give label is not treated as proof of a present stem.'),
 dict(parent='684',citation='CDIAL[684]',evidence='CDIAL 684 arpayati delivers up explicitly gives Prakrit appēi hands over and Gujarati āpvũ give/pay. These short western ap/ape give responses support the same family, retaining vowel quantity and inflection. Local Indo-Aryan transmission is unresolved. Past api-/apyo forms are left for distinction from the independently attested arpita branch.'),
 dict(parent='1387',citation='CDIAL[1387]',evidence='CDIAL 1387 ālīyate explicitly gives Prakrit allivaï hands over and Gujarati ālvũ give/pay. The selected Bhil al-/āl- paired simple giving responses match this regional family, with vowel quantity and past endings preserved. This is not an assumed change from the different āp- family; local Indo-Aryan transmission is unresolved.')]
sets=[{'devat','deṇā','dela','delio','dihi','deva','denāī','deb','deba','deuk','dea','dɛa','devna','dena'}, {'diyoᵘ','dia'}, {'ap','ape'}, {'alo, alyu','āl, ālũ','āl, ālyo'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or 'give' not in r['Gloss'].lower():continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
 else:
  if r['Language_ID']=='Hajong' and r['Form'] in {'di','diva'}:
   held.append(dict(record=r,families=[],reason='Hajong di/diva give: CDIAL 6141 includes Assamese diba under dadāti, but the 6365 dīyate addenda independently places Assamese diba (present diye) under the passive-derived family. The short Hajong forms do not resolve that choice. Keep distinct from the accepted deva series; uncertainty concerns competing stem histories, not simply inter-IA borrowing.',passNumber=130))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc),len(held))
