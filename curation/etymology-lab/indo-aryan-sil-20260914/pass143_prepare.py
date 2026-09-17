import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass143';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='6676',citation='CDIAL[6676]',evidence='CDIAL 6676 *dviḥsara explicitly gives Hindi dūsrā, Nepali dosro, Bengali dosar/dosrā and Oriya dosara another/second. Selected dosar/dūsar responses elicited different extend the directly attested another sense; repeated forms are distributive. Final vowels, vowel quantity and repetition are preserved, with local Indo-Aryan transmission unresolved.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='different':continue
 if r['Form'] in {'dūsaredūsare','dūsarīdūsarī','dusʌɾdusʌɾ','dosare','dosar','dusara'}:
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
 elif r['Form'] in {'jūdojūdo','judui','juḍo','juḍa','juḍõ','jūdā','jūdā jūdā'}:
  held.append(dict(record=r,families=[],reason='Platts 378 explicitly identifies Persian judā separate/different and judā-judā various/separately. A Persian lexical donor node was not found in homepage discovery: the linked Kalasha júda points only to a generic Persian category, while BH judā and Khowar judá are unlinked. Do not use CDIAL 5090 cold, or the generic Persian category, as a lexical etymon. A proper donor attestation/node and qualifications for retroflex d and final-vowel adaptation are needed.',passNumber=143))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc),len(held))
