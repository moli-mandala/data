import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass103';assert not (P/(stem+'-decisions.json')).exists()
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
rules=[
dict(parent='2757',citation='CDIAL[2757]',evidence='CDIAL 2757 kaphoṇi gives Prakrit kuhaṇī, Hindi kohnī/kehunī, Nepali kuhunu/kuinu, Bengali kanui, Oriya kahuṇi and Gujarati koṇī elbow. The selected regional kuh-/ko-/keun- forms fit this nasal elbow family with h placement, contraction, vowel order and source dental/retroflex nasal notation retained. The article discusses competing Munda/Dravidian deeper connections; those and local Indo-Aryan transmission remain unresolved.'),
dict(parent='3413',citation='CDIAL[3413]',evidence='CDIAL 3413 kūrpara gives Pali kappara, Prakrit kuppara/koppara, Marathi kopar and Konkani khoppōru elbow. Selected eastern kopor/kopor-like and Khandesi kopura forms fit this r-final elbow family with vowel and aspiration variation retained. This distinguishes them from the nasal kaphoṇi family. The dictionary discusses competing directions of Dravidian contact; no direct donor route is asserted.'),
dict(parent='10286',citation='CDIAL[10286]',evidence='CDIAL 10286 mṛttikā gives Prakrit maṭṭī/maṭṭiā, eastern māṭi and Hindi māṭī/maṭṭī earth, clay. Selected maṭi/māṭi/maṭ̚ṭi soil/clay responses fit this family with source length, gemination and unreleased-stop notation retained. The similarly shaped mārttika earthen-vessel branch is not selected; these targets name soil, not a pot. Local Indo-Aryan transmission remains open.')]
ks={'keunʰi','kaunhi','kəhuni','keūhnī','kuhina','koəni','koə.ni','kʌu̯ni','ko.ni','ko.ɔni','konui','konui ̯','kohoṇi','kuhṇi','kahṇi','kouni','kuyṇi','kʰũni','kʰuni','khuno'}
rs={'kopura','kopor','kʰopor','kopur'};acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 f=r['Form'];g=r['Gloss'];i=None
 if g=='elbow' and f in ks:i=0
 if g=='elbow' and f in rs:i=1
 if g=='soil/clay' and f in {'maṭi','māṭi','maṭ̚ṭi'}:i=2
 if r['Language_ID']=='Hajong' and f=='tilkuni' and g=='elbow':held.append(dict(record=r,families=[],reason='The kuni portion compares with CDIAL 2757, but the initial til- is not explained by that full entry. Resolve whether this is a compound, extension or different lexical family before assigning the whole form.',passNumber=103))
 if r['Language_ID']=='Hajong' and f=='dadi' and g=='chin':held.append(dict(record=r,families=[],reason='CDIAL 6250 explicitly gives Bengali/Oriya dāṛ(h)i beard/chin in the dāṭhikā derivative family. Hajong dadi has medial dental d rather than the compared rhotic/retroflex consonant; verify the source notation or a local sound correspondence and the appropriate derivative node before linking.',passNumber=103))
 if i is not None:
  q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+f+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
f=P/(stem+'-primary-articles.json');a=json.loads(f.read_text());a.update(json.loads((P/'pass103-kurpara-primary-articles.json').read_text()));f.write_text(json.dumps(a,ensure_ascii=False,indent=1)+'\n')
(P/'pass103_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
from collections import Counter
print(len(acc),len(held),Counter(x['parent'] for x in acc))
