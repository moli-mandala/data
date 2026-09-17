import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass127';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='12598',citation='CDIAL[12598]',evidence='CDIAL 12598 śṛṇoti hears explicitly documents Punjabi sunanā, Jaunsari śūṇnõ, Nepali sunnu, Assamese xuniba, Bengali sunā, Oriya suṇibā, Maithili sunab, Bhojpuri sunal and Hindi sunnā. The selected simple hearing responses belong to this suṇ-/sun- family; verbal endings are retained as elicited, without claiming an exact tense or infinitive analysis for each survey spelling. Regional s/h and ś/ç realization, dental/retroflex notation, vowel length and o/u variation remain recorded qualifications. Cross-Indo-Aryan transmission is unresolved and does not prevent the family link.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
selected={
 'Bhatri':{'sunla','sunte'},'Bishnupriya':{'hunani'},
 'Hajong':{'huni','hone','huna','hunik','huniva','hun'},
 'MagahiNepal':{'sūnab','sūnnāī'},'kaithal':{'soṇo','soṇː','soṇ'},
 'jaun':{'çuṇə','çuṇi'},'Goj':{'soṇ'},
 'AdivasiOriya':{'sunla','sunle','sunlani'},'hal':{'sunuk'},'Dang':{'sunːʊ'}
}
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] in remaining and r['Form'] in selected.get(r['Language_ID'],set()) and ('hear' in r['Gloss'].lower() or 'listen' in r['Gloss'].lower()) and 'gold' not in r['Gloss']:
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc))
