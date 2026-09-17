import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass118';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='5300',citation='CDIAL[5300]',evidence='CDIAL 5300 jyotis gives Prakrit jōi light/fire and Oriya joe/joi/jui fire. Bhatri and Adivasi Oriya joy/zoi/jai fit this eastern oi-family, retaining affricate/fricative and vowel notation. These forms are compared specifically with the Oriya series; local Indo-Aryan transmission remains unresolved.'),dict(parent='55',citation='CDIAL[55]',evidence='CDIAL 55 agni gives eastern āgi/āg and regional āgī fire. Selected āīg/aⁱg/aig retain the source placement of the high vowel before the velar, while Dang aghi retains aspiration notation. These are qualified regional agni-family matches; no source spelling correction or precise immediate Indo-Aryan donor is asserted.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='fire':continue
 i=0 if r['Language_ID'] in {'Bhatri','AdivasiOriya'} and r['Form'] in {'joy','zoi','jai'} else 1 if r['Form'] in {'āīg','aⁱg','aig','aghi'} else None
 if i is not None:
  q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
 elif r['Language_ID']=='Hajong' and r['Form']=='džui':held.append(dict(record=r,families=[],reason='Full CDIAL 5300 gives Assamese zūi fire under jyotis, while 6606 addendum gives Assamese jui (phonetic z-) fire under dyuti. Hajong džui could continue or borrow either competing family; similarity and uncertain cross-IA transmission do not distinguish the etyma. Needs additional primary historical evidence.',passNumber=118))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc),len(held))
